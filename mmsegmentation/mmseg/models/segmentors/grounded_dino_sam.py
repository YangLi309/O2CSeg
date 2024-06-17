import os.path
from typing import List, Optional, Union, Dict, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from torch.nn.modules.module import T
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from copy import deepcopy
import torch.nn as nn

from .base import BaseSegmentor

# Grounding DINO
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

from mmengine.device import get_device

from math import ceil
import pickle
# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)


def _load_sam_model(sam_cfg, sam_checkpoint, device):
    return SamPredictor(sam_hq_model_registry[sam_cfg.sam_encoder_version](checkpoint=sam_checkpoint).to(device))


def _load_dino_model(model_config_path, model_checkpoint_path, device='cpu'):
    dino_args = SLConfig.fromfile(model_config_path)
    dino_args.device = device
    model = build_model(dino_args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    # freeze model
    for param in model.parameters():
        param.requires_grad = False
    model.init_cfg = None
    return model


def _get_grounding_output(model, image, class_set, text_prompt, tokenized, token_spans, box_threshold, with_logits=True):
    outputs = model(image[None], captions=[text_prompt])
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    # filter output
    positive_maps = _create_positive_map_from_span(
        tokenized,
        token_span=token_spans
    ).to(image.device)  # n_phrase, 256

    logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
    all_logits = []
    all_phrases = []
    all_boxes = []
    all_label_idxs = []
    all_conf_scores = []
    for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
        # get phrase
        phrase = ' '.join([text_prompt[_s:_e] for (_s, _e) in token_span])
        # get mask
        filt_mask = logit_phr > box_threshold
        # filt box
        all_boxes.append(boxes[filt_mask])
        # filt logits
        all_logits.append(logit_phr[filt_mask])
        # all_conf_score.append(logit)
        if with_logits:
            logit_phr_num = logit_phr[filt_mask]
            label_idx = class_set.index(phrase)
            for logit in logit_phr_num:
                all_conf_scores.append(logit.item())
                all_phrases.append(phrase + f"({str(logit.item())[:4]})")
                all_label_idxs.append(label_idx)
            # all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
        else:
            all_phrases.extend([phrase for _ in range(len(filt_mask))])
    boxes_filt = torch.cat(all_boxes, dim=0).to(get_device())
    pred_phrases = all_phrases
    return boxes_filt, pred_phrases, all_label_idxs, all_conf_scores


# borrowed from https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/util/vl_utils.py
def _create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Input:
        - tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        - token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                positive_map[j, beg_pos] = 1
                break
            else:
                positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def _build_text_prompt_and_token_span(cat_list, force_lowercase=True):
    """
    Return:
        text_prompt: str
        class2tokenspan: dict
            {
                'dog': [[0, 2]],
                ...
            }
    """

    class2tokenspan = {}
    text_prompt = ""
    for catname in cat_list:
        class_name = catname
        if force_lowercase:
            class_name = class_name.lower()

        tokens_positive_i = []
        subnamelist = [i.strip() for i in class_name.strip().split(" ")]
        for subname in subnamelist:
            if len(subname) == 0:
                continue
            if len(text_prompt) > 0:
                text_prompt = text_prompt + " "
            strat_idx = len(text_prompt)
            end_idx = strat_idx + len(subname)
            tokens_positive_i.append([strat_idx, end_idx])
            text_prompt = text_prompt + subname

        if len(tokens_positive_i) > 0:
            text_prompt = text_prompt + " ."
            class2tokenspan[class_name] = tokens_positive_i
            print(class_name)
            print(tokens_positive_i)

    return text_prompt, class2tokenspan


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.num_context
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        if cfg.use_class_specific_context:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx.requires_grad = True

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

@MODELS.register_module()
class GroundedDinoSAM(BaseSegmentor):
    def __init__(self,
                 dino_cfg: str,
                 dino_checkpoint: str,
                 sam_cfg: ConfigType,
                 sam_checkpoint: str,
                 output_classes: list,
                 bbox_threshold: float,
                 sam_threshold: float,
                 color_palette: list,
                 exp_name:str,
                 dataset: str,
                 visualize_results: bool = False,
                 visualize_sam_masks: bool = False,
                 vis_dir: str = 'vis_dir',
                 visualize_classes: list = None,
                 refined_classes: list = None,
                 keep_background: bool = False,
                 class_refinement: bool = False,
                 refine_class_mapping: dict = None,
                 align_corners: bool = False,
                 class_refined_idx2output_idx: dict = None,
                 class_output_idx2refined_idx: dict = None,
                 kd_training: bool = False,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.output_classes = output_classes
        self.dino_model = _load_dino_model(dino_cfg, dino_checkpoint)
        self.dino_model = self.dino_model.to(get_device())
        self.sam_model = _load_sam_model(sam_cfg, sam_checkpoint, get_device())
        self.bbox_threshold = bbox_threshold
        self.sam_threshold = sam_threshold
        self.dataset_mean = torch.tensor(data_preprocessor.mean).to(get_device())
        self.dataset_std = torch.tensor(data_preprocessor.std).to(get_device())
        self.vis_dir = vis_dir
        self.exp_name = exp_name
        self.dataset = dataset
        self.visualize_results = visualize_results
        self.visualize_sam_masks = visualize_sam_masks
        if self.training:
            mode = 'train'
        else:
            mode = 'test'
        if self.visualize_results:
            self.vis_dir = os.path.join(self.vis_dir, exp_name, mode, self.dataset)
            os.makedirs(self.vis_dir, exist_ok=True)
        self.keep_background = keep_background
        self.color_palette = color_palette
        self.align_corners = align_corners
        self.class_refinement = class_refinement
        if self.class_refinement:
            self.refined_classes = refined_classes
            self.class_refined_idx2output_idx = class_refined_idx2output_idx
            self.class_output_idx2refined_idx = class_output_idx2refined_idx
            self.refine_class_mapping = refine_class_mapping
        else:
            self.refined_classes = output_classes
            self.class_refined_idx2output_idx = None
            self.class_output_idx2refined_idx = None

        self.text_prompt, self.refined_class2tokenspan = _build_text_prompt_and_token_span(self.refined_classes)
        self.token_spans = []
        for refined_class in self.refined_classes:
            self.token_spans.append(self.refined_class2tokenspan[refined_class])
        self.tokenized = self.dino_model.tokenizer(self.text_prompt)

        self.visualize_classes = visualize_classes
        if self.visualize_classes is not None:
            self.vis_class_idxs = []
            for vis_class in self.visualize_classes:
                self.vis_class_idxs.append(self.output_classes.index(vis_class))

        self.kd_training = kd_training
        if self.kd_training:
            self.mask_result_module = nn.Identity()
            self.background_mask_result_module = nn.Identity()

        self.inf_count = 0

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        seg_logits, background_mask = self._forward(inputs, data_samples)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tuple[Tensor, Tensor]:
        batch_size = inputs.size(0)
        batched_logits_map = list()
        batched_background_masks = list()
        for batch_idx in range(batch_size):
            normalized_img_input = inputs[batch_idx]
            data_sample = data_samples[batch_idx]
            # create a logits map which is (C+1, W, H) shape, we add an additional label to indicate background
            # (i.e., non-detected pixel), it could be used for masking loss calculation on a specific pixel
            if self.training:
                logits_map = torch.zeros((len(self.refined_classes) + 1, inputs.size(2), inputs.size(3))).to(get_device())
            else:
                logits_map = torch.zeros((len(self.refined_classes) + 1, data_sample.ori_shape[0],
                                          data_sample.ori_shape[1])).to(get_device())
            boxes_filt, pred_phrases, label_idx_filt, conf_score_filt = _get_grounding_output(
                self.dino_model, normalized_img_input, self.refined_classes, self.text_prompt, self.tokenized,
                self.token_spans, self.bbox_threshold)

            # the next phase only works when there are objects detected
            if boxes_filt.size(0) != 0:
                denormalized_img_input = (normalized_img_input.permute(1, 2, 0) * self.dataset_std) + self.dataset_mean
                denormalized_img_input = denormalized_img_input.type(torch.uint8)
                denormalized_img_input = denormalized_img_input.permute(2, 0, 1).contiguous()[None, :, :, :]
                # if self.training:
                #     H, W = normalized_img_input.shape[-2:]
                # else:
                #     H, W = data_sample.ori_shape[-2:]
                H, W = normalized_img_input.shape[-2:]
                self.sam_model.set_torch_image(denormalized_img_input, (H, W))
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.tensor([W, H, W, H]).to(get_device())
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                transformed_boxes = self.sam_model.transform.apply_boxes_torch(boxes_filt, (H, W)).to(get_device())
                masks, _, _ = self.sam_model.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                    return_logits=True
                )

                # resize the mask in test phase if the expected size is changed
                if not self.training and data_sample.ori_shape != tuple(normalized_img_input.shape[-2:]):
                    H, W = data_sample.ori_shape
                    masks = F.interpolate(
                        masks,
                        (H, W),
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    tmp_coords = boxes_filt.reshape(-1, 2, 2)
                    ori_h, ori_w = H, W
                    input_h, input_w = normalized_img_input.shape[-2:]
                    tmp_coords = deepcopy(tmp_coords).to(torch.float)
                    tmp_coords[..., 0] = tmp_coords[..., 0] * (ori_w / input_w)
                    tmp_coords[..., 1] = tmp_coords[..., 1] * (ori_h / input_h)
                    boxes_filt = tmp_coords.reshape(-1, 4)

                masks = torch.sigmoid(masks)
                mask_filt = masks > self.sam_threshold
                for idx in range(mask_filt.size(0)):
                    mask = mask_filt[idx]
                    label_idx = label_idx_filt[idx]
                    conf_score = conf_score_filt[idx]
                    mask = mask.squeeze()
                    # there's a label shift between the actual label and the label position in logits map
                    logits_map[label_idx + 1][mask] = conf_score

                if self.keep_background:
                    background_mask = torch.sum(logits_map, dim=0) == 0
                    batched_background_masks.append(background_mask)
                else:
                    background_mask = torch.zeros((logits_map.shape[-2:]), dtype=torch.bool)
                    # if not keep background just add a False mask
                    batched_background_masks.append(background_mask)

                    if not self.class_refinement:
                        batched_logits_map.append(logits_map[1:])
                    else:
                        refined_class_logits_map = logits_map[1:]
                        output_class_logits_map = torch.zeros((len(self.output_classes), H, W)).to(get_device())
                        for label_idx in range(len(self.output_classes)):
                            refined_class_idxs = self.class_output_idx2refined_idx[label_idx]
                            refined_class_logits = refined_class_logits_map[refined_class_idxs, :, :]
                            max_refined_class_logits = torch.max(refined_class_logits, dim=0)[0]
                            output_class_logits_map[label_idx] = max_refined_class_logits

                        batched_logits_map.append(output_class_logits_map)
                        logits_map = torch.cat([torch.zeros((1, logits_map.size(1), logits_map.size(2)),
                                                            dtype=torch.float32).to(get_device()), output_class_logits_map], dim=0)
            else:
                # no objects are detected
                background_mask = torch.zeros((logits_map.shape[-2:]), dtype=torch.bool)
                # if not keep background just add a False mask
                batched_background_masks.append(background_mask)
                batched_logits_map.append(logits_map)

            if self.visualize_results:
                if tuple(denormalized_img_input.shape[-2:]) != data_sample.ori_shape:
                    denormalized_img_input = F.interpolate(denormalized_img_input.type(torch.float), size=data_sample.ori_shape,
                                                           mode='bilinear', align_corners=False).type(torch.uint8)

                self._visualize_sample(data_sample, denormalized_img_input, boxes_filt, mask_filt,
                                       pred_phrases, label_idx_filt, logits_map)
                # self._visualize_sample(data_sample.img_path, denormalized_img_input, boxes_filt, mask_filt,
                #                        pred_phrases, label_idx_filt, logits_map)

        batched_logits_map = torch.stack(batched_logits_map)
        batched_background_masks = torch.stack(batched_background_masks)

        if self.kd_training:
            batched_logits_map = self.mask_result_module(batched_logits_map)
            batched_background_masks = self.background_mask_result_module(batched_background_masks)

        # with open(os.path.join(f"/home/aslab/code/yang_code/semantic_segmentation/mmsegmentation/saved_grounded_sam_logits/{self.inf_count}.pkl"), "wb") as output:
        #     pickle.dump(batched_logits_map, output)
        #     self.inf_count += 1

        return batched_logits_map, batched_background_masks

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        logits, background_mask = self._forward(inputs, data_samples)
        seg_gt = list()
        for data_sample in data_samples:
            seg_gt.append(data_sample.gt_sem_seg.data)
        seg_gt = torch.stack(seg_gt).squeeze()
        # loss = F.cross_entropy(logits, seg_gt, reduction='mean')
        fake_loss = dict(
            loss=torch.tensor(0.0)
        )
        return fake_loss

    def _visualize_sample(self, data_sample, img_tensor, boxes_filt, mask_filt, pred_phrases, label_idx_filt, logits_map) -> None:
        gt = data_sample.gt_sem_seg.data.squeeze().cpu().numpy()
        if self.visualize_classes is not None:
            for class_idx in self.vis_class_idxs:
                if class_idx in gt:
                    break
                else:
                    return

        img_path = data_sample.img_path
        filename_without_extension = Path(img_path).stem
        print(filename_without_extension)

        if filename_without_extension != "frankfurt_000001_025921_leftImg8bit":
            return

        boxes_filt = boxes_filt.detach().cpu().numpy()
        mask_filt = mask_filt.detach().cpu().numpy()
        img_tensor = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(img_tensor)
        plt.axis('off')
        plt.savefig(
            os.path.join(self.vis_dir, f"{filename_without_extension}.png"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        plt.clf()
        plt.close('all')
        plt.figure(figsize=(10, 10))
        fig, ax = plt.subplots()
        ax.imshow(img_tensor)

        for i in range(len(label_idx_filt)):
            label_idx = label_idx_filt[i]
            if self.class_refinement:
                color = self.color_palette[self.class_refined_idx2output_idx[label_idx]]
            else:
                color = self.color_palette[label_idx]

            pred_phrase = pred_phrases[i]
            color = np.array(color) / 255
            color = np.concatenate([color, np.array([0.6])], axis=0)
            box = boxes_filt[i]

            # add bbox
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=1))
            ax.text(x0, y0, pred_phrase, fontsize=6, color='white', bbox=dict(facecolor='gold', alpha=0.8, pad=0.2,
                                                                              edgecolor='gold'))

            # # add mask
            # h, w = mask.shape[-2:]
            # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            # ax.imshow(mask_image)

        plt.axis('off')
        plt.savefig(
            os.path.join(self.vis_dir, f"{filename_without_extension}_box.png"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        plt.clf()
        plt.close('all')

        plt.figure(figsize=(10, 5))
        max_logits_map, label_map = torch.max(logits_map, dim=0)
        max_logits_map = max_logits_map.detach().cpu().numpy()
        label_map = label_map.detach().cpu().numpy()

        color_palette = deepcopy(self.color_palette)
        # add background rgb value at the first place
        color_palette.insert(0, [0, 0, 0])
        # Ensure color_palette is of type float for later processing
        color_palette = np.asarray(color_palette).astype(np.float32)

        # Map labels to colors
        height, width = label_map.shape
        label_vis_img = np.zeros((height, width, 3), dtype=np.float32)
        for label_idx in range(color_palette.shape[0]):
            mask = label_map == label_idx
            label_vis_img[mask] = color_palette[label_idx] / 255.0  # Normalize colors to [0, 1]
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        plt.imshow(label_vis_img)
        plt.axis('off')  # Hide axis
        plt.title('pred')

        height, width = gt.shape
        gt_vis_img = np.zeros((height, width, 3), dtype=np.float32)
        for label_idx in range(color_palette.shape[0] - 1):
            mask = gt == label_idx
            gt_vis_img[mask] = color_palette[label_idx+1] / 255.0  # Normalize colors to [0, 1]
        # ignored pixels
        mask = gt == 255
        gt_vis_img[mask] = np.asarray([255.0, 255.0, 255.0]) / 255.0
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2st subplot
        plt.imshow(gt_vis_img)
        plt.axis('off')  # Hide axis
        plt.title('gt')

        plt.savefig(
            os.path.join(self.vis_dir, f"{filename_without_extension}_label.png"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        plt.clf()
        plt.close('all')

        plt.figure(figsize=(10, 5))
        plt.imshow(img_tensor)
        plt.imshow(max_logits_map, cmap='hot', interpolation='nearest', alpha=0.9)
        plt.axis('off')
        plt.savefig(
            os.path.join(self.vis_dir, f"{filename_without_extension}_logits.png"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        plt.clf()
        plt.close('all')

        threshold_logits = [0.447265625, 0.381103515625, 0.400146484375, 0.311279296875, 0.48388671875, 0.36572265625,
                            0.48291015625, 0.421875, 0.43115234375, 0.423095703125, 0.441650390625, 0.4375,
                            0.368408203125, 0.75439453125, 0.61474609375, 0.78466796875, 0.391357421875, 0.60107421875,
                            0.625]
        conf_filt_logits_map = deepcopy(max_logits_map)
        for index in range(len(threshold_logits)):
            logit_threshold = threshold_logits[index]
            label_mask = label_map == index + 1
            qualified_logit_mask = conf_filt_logits_map > logit_threshold
            conf_filt_logits_map[label_mask & qualified_logit_mask] = 0.0
        plt.figure(figsize=(10, 5))
        plt.imshow(img_tensor)
        plt.imshow(conf_filt_logits_map, cmap='hot', interpolation='nearest', alpha=0.9)
        plt.axis('off')
        plt.savefig(
            os.path.join(self.vis_dir, f"{filename_without_extension}_filt_logits.png"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        plt.clf()
        plt.close()

        if self.visualize_sam_masks:
            vis_sam_path = os.path.join(self.vis_dir, f"{filename_without_extension}_sam")
            os.makedirs(vis_sam_path, exist_ok=True)
            for i in range(len(label_idx_filt)):
                plt.figure(figsize=(10, 5))
                mask = np.squeeze(mask_filt[i]).astype(np.uint8)
                plt.imshow(mask, cmap='gray', interpolation='nearest')
                plt.axis('off')
                plt.savefig(os.path.join(vis_sam_path, f"mask_{i}.png"), bbox_inches='tight', pad_inches=0)
                plt.clf()
                plt.close()
        print()

    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        pass

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract visual features from images."""
        pass
