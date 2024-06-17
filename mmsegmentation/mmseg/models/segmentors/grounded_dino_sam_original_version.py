from typing import List, Optional, Union, Dict

import torch
import torch.nn.functional as F
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmseg.registry import MODELS

from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

from mmengine.device import get_device

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

@MODELS.register_module()
class GroundedDinoSAM(BaseSegmentor):
    def __init__(self,
                 dino_cfg: str,
                 dino_checkpoint: str,
                 sam_cfg: str,
                 sam_checkpoint: str,
                 output_classes: list,
                 refined_classes: list,
                 class_refined_idx2output_idx: dict,
                 class_output_idx2refined_idx: dict,
                 det_data_preprocessor: OptConfigType = None,
                 seg_data_preprocessor: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.output_classes = output_classes,
        self.refined_classes = refined_classes,
        self.class_refined_idx2output_idx = class_refined_idx2output_idx,
        self.class_output_idx2refined_idx = class_output_idx2refined_idx
        self.dino_model = self._load_dino_model(dino_cfg, dino_checkpoint, get_device())
        self.sam_model = SamPredictor(sam_hq_model_registry[sam_cfg.sam_encoder_version](checkpoint=sam_checkpoint).to(get_device()))
        self.input_text_prompt = ''
        for refined_class in refined_classes:
            self.input_text_prompt += refined_class + ' . '
        self.input_text_prompt = self.input_text_prompt.strip()
        print()


    def _load_dino_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def _get_grounding_output(self, image, box_threshold, text_threshold, with_logits=True, device="cpu"):
        with torch.no_grad():
            outputs = self.dino_model(image[None], captions=[self.input_text_prompt])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = self.dino_model.tokenizer
        tokenized = tokenlizer(self.input_text_prompt)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        print()

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        self._get_grounding_output(inputs, 0.5, 0.5, device=get_device())
        print()

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract visual features from images."""
        x = self.image_encoder(inputs)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode the name of classes with text_encoder and encode images with
        image_encoder.

        Then decode the class embedding and visual feature into a semantic
        segmentation map of the same size as input.
        """
        classifier_embeds = self.text_encoder()
        clip_inputs = inputs
        if self.asymetric_input:
            clip_inputs = F.interpolate(
                inputs, scale_factor=self.encoder_resolution, mode='bilinear')
        x = self.image_encoder(clip_inputs)
        seg_logits = self.decode_head.predict([inputs, x, classifier_embeds],
                                              batch_img_metas, self.test_cfg)

        return seg_logits

    # def train_step(self, data: Union[dict, tuple, list],
    #                optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
    #     with optim_wrapper.optim_context(self):
    #         seg_data = self.seg_data_preprocessor(data, True)
    #         det_data = self.det_data_preprocessor(data, True)
    #         losses = self._run_forward(data, mode='loss')  # type: ignore
    #     parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
    #     optim_wrapper.update_params(parsed_losses)
    #     return log_vars
