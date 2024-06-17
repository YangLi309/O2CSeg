from typing import List, Optional, Union, Dict

import torch
import torch.nn.functional as F
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmseg.registry import MODELS as SegMODELS
from mmengine.registry import MODELS as PreProcessorMODELS

from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

from mmdet.models.detectors.grounding_dino import GroundingDINO

from mmengine.device import get_device

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

# @SegMODELS.register_module()
class GroundedDinoSAM(BaseSegmentor):
    def __init__(self,
                 dino_cfg: ConfigType,
                 dino_checkpoint: str,
                 sam_cfg: ConfigType,
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
        dino_language_model = dino_cfg.pop('language_model')
        dino_cfg.pop('type')
        dino_cfg.pop('lazy')
        self.dino_model = GroundingDINO(dino_language_model, **dino_cfg)
        dino_checkpoint = torch.load(dino_checkpoint, map_location='cpu')
        load_result = self.dino_model.load_state_dict(dino_checkpoint, strict=False)
        self.sam_model = SamPredictor(sam_hq_model_registry[sam_cfg.sam_encoder_version](checkpoint=sam_checkpoint).to(get_device()))
        self.det_data_preprocessor = PreProcessorMODELS.build(det_data_preprocessor)
        self.seg_data_preprocessor = PreProcessorMODELS.build(seg_data_preprocessor)

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
        print(self.dino_model.training)
        if self.dino_model.training:
            self.dino_model.training = False
        grounding_result = self.dino_model.predict(inputs, data_samples, text_prompts=self.refined_classes)
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
