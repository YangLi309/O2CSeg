from mmengine.model import BaseModule
from mmseg.registry import MODELS

from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)
import numpy as np
from mmengine.config import Config, ConfigDict
import torch
from mmengine.structures import InstanceData
import torch.nn as nn
from mmseg.utils import OptConfigType
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmengine.model.utils import revert_sync_batchnorm

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray, torch.Tensor]
InputsType = Union[InputType, Sequence[InputType]]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict]]
ConfigType = Union[Config, ConfigDict]
ModelType = Union[dict, ConfigType, str]


@MODELS.register_module()
class GroundingDINO(BaseModule):
    def __init__(self,
                 cfg: str,
                 weights: str,
                 device: str = 'cpu',
                 ):
        super().__init__()

        checkpoint: Optional[dict] = None
        if weights is not None:
            checkpoint = _load_checkpoint(weights, map_location='cpu')
        if cfg.model.get('pretrained') is not None:
            del cfg.model.pretrained
        model = MODELS.build(cfg.model)
        model.cfg = cfg
        _load_checkpoint_to_model(model, checkpoint)
        model.to(device)
        model.eval()
