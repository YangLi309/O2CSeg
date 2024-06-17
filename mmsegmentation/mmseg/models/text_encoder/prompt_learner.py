from mmengine.model import BaseModule
from mmseg.registry import MODELS
from typing import List

@MODELS.register_module()
class CoOpPromptLearner(BaseModule):
    def __init__(self,
                 dataset_name: str = None,
                 vocabulary: List[str] = None,
                 templates: str = 'vild',
                 total_vocab_size: int = 49408,
                 context_length: int = 77,
                 embed_dims: int = 512,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 output_dims: int = 512,
                 cache_feature: bool = True,
                 cat_bg: bool = True,
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: dict = None):
        super().__init__(init_cfg)
