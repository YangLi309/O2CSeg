from mmengine.registry import FUNCTIONS
from typing import Any, Mapping, Sequence


@FUNCTIONS.register_module()
def det_seg_collate_func(data_batch: Sequence) -> Any:
    print()
