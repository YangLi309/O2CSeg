# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
import pickle
import torch

from mmengine.device import get_device


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T, boundary_label=None):
        preds_T = preds_T.detach()

        if boundary_label is not None:
            seg_labels = boundary_label
        else:
            seg_labels = preds_T.argmax(dim=1, keepdim=False)

        loss = F.cross_entropy(preds_S, seg_labels)
        return loss * self.loss_weight

# @MODELS.register_module()
# class SampledCrossEntropyLoss(nn.Module):
#     def __init__(self, loss_weight=1.0, sample_weight=None):
#         super(SampledCrossEntropyLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.sample_weight = sample_weight
#
#     def forward(self, preds_S, preds_T):
#         preds_T = preds_T.detach()
#         pseudo_label = preds_T.argmax(dim=1)
#
#
#         loss = F.cross_entropy(preds_S, pseudo_label)
#         return loss * self.loss_weight


@MODELS.register_module()
class MaskedCrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, class_logits_threshold_path: str, loss_weight=1.0, teacher_detach: bool = True):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.class_logits_threshold = pickle.load(open(class_logits_threshold_path, 'rb'))
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

    def forward(self, preds_S, preds_T):
        if self.teacher_detach:
            preds_T = preds_T.detach()
        pesudo_max_logits, pesudo_label = preds_T.max(dim=1)
        unique_labels = torch.unique(pesudo_label)
        qualified_masks = []
        for label_idx in unique_labels:
            pesudo_label_mask = pesudo_label == label_idx
            confident_logits_mask = pesudo_max_logits > self.class_logits_threshold[label_idx]
            qualified_masks.append(confident_logits_mask & pesudo_label_mask)

        stacked_qualified_masks = torch.stack(qualified_masks)
        all_qualified_masks = torch.any(stacked_qualified_masks, dim=0)
        loss_mask = all_qualified_masks.type(torch.float32).to(get_device())
        ce_loss = F.cross_entropy(preds_S, preds_T.argmax(dim=1), reduction='none')
        masked_loss = ce_loss * loss_mask
        # masked_loss = ce_loss[all_qualified_masks]
        loss = torch.mean(masked_loss)
        return loss * self.loss_weight
