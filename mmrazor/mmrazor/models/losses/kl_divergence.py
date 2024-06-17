# Copyright (c) OpenMMLab. All rights reserved.
import os

import torch.nn as nn
import torch.nn.functional as F
import torch
from mmrazor.registry import MODELS
from torch import Tensor
import pickle
from tqdm import tqdm
from mmengine.device import get_device
@MODELS.register_module()
class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
    """

    def __init__(
        self,
        tau: float = 1.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if self.teacher_detach:
            preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        return self.loss_weight * loss


@MODELS.register_module()
class MaskedKLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        class_logits_threshold_path: the pickle file path of the class-wise logits threshold
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
    """

    def __init__(
        self,
        class_logits_threshold_path: str,
        tau: float = 1.0,
        reduction: str = 'none',
        loss_weight: float = 1.0,
        teacher_detach: bool = True
    ):
        super(MaskedKLDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach
        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction
        self.class_logits_threshold = pickle.load(open(class_logits_threshold_path, 'rb'))

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
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
        # loss_mask = all_qualified_masks.to(get_device())
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss_mask_expanded = loss_mask.unsqueeze(1).expand(-1, softmax_pred_T.size(1), -1, -1)
        # mask_ratio = torch.sum(loss_mask_expanded == 0)/(loss_mask_expanded.size(0)*loss_mask_expanded.size(1)*loss_mask_expanded.size(2)*loss_mask_expanded.size(3))
        # mask_ratio = mask_ratio.item()
        # print('mask ratio: ', mask_ratio)
        kd_loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        masked_kd_loss = kd_loss * loss_mask_expanded
        loss = torch.mean(masked_kd_loss)
        # print(loss)
        return self.loss_weight * loss


@MODELS.register_module()
class ConfKLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
        second, reference probability distribution P.

        Args:
            class_logits_threshold_path: the pickle file path of the class-wise logits threshold
            tau (float): Temperature coefficient. Defaults to 1.0.
            reduction (str): Specifies the reduction to apply to the loss:
                ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
                ``'none'``: no reduction will be applied,
                ``'batchmean'``: the sum of the output will be divided by
                    the batchsize,
                ``'sum'``: the output will be summed,
                ``'mean'``: the output will be divided by the number of
                    elements in the output.
                Default: ``'batchmean'``
            loss_weight (float): Weight of loss. Defaults to 1.0.
            teacher_detach (bool): Whether to detach the teacher model prediction.
                Will set to ``'False'`` in some data-free distillation algorithms.
                Defaults to True.
        """

    def __init__(
            self,
            top_k: int = 2,
            tau: float = 1.0,
            reduction: str = 'none',
            loss_weight: float = 1.0,
            teacher_detach: bool = True
    ):
        super(ConfKLDivergence, self).__init__()
        self.top_k = top_k
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach
        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction


    def forward(self, preds_S, preds_T, batch_metas=None):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if self.teacher_detach:
            preds_T = preds_T.detach()
        top_2_logits = torch.topk(preds_T, 2, dim=1)
        conf_diff = top_2_logits.values[:, 0, :, :] - top_2_logits.values[:, 1, :, :]
        conf_reweight = torch.softmax(conf_diff.flatten(1, 2), dim=1)
        conf_reweight_matrix = conf_reweight.unflatten(1, conf_diff.shape[-2:])

        # loss_mask = all_qualified_masks.to(get_device())
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        kd_loss = (self.tau ** 2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        kd_loss = kd_loss.mean(dim=1)
        reweighted_kd_loss = kd_loss * conf_reweight_matrix
        reweighted_kd_loss = reweighted_kd_loss.sum(dim=(1, 2))
        loss = torch.mean(reweighted_kd_loss)
        # print(loss)
        return self.loss_weight * loss


@MODELS.register_module()
class SoftLabelSmoothing(nn.Module):
    def __init__(
            self,
            top_k,
            tau: float = 1.0,
            reduction: str = 'none',
            loss_weight: float = 1.0,
            teacher_detach: bool = True
    ):
        super(SoftLabelSmoothing, self).__init__()
        self.tau = tau
        self.top_k = top_k

    def forward(self, preds_S, preds_T):
        assert preds_S.size() == preds_T.size()
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        if len(preds_T.size()) == 2:
            N, C = preds_T.shape
        else:
            N, C, H, W = preds_T.shape

        pass

def softmax_kl_loss_sl2(input_logits, target_logits, eps=0.35, k=-1, tau=1/7, function='softmax', reduction='batchmean'):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    #import pdb; pdb.set_trace()
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)

    if function == 'softmax':
        target = F.softmax(target_logits, dim=1)
    else:
        raise('Unsupported function')

    N, C = target_logits.shape
    smooth_labels = target.gt(tau).float() * target
    smooth_labels = smooth_labels / smooth_labels.sum(1).unsqueeze(1)
    smooth_labels = smooth_labels * (1 - eps)
    Ks = target.gt(tau).sum(1).unsqueeze(1)
    Ks = Ks + Ks.eq(0).int()
    small_mask = 1 - target.gt(tau).float()
    smooth_labels = smooth_labels + small_mask * (eps / (C - Ks.float()))

    return F.kl_div(input_log_softmax, smooth_labels, reduction=reduction)

@MODELS.register_module()
class EntropyKLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
        second, reference probability distribution P.

        Args:
            class_logits_threshold_path: the pickle file path of the class-wise logits threshold
            tau (float): Temperature coefficient. Defaults to 1.0.
            reduction (str): Specifies the reduction to apply to the loss:
                ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
                ``'none'``: no reduction will be applied,
                ``'batchmean'``: the sum of the output will be divided by
                    the batchsize,
                ``'sum'``: the output will be summed,
                ``'mean'``: the output will be divided by the number of
                    elements in the output.
                Default: ``'batchmean'``
            loss_weight (float): Weight of loss. Defaults to 1.0.
            teacher_detach (bool): Whether to detach the teacher model prediction.
                Will set to ``'False'`` in some data-free distillation algorithms.
                Defaults to True.
        """

    def __init__(
            self,
            top_k: int,
            tau: float = 1.0,
            reduction: str = 'none',
            loss_weight: float = 1.0,
            teacher_detach: bool = True
    ):
        super(EntropyKLDivergence, self).__init__()
        self.top_k = top_k
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach
        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
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
        # loss_mask = all_qualified_masks.to(get_device())
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss_mask_expanded = loss_mask.unsqueeze(1).expand(-1, softmax_pred_T.size(1), -1, -1)
        # mask_ratio = torch.sum(loss_mask_expanded == 0)/(loss_mask_expanded.size(0)*loss_mask_expanded.size(1)*loss_mask_expanded.size(2)*loss_mask_expanded.size(3))
        # mask_ratio = mask_ratio.item()
        # print('mask ratio: ', mask_ratio)
        kd_loss = (self.tau ** 2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        masked_kd_loss = kd_loss * loss_mask_expanded
        loss = torch.mean(masked_kd_loss)
        # print(loss)
        return self.loss_weight * loss