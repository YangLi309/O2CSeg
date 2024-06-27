# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.ops import point_sample
from mmengine.dist import all_reduce
from mmengine.model.weight_init import (caffe2_xavier_init, normal_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmseg.models.backbones.vit import TransformerEncoderLayer
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, MatchMasks, SampleList,
                         seg_data_to_instance_data)
from ..utils import (MLP, LayerNorm2d, PatchEmbed, cross_attn_layer,
                     get_uncertain_point_coords_with_randomness, resize)
from .decode_head import BaseDecodeHead
from mmengine.device import get_device
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import pickle
import shutil


class MLPMaskDecoder(nn.Module):
    """Module for decoding query and visual features with MLP layers to
    generate the attention biases and the mask proposals."""

    def __init__(
        self,
        *,
        in_channels: int,
        total_heads: int = 1,
        total_layers: int = 1,
        embed_channels: int = 256,
        mlp_channels: int = 256,
        mlp_num_layers: int = 3,
        rescale_attn_bias: bool = False,
    ):
        super().__init__()
        self.total_heads = total_heads
        self.total_layers = total_layers

        dense_affine_func = partial(nn.Conv2d, kernel_size=1)
        # Query Branch
        self.query_mlp = MLP(in_channels, mlp_channels, embed_channels,
                             mlp_num_layers)
        # Pixel Branch
        self.pix_mlp = MLP(
            in_channels,
            mlp_channels,
            embed_channels,
            mlp_num_layers,
            affine_func=dense_affine_func,
        )
        # Attention Bias Branch
        self.attn_mlp = MLP(
            in_channels,
            mlp_channels,
            embed_channels * self.total_heads * self.total_layers,
            mlp_num_layers,
            affine_func=dense_affine_func,
        )
        if rescale_attn_bias:
            self.bias_scaling = nn.Linear(1, 1)
        else:
            self.bias_scaling = nn.Identity()

    def forward(self, query: torch.Tensor,
                x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward function.
        Args:
            query (Tensor): Query Tokens [B,N,C].
            x (Tensor): Visual features [B,C,H,W]

        Return:
            mask_preds (Tensor): Mask proposals.
            attn_bias (List[Tensor]): List of attention bias.
        """
        query = self.query_mlp(query)
        pix = self.pix_mlp(x)
        b, c, h, w = pix.shape
        # predict mask
        mask_preds = torch.einsum('bqc,bchw->bqhw', query, pix)
        # generate attn bias
        attn = self.attn_mlp(x)
        attn = attn.reshape(b, self.total_layers, self.total_heads, c, h, w)
        attn_bias = torch.einsum('bqc,blnchw->blnqhw', query, attn)
        attn_bias = self.bias_scaling(attn_bias[..., None]).squeeze(-1)
        attn_bias = attn_bias.chunk(self.total_layers, dim=1)
        attn_bias = [attn.squeeze(1) for attn in attn_bias]
        return mask_preds, attn_bias


class SideAdapterNetwork(nn.Module):
    """Side Adapter Network for predicting mask proposals and attention bias.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        clip_channels (int): Number of channels of visual features.
            Default: 768.
        embed_dims (int): embedding dimension. Default: 240.
        patch_size (int): The patch size. Default: 16.
        patch_bias (bool): Whether to use bias in patch embedding.
            Default: True.
        num_queries (int): Number of queries for mask proposals.
            Default: 100.
        fusion_index (List[int]): The layer number of the encode
            transformer to fuse with the CLIP feature.
            Default: [0, 1, 2, 3].
        cfg_encoder (ConfigType): Configs for the encode layers.
        cfg_decoder (ConfigType): Configs for the decode layers.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
    """

    def __init__(
            self,
            in_channels: int = 3,
            clip_channels: int = 768,
            embed_dims: int = 240,
            patch_size: int = 16,
            patch_bias: bool = True,
            num_queries: int = 100,
            fusion_index: list = [0, 1, 2, 3],
            cfg_encoder: ConfigType = ...,
            cfg_decoder: ConfigType = ...,
            norm_cfg: dict = dict(type='LN'),
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            input_size=(640, 640),
            bias=patch_bias,
            norm_cfg=None,
            init_cfg=None,
        )
        ori_h, ori_w = self.patch_embed.init_out_size
        num_patches = ori_h * ori_w
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dims) * .02)
        self.query_pos_embed = nn.Parameter(
            torch.zeros(1, num_queries, embed_dims))
        self.query_embed = nn.Parameter(
            torch.zeros(1, num_queries, embed_dims))
        encode_layers = []
        for i in range(cfg_encoder.num_encode_layer):
            encode_layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=cfg_encoder.num_heads,
                    feedforward_channels=cfg_encoder.mlp_ratio * embed_dims,
                    norm_cfg=norm_cfg))
        self.encode_layers = nn.ModuleList(encode_layers)
        conv_clips = []
        for i in range(len(fusion_index)):
            conv_clips.append(
                nn.Sequential(
                    LayerNorm2d(clip_channels),
                    ConvModule(
                        clip_channels,
                        embed_dims,
                        kernel_size=1,
                        norm_cfg=None,
                        act_cfg=None)))
        self.conv_clips = nn.ModuleList(conv_clips)
        self.fusion_index = fusion_index
        self.mask_decoder = MLPMaskDecoder(
            in_channels=embed_dims,
            total_heads=cfg_decoder.num_heads,
            total_layers=cfg_decoder.num_layers,
            embed_channels=cfg_decoder.embed_channels,
            mlp_channels=cfg_decoder.mlp_channels,
            mlp_num_layers=cfg_decoder.num_mlp,
            rescale_attn_bias=cfg_decoder.rescale)

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)
        for i in range(len(self.conv_clips)):
            caffe2_xavier_init(self.conv_clips[i][1].conv)

    def fuse_clip(self, fused_index: int, x: torch.Tensor,
                  clip_feature: torch.Tensor, hwshape: Tuple[int,
                                                             int], L: int):
        """Fuse CLIP feature and visual tokens."""
        fused_clip = (resize(
            self.conv_clips[fused_index](clip_feature.contiguous()),
            size=hwshape,
            mode='bilinear',
            align_corners=False)).permute(0, 2, 3, 1).reshape(x[:, -L:,
                                                                ...].shape)
        x = torch.cat([x[:, :-L, ...], x[:, -L:, ...] + fused_clip], dim=1)
        return x

    def encode_feature(self, image: torch.Tensor,
                       clip_features: List[torch.Tensor],
                       deep_supervision_idxs: List[int]) -> List[List]:
        """Encode images by a lightweight vision transformer."""
        assert len(self.fusion_index) == len(clip_features)
        x, hwshape = self.patch_embed(image)
        ori_h, ori_w = self.patch_embed.init_out_size
        pos_embed = self.pos_embed
        if self.pos_embed.shape[1] != x.shape[1]:
            # resize the position embedding
            pos_embed = (
                resize(
                    self.pos_embed.reshape(1, ori_h, ori_w,
                                           -1).permute(0, 3, 1, 2),
                    size=hwshape,
                    mode='bicubic',
                    align_corners=False,
                ).flatten(2).permute(0, 2, 1))
        pos_embed = torch.cat([
            self.query_pos_embed.expand(pos_embed.shape[0], -1, -1), pos_embed
        ],
                              dim=1)
        x = torch.cat([self.query_embed.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + pos_embed
        L = hwshape[0] * hwshape[1]
        fused_index = 0
        if self.fusion_index[fused_index] == 0:
            x = self.fuse_clip(fused_index, x, clip_features[0][0], hwshape, L)
            fused_index += 1
        outs = []
        for index, block in enumerate(self.encode_layers, start=1):
            x = block(x)
            if index < len(self.fusion_index
                           ) and index == self.fusion_index[fused_index]:
                x = self.fuse_clip(fused_index, x,
                                   clip_features[fused_index][0], hwshape, L)
                fused_index += 1
            x_query = x[:, :-L, ...]
            x_feat = x[:, -L:, ...].permute(0, 2, 1)\
                .reshape(x.shape[0], x.shape[-1], hwshape[0], hwshape[1])

            if index in deep_supervision_idxs or index == len(
                    self.encode_layers):
                outs.append({'query': x_query, 'x': x_feat})

            if index < len(self.encode_layers):
                x = x + pos_embed
        return outs

    def decode_feature(self, features):
        mask_embeds = []
        attn_biases = []
        for feature in features:
            mask_embed, attn_bias = self.mask_decoder(**feature)
            mask_embeds.append(mask_embed)
            attn_biases.append(attn_bias)
        return mask_embeds, attn_biases

    def forward(
        self, image: torch.Tensor, clip_features: List[torch.Tensor],
        deep_supervision_idxs: List[int]
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward function."""
        # print(torch.mean(self.pos_embed))
        # print(self.pos_embed.requires_grad)

        features = self.encode_feature(image, clip_features,
                                       deep_supervision_idxs)
        mask_embeds, attn_biases = self.decode_feature(features)
        return mask_embeds, attn_biases


class RecWithAttnbias(nn.Module):
    """Mask recognition module by applying the attention biases to rest deeper
    CLIP layers.

    Args:
        sos_token_format (str): The format of sos token. It should be
            chosen from  ["cls_token", "learnable_token", "pos_embedding"].
            Default: 'cls_token'.
        sos_token_num (int): Number of sos token. It should be equal to
            the number of quries. Default: 100.
        num_layers (int): Number of rest CLIP layers for mask recognition.
            Default: 3.
        cross_attn (bool): Whether use cross attention to update sos token.
            Default: False.
        embed_dims (int): The feature dimension of CLIP layers.
            Default: 768.
        num_heads (int): Parallel attention heads of CLIP layers.
            Default: 768.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Whether to use bias in multihead-attention.
            Default: True.
        out_dims (int): Number of channels of the output mask proposals.
            It should be equal to the out_dims of text_encoder.
            Default: 512.
        final_norm (True): Whether use norm layer for sos token.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        frozen_exclude (List): List of parameters that are not to be frozen.
    """

    def __init__(self,
                 sos_token_format: str = 'cls_token',
                 sos_token_num: int = 100,
                 num_layers: int = 3,
                 cross_attn: bool = False,
                 embed_dims: int = 768,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 out_dims: int = 512,
                 final_norm: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 frozen_exclude: List = []):
        super().__init__()

        assert sos_token_format in [
            'cls_token', 'learnable_token', 'pos_embedding'
        ]
        self.sos_token_format = sos_token_format
        self.sos_token_num = sos_token_num
        self.frozen_exclude = frozen_exclude
        self.cross_attn = cross_attn
        self.num_layers = num_layers
        self.num_heads = num_heads
        if sos_token_format in ['learnable_token', 'pos_embedding']:
            self.sos_token = nn.Parameter(
                torch.randn(sos_token_num, 1, self.proj.shape[0]))
            self.frozen.append('sos_token')

        layers = []
        for i in range(num_layers):
            layers.append(
                BaseTransformerLayer(
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        batch_first=False,
                        bias=qkv_bias),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=mlp_ratio * embed_dims,
                        act_cfg=act_cfg),
                    operation_order=('norm', 'self_attn', 'norm', 'ffn')))
        self.layers = nn.ModuleList(layers)

        self.ln_post = build_norm_layer(norm_cfg, embed_dims)[1]
        self.proj = nn.Linear(embed_dims, out_dims, bias=False)

        self.final_norm = final_norm
        self._freeze()

    def init_weights(self, rec_state_dict):
        if hasattr(self, 'sos_token'):
            normal_init(self.sos_token, std=0.02)
        if rec_state_dict is not None:
            load_state_dict(self, rec_state_dict, strict=False, logger=None)
        else:
            super().init_weights()

    def _freeze(self):
        if 'all' in self.frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in self.frozen_exclude]):
                param.requires_grad = False

    def _build_attn_biases(self, attn_biases, target_shape):
        formatted_attn_biases = []
        for attn_bias in attn_biases:
            # convert it to proper format: N*num_head,L,L
            # attn_bias: [N, num_head/1, num_sos,H,W]
            n, num_head, num_sos, h, w = attn_bias.shape
            # reshape and downsample
            attn_bias = F.adaptive_max_pool2d(
                attn_bias.reshape(n, num_head * num_sos, h, w),
                output_size=target_shape)
            attn_bias = attn_bias.reshape(n, num_head, num_sos, *target_shape)

            true_num_head = self.num_heads
            assert (num_head == 1 or num_head
                    == true_num_head), f'num_head={num_head} is not supported.'
            if num_head == 1:
                attn_bias = attn_bias.repeat(1, true_num_head, 1, 1, 1)
            attn_bias = attn_bias.reshape(n * true_num_head, num_sos, -1)
            L = attn_bias.shape[-1]
            if self.cross_attn:
                # [n*num_head, num_sos, L]
                formatted_attn_biases.append(attn_bias)
            else:
                # [n*num_head, num_sos+1+L, num_sos+1+L]
                new_attn_bias = attn_bias.new_zeros(num_sos + 1 + L,
                                                    num_sos + 1 + L)
                new_attn_bias[:, :num_sos] = -100
                new_attn_bias[torch.arange(num_sos), torch.arange(num_sos)] = 0
                new_attn_bias[:num_sos, num_sos] = -100
                new_attn_bias = (
                    new_attn_bias[None, ...].expand(n * true_num_head, -1,
                                                    -1).clone())
                new_attn_bias[..., :num_sos, -L:] = attn_bias
                formatted_attn_biases.append(new_attn_bias)

        if len(formatted_attn_biases) == 1:
            formatted_attn_biases = [
                formatted_attn_biases[0] for _ in range(self.num_layers)
            ]
        return formatted_attn_biases

    def forward(self, bias: List[Tensor], feature: List[Tensor]):
        """Forward function to recognize the category of masks
        Args:
            bias (List[Tensor]): Attention bias for transformer layers
            feature (List[Tensor]): Output of the image encoder,
            including cls_token and img_feature.
        """
        cls_token = feature[1].unsqueeze(0)
        img_feature = feature[0]
        b, c, h, w = img_feature.shape
        # construct clip shadow features
        x = torch.cat(
            [cls_token,
             img_feature.reshape(b, c, -1).permute(2, 0, 1)])

        # construct sos token
        if self.sos_token_format == 'cls_token':
            sos_token = cls_token.repeat(self.sos_token_num, 1, 1)
        elif self.sos_token_format == 'learnable_token':
            sos_token = self.sos_token.expand(-1, b, -1)
        elif self.sos_token_format == 'pos_embedding':
            sos_token = self.sos_token.expand(-1, b, -1) + cls_token

        # construct attn bias
        attn_biases = self._build_attn_biases(bias, target_shape=(h, w))

        if self.cross_attn:
            for i, block in enumerate(self.layers):
                if self.cross_attn:
                    sos_token = cross_attn_layer(
                        block,
                        sos_token,
                        x[1:, ],
                        attn_biases[i],
                    )
                    if i < len(self.layers) - 1:
                        x = block(x)
        else:
            x = torch.cat([sos_token, x], dim=0)
            for i, block in enumerate(self.layers):
                x = block(x, attn_masks=[attn_biases[i]])
            sos_token = x[:self.sos_token_num]

        sos_token = sos_token.permute(1, 0, 2)  # LND -> NLD
        sos_token = self.ln_post(sos_token)
        sos_token = self.proj(sos_token)
        if self.final_norm:
            sos_token = F.normalize(sos_token, dim=-1)
        return sos_token



@MODELS.register_module()
class SideAdapterCLIPHead(BaseDecodeHead):
    """Side Adapter Network (SAN) for open-vocabulary semantic segmentation
    with pre-trained vision-language model.

    This decode head is the implementation of `Side Adapter Network
    for Open-Vocabulary Semantic Segmentation`
    <https://arxiv.org/abs/2302.12242>.
    Modified from https://github.com/MendelXu/SAN/blob/main/san/model/side_adapter/side_adapter.py # noqa:E501
    Copyright (c) 2023 MendelXu.
    Licensed under the MIT License

    Args:
        num_classes (int): the number of classes.
        san_cfg (ConfigType): Configs for SideAdapterNetwork module
        maskgen_cfg (ConfigType): Configs for RecWithAttnbias module
    """

    def __init__(self, num_classes: int, san_cfg: ConfigType,
                 maskgen_cfg: ConfigType, deep_supervision_idxs: List[int],
                 train_cfg: ConfigType,
                 output_classes: list,
                 exp_name: str,
                 dataset: str,
                 color_palette: list = None,
                 kd_training: bool = False,
                 save_pseudo_label_result: bool = False,
                 save_pseudo_logits: bool = False,
                 pseudo_save_dir: str = None,
                 crop_size: tuple = None,
                 refined_classes: list = None,
                 refine_class_mapping: dict = None,
                 class_refinement: bool = False,
                 class_refined_idx2output_idx: dict = None,
                 class_output_idx2refined_idx: dict = None,
                 visualize_results: bool = False,
                 vis_dir: str = 'vis_dir',
                 visualize_classes: list = None,
                 **kwargs):
        super().__init__(
            in_channels=san_cfg.in_channels,
            channels=san_cfg.embed_dims,
            num_classes=num_classes,
            **kwargs)
        assert san_cfg.num_queries == maskgen_cfg.sos_token_num, \
            'num_queries in san_cfg should be equal to sos_token_num ' \
            'in maskgen_cfg'
        del self.conv_seg
        self.side_adapter_network = SideAdapterNetwork(**san_cfg)
        self.rec_with_attnbias = RecWithAttnbias(**maskgen_cfg)
        self.deep_supervision_idxs = deep_supervision_idxs
        self.train_cfg = train_cfg
        self.mask_result_module = nn.Identity()
        self.kd_training = kd_training
        self.crop_size = crop_size
        self.output_classes = output_classes
        self.class_refinement = class_refinement
        self.save_pseudo_label_result = save_pseudo_label_result
        self.save_pseudo_logits = save_pseudo_logits
        self.visualize_results = visualize_results
        self.visualize_classes = visualize_classes
        self.vis_dir = vis_dir
        self.dataset = dataset
        self.color_palette = np.asarray(color_palette).astype(np.float32)
        if self.training:
            mode = 'train'
        else:
            mode = 'test'
        if self.visualize_results:
            self.vis_dir = os.path.join(self.vis_dir, exp_name, mode, self.dataset)
            os.makedirs(self.vis_dir, exist_ok=True)
            self.vis_class_idxs = [output_classes.index(cls) for cls in visualize_classes]
        if pseudo_save_dir is None:
            self.pseudo_save_dir = './pseudo_san_results'
        else:
            self.pseudo_save_dir = pseudo_save_dir
        if save_pseudo_logits or save_pseudo_label_result:
            os.makedirs(self.pseudo_save_dir, exist_ok=True)
        if self.class_refinement:
            self.refined_classes = refined_classes
            self.class_refined_idx2output_idx = class_refined_idx2output_idx
            self.class_output_idx2refined_idx = class_output_idx2refined_idx
            self.refine_class_mapping = refine_class_mapping
        else:
            self.refined_classes = output_classes
            self.class_refined_idx2output_idx = None
            self.class_output_idx2refined_idx = None
        if train_cfg:
            if self.class_refinement:
                self.match_masks = MatchMasks(
                    num_points=train_cfg.num_points,
                    num_queries=san_cfg.num_queries,
                    num_classes=len(self.output_classes),
                    assigner=train_cfg.assigner)
            else:
                self.match_masks = MatchMasks(
                    num_points=train_cfg.num_points,
                    num_queries=san_cfg.num_queries,
                    num_classes=num_classes,
                    assigner=train_cfg.assigner)

        self.save_count = 0

    def init_weights(self):

        rec_state_dict = None
        if isinstance(self.init_cfg, dict) and \
                self.init_cfg.get('type') == 'Pretrained_Part':
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            rec_state_dict = checkpoint.copy()
            para_prefix = 'decode_head.rec_with_attnbias'
            prefix_len = len(para_prefix) + 1
            for k, v in checkpoint.items():
                rec_state_dict.pop(k)
                if para_prefix in k:
                    rec_state_dict[k[prefix_len:]] = v

        self.side_adapter_network.init_weights()
        self.rec_with_attnbias.init_weights(rec_state_dict)

    def forward(self, inputs: Tuple[Tensor],
                deep_supervision_idxs) -> Tuple[List]:
        """Forward function.

        Args:
            inputs (Tuple[Tensor]): A triplet including images,
            list of multi-level visual features from image encoder and
            class embeddings from text_encoder.

        Returns:
            mask_props (List[Tensor]): Mask proposals predicted by SAN.
            mask_logits (List[Tensor]): Class logits of mask proposals.
        """
        imgs, clip_feature, class_embeds = inputs
        # predict mask proposals and attention bias
        mask_props, attn_biases = self.side_adapter_network(
            imgs, clip_feature, deep_supervision_idxs)

        # mask recognition with attention bias
        mask_embeds = [
            self.rec_with_attnbias(att_bias, clip_feature[-1])
            for att_bias in attn_biases
        ]
        # Obtain class prediction of masks by comparing the similarity
        # between the image token and the text embedding of class names.
        mask_logits = [
            torch.einsum('bqc,nc->bqn', mask_embed, class_embeds)
            for mask_embed in mask_embeds
        ]

        if self.kd_training:
            output = self.mask_result_module(
                self.predict_by_feat([mask_props[-1], mask_logits[-1]],
                                     [{'img_shape': self.crop_size}]))
            if self.save_pseudo_logits:
                with open(os.path.join(self.pseudo_save_dir, f"{self.save_count}.pkl"), "wb") as f:
                    pickle.dump(output, f)
                self.save_count += 1
        return mask_props, mask_logits

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): Images, visual features from image encoder
            and class embedding from text encoder.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        mask_props, mask_logits = self.forward(inputs, [])

        return self.predict_by_feat([mask_props[-1], mask_logits[-1]],
                                    batch_img_metas)

    def predict_by_feat(self, seg_logits: List[Tensor],
                        batch_img_metas: List[dict]) -> Tensor:
        """1. Transform a batch of mask proposals to the input shape.
           2. Generate segmentation map with mask proposals and class logits.
        """
        mask_pred = seg_logits[0]
        cls_score = seg_logits[1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred = F.interpolate(
            mask_pred, size=size, mode='bilinear', align_corners=False)

        mask_cls = F.softmax(cls_score, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        seg_logits = torch.einsum('bqc,bqhw->bchw', mask_cls, mask_pred)

        img_path = batch_img_metas[0]['img_path']
        # if 'frankfurt_000000_003357_leftImg8bit' in img_path:
        #     print('got the img')
        filename_without_extension = Path(img_path).stem
        #
        if self.visualize_results:
            shutil.copyfile(img_path, os.path.join(self.vis_dir, Path(img_path).name))
        #
        #     mean_seg_logits_before_refinement= torch.mean(seg_logits, dim=1)
        #
        #     # print(mean_seg_logits_before_refinement)
        #
        if self.class_refinement:
            H, W = seg_logits.shape[-2:]
            output_class_logits_map = torch.zeros(seg_logits.size(0), len(self.output_classes), H, W).to(get_device())
            for label_idx in range(len(self.output_classes)):
                refined_class_idxs = self.class_output_idx2refined_idx[label_idx]
                refined_class_logits = seg_logits[:, refined_class_idxs, :, :]
                max_refined_class_logits = torch.max(refined_class_logits, dim=1)[0]
                output_class_logits_map[:, label_idx] = max_refined_class_logits
            seg_logits = output_class_logits_map
        #
        #     # if 'frankfurt_000001_025921_leftImg8bit' in img_path:
        #     mean_seg_logits_after_refinement = torch.mean(seg_logits, dim=1)
        #
        #     # print(mean_seg_logits_after_refinement)
        #
        #     min_val = torch.min(torch.min(mean_seg_logits_before_refinement),
        #                         torch.min(mean_seg_logits_after_refinement)).item()
        #     max_val = torch.max(torch.max(mean_seg_logits_before_refinement),
        #                         torch.max(mean_seg_logits_after_refinement)).item()
        #


        top_2_logits = torch.topk(seg_logits, 2, dim=1)
        conf_diff = top_2_logits.values[:, 0, :, :] - top_2_logits.values[:, 1, :, :]
        conf_reweight = torch.softmax(conf_diff.flatten(1, 2), dim=1)
        conf_reweight_matrix = conf_reweight.unflatten(1, conf_diff.shape[-2:]).squeeze().cpu().numpy()
        if self.visualize_results:
            plt.figure(figsize=(10, 10))
            plt.imshow(conf_reweight_matrix, cmap='viridis', interpolation='nearest',
                    alpha=0.9)
            plt.axis('off')
            plt.savefig(os.path.join(self.vis_dir, f"{filename_without_extension}_conf_weight.png"),
                        bbox_inches="tight", dpi=300, pad_inches=0.0
                        )
            plt.clf()
            plt.close('all')

            _, pseudo_label = torch.max(seg_logits, dim=1)
            plt.figure(figsize=(10, 10))
            # Map labels to colors
            label_map = pseudo_label.squeeze().cpu().numpy()
            height, width = label_map.shape
            label_vis_img = np.zeros((height, width, 3), dtype=np.float32)
            for label_idx in range(self.color_palette.shape[0]):
                mask = label_map == label_idx
                label_vis_img[mask] = self.color_palette[label_idx] / 255.0  # Normalize colors to [0, 1]
            plt.imshow(label_vis_img)
            plt.axis('off')  # Hide axis
            plt.savefig(os.path.join(self.vis_dir, f"{filename_without_extension}_pseudo_label.png"),
                        bbox_inches="tight", dpi=300, pad_inches=0.0
                        )
            plt.clf()
            plt.close('all')

            seg_map_path = batch_img_metas[0]['seg_map_path']
            gt = np.asarray(Image.open(seg_map_path), dtype=np.uint8)
            height, width = gt.shape
            gt_vis_img = np.zeros((height, width, 3), dtype=np.float32)
            for label_idx in range(0, self.color_palette.shape[0]):
                mask = gt == label_idx
                # gt_vis_img[mask] = self.color_palette[label_idx - 1] / 255.0  # Normalize colors to [0, 1]
                gt_vis_img[mask] = self.color_palette[label_idx] / 255.0  # Normalize colors to [0, 1]
            # ignored pixels
            mask = gt == 255
            gt_vis_img[mask] = np.asarray([255.0, 255.0, 255.0]) / 255.0
            mask = gt == 254
            gt_vis_img[mask] = np.asarray([255.0, 255.0, 255.0]) / 255.0
            plt.imshow(gt_vis_img)
            plt.axis('off')  # Hide axis
            plt.savefig(os.path.join(self.vis_dir, f"{filename_without_extension}_gt.png"),
                        bbox_inches="tight", dpi=300, pad_inches=0.0
                        )
            plt.clf()
            plt.close('all')
        #
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(mean_seg_logits_before_refinement.squeeze().cpu().numpy(), cmap='viridis', interpolation='nearest',
        #                alpha=0.9, vmin=min_val, vmax=max_val)
        #     plt.axis('off')
        #     plt.savefig(os.path.join(self.vis_dir, f"{filename_without_extension}_san_pe.png"),
        #                 bbox_inches="tight", dpi=300, pad_inches=0.0
        #                 )
        #     plt.clf()
        #     plt.close('all')
        #
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(mean_seg_logits_after_refinement.squeeze().cpu().numpy(), cmap='viridis', interpolation='nearest',
        #                alpha=0.9, vmin=min_val, vmax=max_val)
        #     plt.axis('off')
        #     plt.savefig(os.path.join(self.vis_dir, f"{filename_without_extension}_san_pe_merged.png"),
        #                 bbox_inches="tight", dpi=300, pad_inches=0.0
        #                 )
        #     plt.clf()
        #     plt.close('all')
        #
        #
        #
        #     threshold = pickle.load(open('/mnt/storage/yang_code/semantic_segmentation/mmsegmentation/checkpoints/cityscapes_class_san_threshold_logits_0.1.pkl', 'rb'))
        #     thres_seg_logits = seg_logits.squeeze()
        #     min_val = torch.min(thres_seg_logits)
        #     for idx, thr in enumerate(threshold):
        #         thres_seg_logits[idx][thres_seg_logits[idx] < thr] = min_val
        #     thres_seg_logits = torch.mean(thres_seg_logits, dim=0).cpu().numpy()
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(thres_seg_logits, cmap='viridis', interpolation='nearest', alpha=0.9)
        #     plt.axis('off')
        #     plt.savefig(os.path.join(self.vis_dir, f"{filename_without_extension}_san_masked.png"),
        #                 bbox_inches="tight", dpi=300, pad_inches=0.0
        #                 )
        #     plt.clf()
        #     plt.close('all')


        if self.visualize_results:
            self._visualize_sample(batch_img_metas, seg_logits)

        return seg_logits

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances = seg_data_to_instance_data(self.ignore_index,
                                                       batch_data_samples)

        # forward
        all_mask_props, all_mask_logits = self.forward(
            x, self.deep_supervision_idxs)

        # loss
        losses = self.loss_by_feat(all_mask_logits, all_mask_props,
                                   batch_gt_instances)

        return losses

    def loss_by_feat(
            self, all_cls_scores: Tensor, all_mask_preds: Tensor,
            batch_gt_instances: List[InstanceData]) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        output_all_cls_scores = []
        if self.class_refinement:
            for cls_scores in all_cls_scores:
                output_cls_scores = torch.zeros(cls_scores.size(0), cls_scores.size(1), len(self.output_classes)+1).to(get_device())
                for label_idx in range(len(self.output_classes)):
                    refined_class_idxs = self.class_output_idx2refined_idx[label_idx]
                    refined_class_logits = cls_scores[:, :, refined_class_idxs]
                    max_refined_class_logits = torch.max(refined_class_logits, dim=2)[0]
                    output_cls_scores[:, :, label_idx] = max_refined_class_logits
                output_cls_scores[:, :, -1] = cls_scores[:, :, -1]
                output_all_cls_scores.append(output_cls_scores)
            all_cls_scores = output_all_cls_scores
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]

        losses = []
        for i in range(num_dec_layers):
            cls_scores = all_cls_scores[i]
            mask_preds = all_mask_preds[i]
            # matching N mask predictions to K category labels
            (labels, mask_targets, mask_weights,
             avg_factor) = self.match_masks.get_targets(
                 cls_scores, mask_preds, batch_gt_instances_list[i])
            cls_scores = cls_scores.flatten(0, 1)
            labels = labels.flatten(0, 1)
            num_total_masks = cls_scores.new_tensor([avg_factor],
                                                    dtype=torch.float)
            all_reduce(num_total_masks, op='mean')
            num_total_masks = max(num_total_masks, 1)

            # extract positive ones
            # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
            mask_preds = mask_preds[mask_weights > 0]

            if mask_targets.shape[0] != 0:
                with torch.no_grad():
                    points_coords = get_uncertain_point_coords_with_randomness(
                        mask_preds.unsqueeze(1), None,
                        self.train_cfg.num_points,
                        self.train_cfg.oversample_ratio,
                        self.train_cfg.importance_sample_ratio)
                    # shape (num_total_gts, h, w)
                    # -> (num_total_gts, num_points)
                    mask_point_targets = point_sample(
                        mask_targets.unsqueeze(1).float(),
                        points_coords).squeeze(1)
                # shape (num_queries, h, w) -> (num_queries, num_points)
                mask_point_preds = point_sample(
                    mask_preds.unsqueeze(1), points_coords).squeeze(1)

            if not isinstance(self.loss_decode, nn.ModuleList):
                losses_decode = [self.loss_decode]
            else:
                losses_decode = self.loss_decode
            loss = dict()
            for loss_decode in losses_decode:
                if 'loss_cls' in loss_decode.loss_name:
                    if loss_decode.loss_name == 'loss_cls_ce':
                        loss[loss_decode.loss_name] = loss_decode(
                            cls_scores, labels)
                    else:
                        assert False, "Only support 'CrossEntropyLoss' in" \
                                      ' classification loss'

                elif 'loss_mask' in loss_decode.loss_name:
                    if mask_targets.shape[0] == 0:
                        loss[loss_decode.loss_name] = mask_preds.sum()
                    elif loss_decode.loss_name == 'loss_mask_ce':
                        loss[loss_decode.loss_name] = loss_decode(
                            mask_point_preds,
                            mask_point_targets,
                            avg_factor=num_total_masks *
                            self.train_cfg.num_points)
                    elif loss_decode.loss_name == 'loss_mask_dice':
                        loss[loss_decode.loss_name] = loss_decode(
                            mask_point_preds,
                            mask_point_targets,
                            avg_factor=num_total_masks)
                    else:
                        assert False, "Only support 'CrossEntropyLoss' and" \
                                      " 'DiceLoss' in mask loss"
                else:
                    assert False, "Only support for 'loss_cls' and 'loss_mask'"

            losses.append(loss)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict.update(losses[-1])
        # loss from other decoder layers
        for i, loss in enumerate(losses[:-1]):
            for k, v in loss.items():
                loss_dict[f'd{self.deep_supervision_idxs[i]}.{k}'] = v
        return loss_dict

    def _visualize_sample(self, data_metas, logits_map) -> None:
        max_logits_maps, label_maps = torch.max(logits_map, dim=1)
        max_logits_maps = max_logits_maps.detach().cpu().numpy()
        label_maps = label_maps.detach().cpu().numpy()
        for idx, data_info in enumerate(data_metas):
            # img_size = data_info['pad_shape']
            seg_map_path = data_info['seg_map_path']
            img_path = data_info['img_path']
            filename_without_extension = Path(img_path).stem
            # print(filename_without_extension)
            img = Image.open(img_path)
            gt = np.asarray(Image.open(seg_map_path), dtype=np.uint8)
            if self.visualize_classes is not None:
                for class_idx in self.vis_class_idxs:
                    if class_idx in gt:
                        break
                    else:
                        return
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(
                os.path.join(self.vis_dir, f"{filename_without_extension}.png"),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )
            plt.clf()
            plt.close('all')

            plt.figure(figsize=(10, 5))
            # Map labels to colors
            label_map = label_maps[idx]
            height, width = label_map.shape
            label_vis_img = np.zeros((height, width, 3), dtype=np.float32)
            for label_idx in range(self.color_palette.shape[0]):
                mask = label_map == label_idx
                label_vis_img[mask] = self.color_palette[label_idx] / 255.0  # Normalize colors to [0, 1]
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
            plt.imshow(label_vis_img)
            plt.axis('off')  # Hide axis
            plt.title('pred')
            # print('gt shape', gt.shape)
            height, width = gt.shape
            gt_vis_img = np.zeros((height, width, 3), dtype=np.float32)
            for label_idx in range(1, self.color_palette.shape[0]):
                mask = gt == label_idx
                # gt_vis_img[mask] = self.color_palette[label_idx - 1] / 255.0  # Normalize colors to [0, 1]
                gt_vis_img[mask] = self.color_palette[label_idx] / 255.0  # Normalize colors to [0, 1]
            # ignored pixels
            mask = gt == 255
            gt_vis_img[mask] = np.asarray([255.0, 255.0, 255.0]) / 255.0
            mask = gt == 254
            gt_vis_img[mask] = np.asarray([255.0, 255.0, 255.0]) / 255.0

            plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2st subplot
            plt.imshow(gt_vis_img)
            plt.axis('off')  # Hide axis
            # gt_unique_idxs = np.unique(gt) - 1
            gt_unique_idxs = np.unique(gt)
            gt_unique_idxs = gt_unique_idxs.tolist()
            gt_title = ''
            # print(gt_unique_idxs)
            for label_idx in gt_unique_idxs:
                if label_idx != -1 and label_idx != 255 and label_idx != 254:
                    gt_title += self.output_classes[label_idx] + " "
            plt.title(gt_title)

            plt.savefig(
                os.path.join(self.vis_dir, f"{filename_without_extension}_label.png"),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )
            plt.clf()
            plt.close('all')

            # print()
