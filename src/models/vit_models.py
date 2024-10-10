#!/usr/bin/env python3

"""
ViT-related models
Note: models return logits instead of prob
"""
import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision import models

from .build_vit_backbone import (
    build_vit_sup_models, build_swin_model,
    build_mocov3_model, build_mae_model
)
from .mlp import MLP
from ..utils import logging

logger = logging.get_logger("visual_prompt")


class ViT(nn.Module):
    """ViT 相关模型。"""

    def __init__(self, cfg, load_pretrain=True, vis=True):
        super(ViT, self).__init__()

        if "prompt" in cfg.MODEL.TRANSFER_TYPE:  # 如果配置中包含 "prompt"
            prompt_cfg = cfg.MODEL.PROMPT
        else:
            prompt_cfg = None

        self.froze_enc = False
        adapter_cfg = None

        self.build_backbone(
            prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis)  # 构建骨干网络
        self.cfg = cfg
        self.setup_side()  # 设置辅助功能
        self.setup_head(cfg)  # 设置头部结构

    def setup_side(self):
        if self.cfg.MODEL.TRANSFER_TYPE != "side":  # 如果模型的传输类型不是“side”
            self.side = None  # 将self.side设置为None
        else:
            self.side_alpha = nn.Parameter(torch.tensor(0.0))  # 初始化一个可学习的参数self.side_alpha
            m = models.alexnet(pretrained=True)  # 加载预训练的AlexNet模型
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),  # 包含AlexNet的特征提取部分
                ("avgpool", m.avgpool),  # 包含AlexNet的平均池化层
            ]))
            self.side_projection = nn.Linear(9216, self.feat_dim,
                                             bias=False)  # 定义一个线性层用于投影，输入维度为9216，输出维度为self.feat_dim，不使用偏置

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):  ##############################
        transfer_type = cfg.MODEL.TRANSFER_TYPE  # 获取传输类型
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg, load_pretrain, vis
        )  # 构建ViT模型，返回编码器和特征维度

        if transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "embeddings.patch_embeddings.pretrained_weight" not in k and "embeddings.patch_embeddings.bias" not in k:
                    p.requires_grad = False  # 只更新提示相关参数和嵌入层的权重和偏置

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False  # 只更新提示相关参数


    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                     [cfg.DATA.NUMBER_CLASSES],  # noqa
            special_bias=True
        )  # 设置MLP头部网络结构，输入维度为self.feat_dim，输出维度为cfg.DATA.NUMBER_CLASSES

    def forward(self, x, only_attn=False, is_extract=False):
        if self.side is not None:  # 如果存在侧边模型
            side_output = self.side(x)  # 获取侧边模型的输出
            side_output = side_output.view(side_output.size(0), -1)  # 将输出展平
            side_output = self.side_projection(side_output)  # 对展平后的输出进行投影

        if self.froze_enc and self.enc.training:  # 如果编码器被冻结且处于训练模式
            self.enc.eval()  # 将编码器设为评估模式
        x, attn_weights = self.enc(x, vis=True, is_extract=is_extract)  # 通过编码器获取特征表示，形状为(batch_size, self.feat_dim)  x为输入特征张量，我希望是特征张量##################

        if self.side is not None:  # 如果存在侧边模型
            alpha_squashed = torch.sigmoid(self.side_alpha)  # 计算sigmoid函数值
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output  # 使用alpha_squashed对编码器输出和侧边模型输出进行加权融合

        if is_extract:
            return x

        # if return_feature:  # 如果需要返回特征表示
        #     return x, x  # 返回编码器输出和自身
        x = self.head(x)  # 将特征表示通过MLP头部网络进行分类

        # return x, attn_weights  # 返回分类结果
        if only_attn:
            return attn_weights  # 返回注意力
        else:
            return x  # 分类结果

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)  # 获取分类器层级的嵌入表示
        return cls_embeds  # 返回分类器层级的嵌入表示

    def get_features(self, x):
        """获取(batch_size, self.feat_dim)的特征表示"""
        x = self.enc(x)  # 通过编码器获取特征表示，形状为(batch_size, self.feat_dim)
        return x  # 返回特征表示


class Swin(ViT):
    """Swin-related model."""

    def __init__(self, cfg):
        super(Swin, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_swin_model(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT
        )

        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-1].blocks)
            for k, p in self.enc.named_parameters():
                if "layers.{}.blocks.{}".format(total_layer - 1,
                                                total_blocks - 1) not in k and "norm.pretrained_weight" != k and "norm.bias" != k:  # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-2":
            total_layer = len(self.enc.layers)
            for k, p in self.enc.named_parameters():
                if "layers.{}".format(
                        total_layer - 1) not in k and "norm.pretrained_weight" != k and "norm.bias" != k:  # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-2].blocks)

            for k, p in self.enc.named_parameters():
                if "layers.{}".format(total_layer - 1) not in k and "layers.{}.blocks.{}".format(total_layer - 2,
                                                                                                 total_blocks - 1) not in k and "layers.{}.blocks.{}".format(
                        total_layer - 2, total_blocks - 2) not in k and "layers.{}.downsample".format(
                        total_layer - 2) not in k and "norm.pretrained_weight" != k and "norm.bias" != k:  # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION in ["below"]:
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "patch_embed" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))


class SSLViT(ViT):
    """moco-v3 and mae model."""

    def __init__(self, cfg):
        super(SSLViT, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        if "moco" in cfg.DATA.FEATURE:
            build_fn = build_mocov3_model
        elif "mae" in cfg.DATA.FEATURE:
            build_fn = build_mae_model

        self.enc, self.feat_dim = build_fn(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg=adapter_cfg
        )

        transfer_type = cfg.MODEL.TRANSFER_TYPE
        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "fc_norm" not in k and k != "norm":  # noqa
                    p.requires_grad = False
        elif transfer_type == "partial-2":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(
                        total_layer - 2) not in k and "fc_norm" not in k and k != "norm":  # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(
                        total_layer - 2) not in k and "blocks.{}".format(
                        total_layer - 3) not in k and "blocks.{}".format(
                        total_layer - 4) not in k and "fc_norm" not in k and k != "norm":  # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "sidetune":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "patch_embed.proj.pretrained_weight" not in k and "patch_embed.proj.bias" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        # adapter
        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))
