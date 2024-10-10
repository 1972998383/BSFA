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
from src.models.mlp import MLP
from ..utils import logging
logger = logging.get_logger("visual_prompt")

class ViT(nn.Module):
    """ViT 相关模型。"""

    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        if "prompt" in cfg.MODEL.TRANSFER_TYPE:  # 如果配置中包含 "prompt"
            prompt_cfg = cfg.MODEL.PROMPT
        else:
            prompt_cfg = None

        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "prompt" not in cfg.MODEL.TRANSFER_TYPE:    #######
            # 如果传输类型不是 "end2end"，且不包含 "prompt"
            self.froze_enc = True  # 冻结编码器     ######
        else:
            # 否则
            self.froze_enc = False

        if cfg.MODEL.TRANSFER_TYPE == "adapter":  # 如果传输类型是 "adapter"
            adapter_cfg = cfg.MODEL.ADAPTER
        else:
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
            self.side_projection = nn.Linear(9216, self.feat_dim,bias=False)  # 定义一个线性层用于投影，输入维度为9216，输出维度为self.feat_dim，不使用偏置

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):  ##############################
        transfer_type = cfg.MODEL.TRANSFER_TYPE  # 获取传输类型
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg, load_pretrain, vis
        )  # 构建ViT模型，返回编码器和特征维度

        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.transformer.encoder.layer)  # 获取总的编码器层数
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.encoder_norm" not in k:
                    p.requires_grad = False  # 只更新最后一层编码器层和编码器归一化层的参数

        elif transfer_type == "partial-2":
            total_layer = len(self.enc.transformer.encoder.layer)  # 获取总的编码器层数
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 2) not in k and "transformer.encoder.encoder_norm" not in k:
                    p.requires_grad = False  # 只更新最后两层编码器层和编码器归一化层的参数

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.transformer.encoder.layer)  # 获取总的编码器层数
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 2) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 3) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 4) not in k and "transformer.encoder.encoder_norm" not in k:
                    p.requires_grad = False  # 只更新最后四层编码器层和编码器归一化层的参数

        elif transfer_type == "linear" or transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False  # 线性或side传输类型下，不更新任何参数

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False  # 只更新偏置参数

        elif transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "embeddings.patch_embeddings.pretrained_weight" not in k and "embeddings.patch_embeddings.bias" not in k:
                    p.requires_grad = False  # 只更新提示相关参数和嵌入层的权重和偏置

        elif transfer_type == "prompt":             ###############################################
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False  # 只更新提示相关参数

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False  # 只更新提示相关参数和偏置参数

        elif transfer_type == "prompt-noupdate":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False  # 不更新任何参数

        elif transfer_type == "cls":
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False  # 只更新cls_token参数

        elif transfer_type == "cls-reinit":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )  # 重新初始化cls_token
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False  # 只更新cls_token参数

        elif transfer_type == "cls+prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False  # 只更新提示相关参数和cls_token参数

        elif transfer_type == "cls-reinit+prompt":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )  # 重新初始化cls_token
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False  # 只更新提示相关参数和cls_token参数

        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False  # 只更新adapter相关参数

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")  # 启用所有参数更新

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))  # 抛出不支持的传输类型的错误

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                     [cfg.DATA.NUMBER_CLASSES],  # noqa
            special_bias=True
        )  # 设置MLP头部网络结构，输入维度为self.feat_dim，输出维度为cfg.DATA.NUMBER_CLASSES

    def forward(self, x, return_feature=False):
        if self.side is not None:  # 如果存在侧边模型
            side_output = self.side(x)  # 获取侧边模型的输出
            side_output = side_output.view(side_output.size(0), -1)  # 将输出展平
            side_output = self.side_projection(side_output)  # 对展平后的输出进行投影

        if self.froze_enc and self.enc.training:  # 如果编码器被冻结且处于训练模式
            self.enc.eval()  # 将编码器设为评估模式
        x, attn_weights = self.enc(x, vis=True)  # 通过编码器获取特征表示，形状为(batch_size, self.feat_dim)   ################正文嵌入#############
        # print(attn_weights)
        if self.side is not None:  # 如果存在侧边模型
            alpha_squashed = torch.sigmoid(self.side_alpha)  # 计算sigmoid函数值
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output  # 使用alpha_squashed对编码器输出和侧边模型输出进行加权融合

        if return_feature:  # 如果需要返回特征表示
            return x, x  # 返回编码器输出和自身
        # print(attn_weights)
        x = self.head(x)  # 将特征表示通过MLP头部网络进行分类

        return x  # 返回分类结果

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)  # 获取分类器层级的嵌入表示
        return cls_embeds  # 返回分类器层级的嵌入表示

    def get_features(self, x):
        """获取(batch_size, self.feat_dim)的特征表示"""
        x = self.enc(x)  # 通过编码器获取特征表示，形状为(batch_size, self.feat_dim)
        return x  # 返回特征表示
