#!/usr/bin/env python3
import numpy as np
import torch
import os
from .vit_backbones.swin_transformer import SwinTransformer
from .vit_backbones.vit import VisionTransformer
from .vit_backbones.vit_moco import vit_base
from .vit_backbones.vit_mae import build_model as mae_vit_model

from .vit_prompt.vit import PromptedVisionTransformer
from .vit_prompt.swin_transformer import PromptedSwinTransformer
from .vit_prompt.vit_moco import vit_base as prompt_vit_base
from .vit_prompt.vit_mae import build_model as prompt_mae_vit_model

from .vit_adapter.vit_mae import build_model as adapter_mae_vit_model
from .vit_adapter.vit_moco import vit_base as adapter_vit_base

from .vit_adapter.vit import ADPT_VisionTransformer
MODEL_ZOO = {
    "swint_imagenet": "swin_tiny_patch4_window7_224.pth",
    "swint_imagenet_ssl": "moby_swin_t_300ep_pretrained.pth",
    "swins_imagenet": "swin_small_patch4_window7_224.pth",
    "swinb_imagenet_224": "swin_base_patch4_window7_224.pth",
    "swinb_imagenet_384": "swin_base_patch4_window12_384.pth",
    "swinb_imagenet22k_224":  "swin_base_patch4_window7_224_22k.pth",
    "swinb_imagenet22k_384": "swin_base_patch4_window12_384_22k.pth",
    "swinl_imagenet22k_224": "swin_large_patch4_window7_224_22k.pth",
    "sup_vitb8": "ViT-B_8.npz",
    "sup_vitb16_224": "ViT-B_16-224.npz",
    "sup_vitb16": "ViT-B_16.npz",
    "sup_vitl16_224": "ViT-L_16-224.npz",
    "sup_vitl16": "ViT-L_16.npz",
    "sup_vitb8_imagenet21k": "imagenet21k_ViT-B_8.npz",
    "sup_vitb32_imagenet21k": "imagenet21k_ViT-B_32.npz",
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
    "sup_vitl16_imagenet21k": "imagenet21k_ViT-L_16.npz",
    "sup_vitl32_imagenet21k": "imagenet21k_ViT-L_32.npz",
    "sup_vith14_imagenet21k": "imagenet21k_ViT-H_14.npz",
    "mae_vith14": "mae_pretrain_vit_huge.pth",
    "mae_vitb16": "mae_pretrain_vit_base.pth",
    "mae_vitl16": "mae_pretrain_vit_large.pth",
}


def build_mae_model(
    model_type, crop_size, prompt_cfg, model_root, adapter_cfg=None
):
    if prompt_cfg is not None:  # 如果存在prompt配置
        model = prompt_mae_vit_model(model_type, prompt_cfg)  # 构建prompt MAE ViT模型
    elif adapter_cfg is not None:  # 如果存在adapter配置
        model = adapter_mae_vit_model(model_type, adapter_cfg)  # 构建adapter MAE ViT模型
    else:  # 否则
        model = mae_vit_model(model_type)  # 构建普通的MAE ViT模型
    out_dim = model.embed_dim  # 获取输出维度

    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])  # 获取模型的checkpoint路径
    checkpoint = torch.load(ckpt, map_location="cpu")  # 加载模型checkpoint
    state_dict = checkpoint['model']  # 获取模型的状态字典

    model.load_state_dict(state_dict, strict=False)  # 加载模型的状态字典到模型中，允许部分参数不严格匹配
    model.head = torch.nn.Identity()  # 将模型的头部设置为Identity，即不进行任何操作

    return model, out_dim  # 返回模型和输出维度



def build_mocov3_model(
    model_type, crop_size, prompt_cfg, model_root, adapter_cfg=None
):
    if model_type != "mocov3_vitb":
        raise ValueError("Does not support other arch")
    if prompt_cfg is not None:
        model = prompt_vit_base(prompt_cfg)
    elif adapter_cfg is not None:
        model = adapter_vit_base(adapter_cfg)
    else:
        model = vit_base()
    out_dim = 768
    ckpt = os.path.join(model_root,"mocov3_linear-vit-b-300ep.pth.tar")

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.head = torch.nn.Identity()
    return model, out_dim


def build_swin_model(model_type, crop_size, prompt_cfg, model_root):
    if prompt_cfg is not None:
        return _build_prompted_swin_model(
            model_type, crop_size, prompt_cfg, model_root)
    else:
        return _build_swin_model(model_type, crop_size, model_root)


def _build_prompted_swin_model(model_type, crop_size, prompt_cfg, model_root):
    if model_type == "swint_imagenet":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swint_imagenet_ssl":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4

    elif model_type == "swins_imagenet":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swinb_imagenet_224":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet_384":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4

    elif model_type == "swinb_imagenet22k_224":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet22k_384":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinl_imagenet22k_224":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 192
        num_layers = 4

    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model']

    if crop_size == 448:
        for k in list(state_dict.keys()):
            if "attn_mask" not in k:
                # remove prefix
                state_dict[k] = state_dict[k]
            # delete renamed or unused k
            else:
                del state_dict[k]

    # rename some keys for ssl models
    if model_type.endswith("ssl"):
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder.'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim


def _build_swin_model(model_type, crop_size, model_root):
    if model_type == "swint_imagenet":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,  # setting to a negative value will make head as identity
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swint_imagenet_ssl":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4

    elif model_type == "swins_imagenet":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swinb_imagenet_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet_384":
        model = SwinTransformer(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4

    elif model_type == "swinb_imagenet22k_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet22k_384":
        model = SwinTransformer(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinl_imagenet22k_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 192
        num_layers = 4

    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model']

    if crop_size == 448:
        for k in list(state_dict.keys()):
            if "attn_mask" not in k:
                # remove prefix
                state_dict[k] = state_dict[k]
            # delete renamed or unused k
            else:
                del state_dict[k]

    # rename some keys for ssl models
    if model_type.endswith("ssl"):
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder.'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim


def build_vit_sup_models(
        model_type, crop_size, prompt_cfg=None, model_root=None, adapter_cfg=None, load_pretrain=True, vis=False
):
    # image size is the size of actual image
    m2featdim = {
        "sup_vitb16_224": 768,
        "sup_vitb16": 768,
        "sup_vitl16_224": 1024,
        "sup_vitl16": 1024,
        "sup_vitb8_imagenet21k": 768,
        "sup_vitb16_imagenet21k": 768,
        "sup_vitb32_imagenet21k": 768,
        "sup_vitl16_imagenet21k": 1024,
        "sup_vitl32_imagenet21k": 1024,
        "sup_vith14_imagenet21k": 1280,
    }  # 不同模型类型对应的特征维度
    if prompt_cfg is not None:  # 如果存在prompt配置
        model = PromptedVisionTransformer(
            prompt_cfg, model_type,
            crop_size, num_classes=-1, vis=vis
        )  # 构建Prompted Vision Transformer模型
    elif adapter_cfg is not None:  # 如果存在adapter配置
        model = ADPT_VisionTransformer(model_type, crop_size, num_classes=-1,
                                       adapter_cfg=adapter_cfg)  # 构建ADPT Vision Transformer模型
    else:  # 否则
        model = VisionTransformer(
            model_type, crop_size, num_classes=-1, vis=vis)  # 构建普通的Vision Transformer模型

    if load_pretrain:  # 如果加载预训练模型
        model.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))  # 加载预训练模型的权重

    return model, m2featdim[model_type]  # 返回模型和对应的特征维度


