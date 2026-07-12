import glob
import importlib
import os

import torch
import torch.nn as nn
from timm.models._registry import is_model_in_modules
from timm.models._helpers import load_checkpoint
from timm.models.layers import set_layer_config
from timm.models._hub import load_model_config_from_hf
from timm.models._factory import parse_model_name
from util.registry import Registry
MODEL = Registry('Model')

# 本地预训练权重查找映射：timm/torchvision 模型名 → 本地文件名
_LOCAL_CKPT_MAP = {
    'timm_resnet34': 'resnet34-43635321.pth',
    'timm_resnet50': 'resnet50_a1_0-14fe96d1.pth',
    'timm_resnet18': 'resnet18-5c106cde.pth',
    'timm_wide_resnet50_2': 'wide_resnet50_racm-8234f177.pth',
    'tv_wide_resnet50_2': 'wide_resnet50_racm-8234f177.pth',
    'timm_wide_resnet101_2': 'wide_resnet101_2-ee7c0276.pth',
    'vit_small_patch16_224_dino': 'vitad/vit_small_patch16_224_dino.pth',
    'vit_base_patch16_224_dino': 'dino_vitbase16_pretrain.pth',
    'vit_small_patch14_dinov2': 'dinov2/dinov2_vits14_pretrain.pth',
    'vit_small_patch14_reg4_dinov2': 'dinov2/dinov2_vits14_reg4_pretrain.pth',
    'vit_base_patch14_dinov2': 'dinov2/dinov2_vitb14_pretrain.pth',
    'vit_base_patch14_reg4_dinov2': 'dinov2/dinov2_vitb14_reg4_pretrain.pth',
    'vit_large_patch14_dinov2': 'dinov2/dinov2_vitl14_pretrain.pth',
    'vit_large_patch14_reg4_dinov2': 'dinov2/dinov2_vitl14_reg4_pretrain.pth',
    'timm_tf_efficientnet_b4': 'tf_efficientnet_b4_aa-818f208c.pth',
    'timm_efficientnet_b5': 'efficientnet_b5_lukemelas-1a07897c.pth',
    'alexnet': 'alexnet-owt-7be5be79.pth',
}


def _find_local_checkpoint(model_name):
    """从 models/pre_trained/ 查找本地预训练权重文件"""
    ckpt_filename = _LOCAL_CKPT_MAP.get(model_name)
    if not ckpt_filename:
        return None
    # 相对于 algorithms/ADer/ 的路径
    candidates = [
        os.path.join('..', '..', 'models', 'pre_trained', ckpt_filename),
        os.path.join('model', 'pretrain', ckpt_filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _interpolate_pos_embed(state_dict, model):
    """对 pos_embed 进行双三次插值以适配不同分辨率"""
    if 'pos_embed' not in state_dict:
        return
    model_pe = getattr(model, 'pos_embed', None)
    if model_pe is None or state_dict['pos_embed'].shape == model_pe.shape:
        return
    import torch.nn.functional as F_interp
    src_pe = state_dict['pos_embed']  # [1, N_src, C]
    cls_token = src_pe[:, :1, :]
    src_patches = src_pe[:, 1:, :]  # [1, H_src*W_src, C]
    C = src_patches.shape[-1]
    H_src = int(src_patches.shape[1] ** 0.5)
    H_dst = int((model_pe.shape[1] - 1) ** 0.5)
    if H_src != H_dst:
        src_patches = src_patches.reshape(1, H_src, H_src, C).permute(0, 3, 1, 2)
        dst_patches = F_interp.interpolate(src_patches, size=(H_dst, H_dst), mode='bicubic', align_corners=False)
        dst_patches = dst_patches.permute(0, 2, 3, 1).reshape(1, -1, C)
        state_dict['pos_embed'] = torch.cat([cls_token, dst_patches], dim=1)

def get_model(cfg_model):
    model_name = cfg_model.name
    kwargs = {k: v for k, v in cfg_model.kwargs.items()}
    model_fn = MODEL.get_module(model_name)
    pretrained = kwargs.pop('pretrained')
    checkpoint_path = kwargs.pop('checkpoint_path')
    strict = kwargs.pop('strict')

    if model_name.startswith('timm_'):
        # pretrained=True 但 checkpoint_path 为空时，自动查找本地权重
        if pretrained and not checkpoint_path:
            local_ckpt = _find_local_checkpoint(model_name)
            if local_ckpt:
                checkpoint_path = local_ckpt
                pretrained = False
        if pretrained and checkpoint_path:
            pretrained = False
        if 'hf' in kwargs:
            model_name_hf = kwargs.pop('hf')
        else:
            model_name_hf = None
        # 传空 URL 的 pretrained_cfg 让 timm 不发网络请求
        pretrained_cfg = dict(url='', num_classes=0, input_size=(3, 224, 224))
        with set_layer_config(scriptable=None, exportable=None, no_jit=None):
            model = model_fn(pretrained=False, pretrained_cfg=pretrained_cfg, **kwargs)
        if checkpoint_path:
            load_checkpoint(model, checkpoint_path, strict=strict)
    else:
        # 非 timm 模型：pretrained=True 但 checkpoint_path 为空时，也自动查找本地权重
        if pretrained and not checkpoint_path:
            local_ckpt = _find_local_checkpoint(model_name)
            if local_ckpt:
                checkpoint_path = local_ckpt
                pretrained = False
        model = model_fn(pretrained=pretrained, **kwargs)
        if checkpoint_path:
            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except Exception:
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'net' in ckpt.keys():
                state_dict = ckpt['net']
            else:
                state_dict = ckpt
            _interpolate_pos_embed(state_dict, model)
            if not strict and False:
                no_ft_keywords = model.no_ft_keywords()
                for no_ft_keyword in no_ft_keywords:
                    del state_dict[no_ft_keyword]
                ft_head_keywords, num_classes = model.ft_head_keywords()
                for ft_head_keyword in ft_head_keywords:
                    if state_dict[ft_head_keyword].shape[0] <= num_classes:
                        del state_dict[ft_head_keyword]
                    elif state_dict[ft_head_keyword].shape[0] == num_classes:
                        continue
                    else:
                        state_dict[ft_head_keyword] = state_dict[ft_head_keyword][:num_classes]
            if isinstance(model, nn.Module):
                model.load_state_dict(state_dict, strict=strict)
            else:
                for sub_model_name, sub_state_dict in state_dict.items():
                    sub_model = getattr(model, sub_model_name, None)
                    sub_model.load_state_dict(sub_state_dict, strict=strict) if sub_model else None
    return model

files = glob.glob('model/[!_]*.py')
for file in files:
    try:
        model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))
    except (ImportError, AttributeError) as e:
        import warnings
        warnings.warn(f"Skip loading model module {file}: {e}")
