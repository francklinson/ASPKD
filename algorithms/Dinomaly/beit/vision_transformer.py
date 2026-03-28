"""
BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)

Model from official source: https://github.com/microsoft/unilm/tree/master/beit
and
https://github.com/microsoft/unilm/tree/master/beit2

@inproceedings{beit,
title={{BEiT}: {BERT} Pre-Training of Image Transformers},
author={Hangbo Bao and Li Dong and Songhao Piao and Furu Wei},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=p-BhZSz59o4}
}

@article{beitv2,
title={{BEiT v2}: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
author={Zhiliang Peng and Li Dong and Hangbo Bao and Qixiang Ye and Furu Wei},
year={2022},
eprint={2208.06366},
archivePrefix={arXiv},
primaryClass={cs.CV}
}

At this point only the 1k fine-tuned classification weights and model configs have been added,
see original source above for pre-training models and procedure.

Modifications by / Copyright 2021 Ross Wightman, original copyrights below
"""
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import build_model_with_cfg
from timm.models.layers import Mlp, DropPath, trunc_normal_, to_2tuple
from timm.models.vision_transformer import checkpoint_filter_fn
from torch.utils.checkpoint import checkpoint


def _cfg(url='', **kwargs):
    """
    配置函数，用于生成模型配置字典
    参数:
        url (str): 模型预训练权重的下载链接，默认为空字符串
        **kwargs: 其他可变关键字参数，用于扩展配置
    返回:
        dict: 包含模型配置信息的字典，包括:
            - url: 模型预训练权重的下载链接
            - num_classes: 分类任务的类别数量，默认为1000
            - input_size: 模型输入尺寸，格式为(通道数, 高度, 宽度)，默认为(3, 224, 224)
            - pool_size: 池化层尺寸，默认为None
            - crop_pct: 裁剪比例，默认为0.9
            - interpolation: 插值方法，默认为'bicubic'
            - fixed_input_size: 是否固定输入尺寸，默认为True
            - mean: 图像归一化的均值，默认为(0.5, 0.5, 0.5)
            - std: 图像归一化的标准差，默认为(0.5, 0.5, 0.5)
            - first_conv: 第一个卷积层的名称，默认为'patch_embed.proj'
            - classifier: 分类器层的名称，默认为'head'
            - **kwargs: 其他自定义配置项
    """
    return {
        'url': url,  # 模型预训练权重的下载链接
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'beit_base_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth'),
    'beit_base_patch16_384': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_384_pt22k_ft22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_base_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth',
        num_classes=21841,
    ),
    'beit_large_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22kto1k.pth'),
    'beit_large_patch16_384': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_384_pt22k_ft22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_large_patch16_512': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_512_pt22k_ft22kto1k.pth',
        input_size=(3, 512, 512), crop_pct=1.0,
    ),
    'beit_large_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth',
        num_classes=21841,
    ),

    'beitv2_base_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21kto1k.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_base_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth',
        num_classes=21841,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_224': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21kto1k.pth',
        crop_pct=0.95,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
    'beitv2_large_patch16_224_in22k': _cfg(
        url='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth',
        num_classes=21841,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    ),
}


def gen_relative_position_index(window_size: Tuple[int, int]) -> torch.Tensor:
    """
    生成窗口内每个token之间的相对位置索引
    参数:
        window_size: 一个包含两个元素的元组，表示窗口的高度和宽度
    返回:
        torch.Tensor: 一个包含相对位置索引的二维张量
    """
    # 计算相对位置的总数量，包括特殊位置关系
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    # cls to token & token 2 cls & cls to cls
    # get pair-wise relative position index for each token inside the window
    window_area = window_size[0] * window_size[1]
    coords = torch.stack(torch.meshgrid(
        [torch.arange(window_size[0]),
         torch.arange(window_size[1])]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = torch.zeros(size=(window_area + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    将2D图像分割成多个patch并嵌入到向量空间中
    """

    def __init__(
            self,  # 构造函数
            img_size=224,  # 图像大小，默认为224x224
            patch_size=16,  # 补丁大小，默认为16x16
            in_chans=3,  # 输入通道数，默认为3（RGB图像）
            embed_dim=768,  # 嵌入维度，默认为768
            norm_layer=None,  # 归一化层，默认为None
            flatten=True,  # 是否展平，默认为True
            bias=True,  # 是否使用偏置，默认为True
    ):
        super().__init__()  # 调用父类的构造函数
        # 将输入参数转换为2元组形式
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 保存图像大小和补丁大小
        self.img_size = img_size
        self.patch_size = patch_size
        # 计算网格大小和补丁数量
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 保存是否展平的标志
        self.flatten = flatten

        # 创建卷积层用于投影
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        # 创建归一化层，如果未提供则使用恒等变换
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):

        """
        初始化函数，用于创建类的实例并设置初始属性

        参数:
            dim: 输入特征的维度
            num_heads: 注意力头的数量，默认为8
            qkv_bias: 是否在QKV线性变换中使用偏置，默认为False
            attn_drop: 注意力dropout比率，默认为0
            proj_drop: 输出dropout比率，默认为0
            window_size: 窗口大小，用于相对位置编码，默认为None
            attn_head_dim: 每个注意力头的维度，如果为None则自动计算，默认为None
        """
        super().__init__()  # 调用父类的初始化方法
        # 设置注意力头数量
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 计算每个注意力头的维度
        # 如果指定了注意力头维度，则使用指定的维度
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads  # 计算所有注意力头的总维度
        self.scale = head_dim ** -0.5  # 缩放因子，用于缩放注意力分数

        # QKV线性变换层，没有偏置
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        # 如果启用了QKV偏置
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))  # 查询向量的偏置
            self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)  # 键向量的偏置
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))  # 值向量的偏置
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            self.register_buffer("relative_position_index", gen_relative_position_index(window_size))
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self):
        """
        获取相对位置偏置的方法
        该方法用于计算并返回相对位置偏置张量，通常用于自注意力机制中，
        以帮助模型更好地理解序列中不同位置之间的关系。
        Returns:
            torch.Tensor: 返回一个形状为(1, nH, Wh*Ww, Wh*Ww)的相对位置偏置张量
            其中:
            - nH 是头的数量
            - Wh 是窗口的高度
            - Ww 是窗口的宽度
        """
        # 从相对位置偏置表中获取偏置值，并通过相对位置索引进行选择
        # 首先将相对位置索引展平，然后从偏置表中获取对应的值
        # 最后重新塑形为(Wh*Ww, Wh*Ww, nH)的形状
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        # 调整张量的维度顺序，并确保内存是连续的
        # 将维度顺序从(Wh*Ww, Wh*Ww, nH)变为(nH, Wh*Ww, Wh*Ww)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # 在第0维添加一个维度，使最终的形状为(1, nH, Wh*Ww, Wh*Ww)
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None):

        """
        前向传播函数，实现多头自注意力机制

        参数:
            x: 输入张量，形状为 [批次大小, 序列长度, 通道数]
            shared_rel_pos_bias: 可选的共享相对位置偏置张量

        返回:
            输出张量，形状与输入相同
        """
        B, N, C = x.shape  # 获取批次大小、序列长度和通道数

        # 如果存在q/k/v偏置，则将它们拼接在一起；否则为None
        qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
        # 使用单个线性层计算查询、键和值
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # 重塑并转置张量以准备多头注意力计算
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # 分离查询、键和值
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # 缩放查询向量
        q = q * self.scale
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1))

        # 如果存在相对位置偏置表，则添加相对位置偏置
        if self.relative_position_bias_table is not None:
            attn = attn + self._get_rel_pos_bias()
        # 如果存在共享相对位置偏置，则添加它
        if shared_rel_pos_bias is not None:
            attn = attn + shared_rel_pos_bias

        # 应用softmax归一化
        attn = attn.softmax(dim=-1)
        # 应用注意力dropout
        attn = self.attn_drop(attn)

        # 计算输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # 应用投影层
        x = self.proj(x)
        # 应用投影dropout
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            window_size=None, attn_head_dim=None):

        """
        初始化函数，用于构建模型的基本结构

        参数:
            dim: 输入特征的维度
            num_heads: 注意力机制中的头数
            mlp_ratio: MLP隐藏层维度与输入维度的比例，默认为4
            qkv_bias: 是否在QKV投影中添加偏置，默认为False
            drop: Dropout概率，默认为0
            attn_drop: 注意力机制的Dropout概率，默认为0
            drop_path: 随机深度(drop path)的Dropout概率，默认为0
            init_values: 初始化缩放因子的值，默认为None
            act_layer: 激活函数层，默认为GELU
            norm_layer: 归一化层，默认为LayerNorm
            window_size: 注意力机制的窗口大小，默认为None
            attn_head_dim: 注意力头的维度，默认为None
        """
        super().__init__()  # 调用父类的初始化方法
        # 第一个归一化层
        self.norm1 = norm_layer(dim)
        # 注意力机制层
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            window_size=window_size, attn_head_dim=attn_head_dim)
        # 随机深度(drop path)层，用于训练时的随机深度
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二个归一化层
        self.norm2 = norm_layer(dim)
        # 计算MLP隐藏层的维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP层
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 如果提供了初始化值，则创建可学习的缩放参数
        if init_values:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, shared_rel_pos_bias: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        """
        初始化函数
        参数:
            window_size: 窗口大小，是一个包含两个元素的元组，表示窗口的高度和宽度
            num_heads: 注意力头的数量
        """
        super().__init__()  # 调用父类的初始化方法
        self.window_size = window_size  # 保存窗口大小
        self.window_area = window_size[0] * window_size[1]  # 计算窗口面积
        # 计算相对位置偏置表的大小
        # 公式：(2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 创建相对位置偏置表参数，形状为 (num_relative_distance, num_heads)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads))
        # 注释掉的代码：对相对位置偏置表进行截断正态分布初始化
        # trunc_normal_(self.relative_position_bias_table, std=.02)
        # 注册一个缓冲区，用于存储相对位置索引
        self.register_buffer("relative_position_index", gen_relative_position_index(window_size))

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area + 1, self.window_area + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class Beit(nn.Module):
    """
    Vision Transformer with support for patch or hybrid CNN input stage
    支持patch或混合CNN输入阶段的视觉Transformer
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='avg',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
            attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=None, use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
            head_init_scale=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.grad_checkpointing = False

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.grid_size, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.grid_size if use_rel_pos_bias else None)
            for i in range(depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed|rel_pos_bias',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def prepare_tokens(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
            else:
                x = blk(x, shared_rel_pos_bias=rel_pos_bias)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.fc_norm is not None:
            x = x[:, 1:].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _beit_checkpoint_filter_fn(state_dict, model):
    if 'module' in state_dict:
        # beit v2 didn't strip module
        state_dict = state_dict['module']
    return checkpoint_filter_fn(state_dict, model)


def _create_beit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Beit models.')

    model = build_model_with_cfg(
        Beit, variant, pretrained,
        # FIXME an updated filter fn needed to interpolate rel pos emb if fine tuning to diff model sizes
        pretrained_filter_fn=_beit_checkpoint_filter_fn,
        **kwargs)
    return model


# @register_model
def beit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
def beit_base_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
def beit_base_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


def beit_base_patch16_448(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, img_size=448, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
def beit_large_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
def beit_large_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
def beit_large_patch16_512(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beit_large_patch16_512', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
def beit_large_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beit_large_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
def beitv2_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beitv2_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
def beitv2_base_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beitv2_base_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


def beitv2_base_patch16_448(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, img_size=448, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beitv2_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
def beitv2_large_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beitv2_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
def beitv2_large_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beitv2_large_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model
