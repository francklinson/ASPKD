import math
from functools import partial

import torch
import torch.nn as nn

from .utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    实现DropPath功能，这是一种正则化方法，在训练过程中随机丢弃一些路径。
    这个函数可以处理任意维度的张量，而不仅仅是2D卷积网络。
    参数:
        x: 输入张量
        drop_prob: float类型，默认为0。表示丢弃路径的概率
        training: bool类型，默认为False。表示是否在训练模式下
    返回:
        处理后的张量，如果不在训练模式或drop_prob为0，则返回原始输入
    """
    # 如果drop_prob为0或者不在训练模式下，直接返回输入张量
    if drop_prob == 0. or not training:
        return x
    # 计算保留路径的概率
    keep_prob = 1 - drop_prob
    # 创建一个形状元组，用于生成随机张量
    # 形状为(批次大小,) + (1,) * (维度数 - 1)，这样可以广播到所有维度
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # 生成随机张量，值在[keep_prob, 1.0]范围内
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # 将随机张量二值化，大于等于1的变为1，小于1的变为0
    random_tensor.floor_()  # binarize
    # 应用DropPath: 先除以keep_prob进行缩放，然后乘以二值化的随机张量
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    这是一个实现随机深度（Stochastic Depth）的类，用于在残差块的主路径中按样本随机丢弃路径。
    参数:
        drop_prob (float, optional): 丢弃路径的概率。默认为None。
    """

    def __init__(self, drop_prob=None):
        """
        初始化DropPath模块。

        参数:
            drop_prob (float, optional): 丢弃路径的概率。默认为None。
        """
        super(DropPath, self).__init__()  # 调用父类nn.Module的初始化方法
        self.drop_prob = drop_prob  # 设置丢弃路径的概率

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 经过随机深度处理后的输出张量
        """
        return drop_path(x, self.drop_prob, self.training)  # 调用drop_path函数处理输入


class DropKey(nn.Module):
    """
    DropKey 类：实现了在训练过程中随机丢弃注意力键的功能，这是一种正则化技术，有助于防止模型过拟合。
    该模块通过在注意力分数中添加一个很小的负值（-1e12）来模拟键的丢弃效果，这样在softmax计算后，
    被丢弃的键对应的注意力权重会趋近于零。
    """

    def __init__(self, p=0.):
        """
        初始化 DropKey 模块
        参数:
            p (float): 丢弃概率，默认值为0.0，表示不丢弃任何键。
                      当p=0.5时，表示每个键有50%的概率被丢弃。
        """
        super(DropKey, self).__init__()
        self.p = p

    def forward(self, attn):
        """
        前向传播函数
        参数:
            attn (Tensor): 输入的注意力分数矩阵，形状通常为 [batch_size, num_heads, seq_len, seq_len]
        返回:
            Tensor: 经过可能的键丢弃操作后的注意力分数矩阵
        工作原理:
        1. 仅在训练模式下应用丢弃操作
        2. 创建一个与注意力矩阵相同形状的掩码矩阵m_r，其值为p
        3. 使用伯努利分布生成一个二值掩码，与-1e12相乘
        4. 将结果加到原始注意力分数上，实现"丢弃"效果
        """
        if self.training:
            m_r = torch.ones_like(attn) * self.p  # 创建一个与注意力矩阵相同形状的矩阵，所有值为p
            attn = attn + torch.bernoulli(m_r) * -1e12  # 根据伯努利分布随机选择位置，添加一个很小的负值
        return attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        多层感知机(MLP)类的初始化函数
        参数:
            in_features: 输入特征的维度
            hidden_features: 隐藏层特征的维度，默认为输入特征的维度
            out_features: 输出特征的维度，默认为输入特征的维度
            act_layer: 激活函数层，默认为GELU
            drop: dropout比例，默认为0
        """
        super().__init__()
        out_features = out_features or in_features  # 如果未指定输出特征维度，则使用输入特征维度
        hidden_features = hidden_features or in_features  # 如果未指定隐藏层特征维度，则使用输入特征维度
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层
        self.act = act_layer()  # 激活函数层
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二个全连接层
        self.drop = nn.Dropout(drop)  # dropout层

    def forward(self, x):
        """
        前向传播函数
        参数:
            x: 输入张量
        返回:
            处理后的输出张量
        """
        x = self.fc1(x)  # 通过第一个全连接层
        x = self.act(x)  # 应用激活函数
        x = self.drop(x)  # 应用dropout
        x = self.fc2(x)  # 通过第二个全连接层
        x = self.drop(x)  # 再次应用dropout
        return x  # 返回最终输出


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., qk_norm=None):
        """
        注意力机制的初始化函数
        参数:
            dim: 输入特征的维度
            num_heads: 注意力头的数量，默认为8
            qkv_bias: 在查询、键、值线性层中是否使用偏置，默认为False
            qk_scale: 缩放因子，默认为None
            attn_drop: 注意力权重的dropout率，默认为0
            proj_drop: 输出投影的dropout率，默认为0
            qk_norm: 查询和键的归一化方法，默认为None
        """
        super().__init__()
        self.num_heads = num_heads  # 设置注意力头的数量
        head_dim = dim // num_heads  # 计算每个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 设置缩放因子，如果没有提供则使用默认值

        # 初始化查询、键、值的线性变换层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)  # 原始的注意力dropout层
        self.attn_dropkey = DropKey(attn_drop)  # 使用DropKey替代原始的注意力dropout

        # 初始化输出投影层和dropout
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 如果提供了qk_norm，则初始化查询和键的归一化层
        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)  # 查询的归一化层
            self.k_norm = qk_norm(head_dim)  # 键的归一化层
            self.qk_norm = True  # 标记启用了qk归一化
        else:
            self.qk_norm = False  # 标记未启用qk归一化

    def forward(self, x):

        """
        注意力机制的前向传播函数

        参数:
            x: 输入张量，形状为(B, N, C)，其中B是批次大小，N是序列长度，C是特征维度

        返回:
            x: 经过注意力处理后的输出张量，形状为(B, N, C)
            attn: 注意力权重矩阵，形状为(B, num_heads, N, N)
        """
        B, N, C = x.shape  # 获取输入张量的批次大小、序列长度和特征维度
        # 将输入通过qkv线性层，然后重塑并转置为(3, B, num_heads, N, head_dim)的形状
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离查询、键和值

        # 如果启用了qk归一化，则对查询和键进行归一化
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 计算注意力权重矩阵
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算注意力分数
        attn = self.attn_dropkey(attn)  # 应用DropKey
        attn = attn.softmax(dim=-1)  # 应用softmax归一化
        # attn = self.attn_drop(attn)  # 原始的注意力dropout

        # 将注意力权重应用到值上，并通过输出投影层
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 聚合值
        x = self.proj(x)  # 输出投影
        x = self.proj_drop(x)  # 应用输出dropout
        return x, attn  # 返回输出和注意力权重


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_norm=None):
        """
        初始化Block模块
        参数:
            dim: 输入特征的维度
            num_heads: 多头注意力机制的头数
            mlp_ratio: MLP隐藏层维度的比例，默认为4
            qkv_bias: 是否在QKV投影中添加偏置，默认为False
            qk_scale: 缩放因子，用于缩放QK点积，默认为None
            drop: dropout率，默认为0
            attn_drop: 注意力dropout率，默认为0
            drop_path: drop path率，默认为0
            act_layer: 激活函数，默认为nn.GELU
            norm_layer: 归一化层，默认为nn.LayerNorm
            qk_norm: 是否对QK进行归一化，默认为None
        """
        super().__init__()
        # 第一个归一化层
        self.norm1 = norm_layer(dim)
        # 注意力机制层
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            qk_norm=qk_norm)
        # drop path层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二个归一化层
        self.norm2 = norm_layer(dim)
        # 计算MLP隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP层
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        """
        前向传播函数
        参数:
            x: 输入张量
            return_attention: 是否返回注意力权重，默认为False
        返回:
            如果return_attention为True，返回输出张量和注意力权重
            否则，仅返回输出张量
        """
        # 应用注意力机制
        y, attn = self.attn(self.norm1(x))
        # 残差连接和drop path
        x = x + self.drop_path(y)
        # MLP处理和残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # 根据参数决定是否返回注意力权重
        if return_attention:
            return x, attn
        else:
            return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    将图像分割成小块并嵌入到向量空间中的模块
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        # 初始化函数，设置图像大小、块大小、输入通道数和嵌入维度
        super().__init__()
        # 计算图像可以被分割成多少个块
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        # 存储图像大小
        self.img_size = img_size
        # 存储块大小
        self.patch_size = patch_size
        # 存储网格大小，即图像在高度和宽度方向上可以分割的块数
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        # 存储总块数
        self.num_patches = num_patches

        # 创建一个二维卷积层，用于将图像块嵌入到指定维度
        # 输入通道数为in_chans，输出通道数为embed_dim
        # 卷积核大小和步长都为patch_size，这样每个卷积操作正好提取一个图像块
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 前向传播函数
        # 获取输入张量的批次大小和通道数
        B, C, H, W = x.shape
        # 使用卷积层处理输入图像，然后展平最后两个维度，并转置维度顺序
        # 最终输出形状为 (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer  (ViT) 模型类
    实现基于Transformer的视觉模型，将图像分割成patch序列并使用Transformer进行处理
    """

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # 设置特征维度

        # 图像patch嵌入层
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # 计算patch数量

        # 分类token (class token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # 位置dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h, use_sinusoid=False):
        """
        对位置编码进行插值处理，以适应不同尺寸的输入

        参数:
            x: 输入张量
            w: 目标宽度
            h: 目标高度
            use_sinusoid: 是否使用正弦位置编码，默认为False

        返回:
            插值后的位置编码
        """
        npatch = x.shape[1] - 1  # 计算补丁数量
        N = self.pos_embed.shape[1] - 1  # 原始位置编码的补丁数量
        dim = x.shape[-1]  # 特征维度
        # 如果输入尺寸与原始位置编码尺寸相同，直接返回原始位置编码
        if npatch == N and w == h:
            return self.pos_embed

        # print("Interpolate positional encoding...")
        if not use_sinusoid:

            # 不使用正弦位置编码的情况
            class_pos_embed = self.pos_embed[:, 0]  # 分类token的位置编码
            patch_pos_embed = self.pos_embed[:, 1:]  # 补丁的位置编码
            # 计算目标尺寸与原始尺寸的比例
            w0 = w // self.patch_embed.patch_size
            h0 = h // self.patch_embed.patch_size
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                recompute_scale_factor=False,
                align_corners=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        else:

            def build_2d_sincos_position_embedding(h, w, temperature=10000.):
                h //= self.patch_embed.patch_size
                w //= self.patch_embed.patch_size
                grid_w = torch.arange(w, dtype=torch.float32)
                grid_h = torch.arange(h, dtype=torch.float32)
                grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
                assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
                pos_dim = self.embed_dim // 4
                omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
                omega = 1. / (temperature ** omega)
                out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
                out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
                pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[
                          None, :, :]

                # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
                pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
                return torch.cat([pe_token, pos_emb], dim=1)

            pe = build_2d_sincos_position_embedding(h, w).cuda()
            return pe

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1, norm=False):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_all_selfattention(self, x):
        """Get a self-attention matrix from every layer."""
        x = self.prepare_tokens(x)
        attns = []

        for blk in self.blocks:
            attns.append(blk(x, return_attention=True))
            x = blk(x)

        return attns


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        """
        DINOHead类的初始化函数
        参数:
            in_dim: 输入特征的维度
            out_dim: 输出特征的维度
            use_bn: 是否使用批归一化
            norm_last_layer: 是否对最后一层进行归一化
            nlayers: MLP的层数
            hidden_dim: 隐藏层的维度
            bottleneck_dim: 瓶颈层的维度
        """
        super().__init__()
        nlayers = max(nlayers, 1)  # 确保至少有一层
        if nlayers == 1:
            # 如果只有一层，直接创建一个线性层
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            # 如果有多层，构建一个MLP序列
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            # 添加隐藏层
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)  # 应用权重初始化
        # 使用权重归一化的最后一层
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)  # 初始化权重g为1
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False  # 如果需要归一化，则冻结梯度

    def _init_weights(self, m):
        """
        权重初始化函数
        参数:
            m: 需要初始化权重的模块
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # 使用截断正态分布初始化权重
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为0

    def forward(self, x):

        """
        前向传播函数
        参数:
            x: 输入特征
        返回:
            x: 经过MLP、归一化和最后一层处理后的输出特征
        """
        x = self.mlp(x)  # 通过MLP
        x = nn.functional.normalize(x, dim=-1, p=2)  # 进行L2归一化
        x = self.last_layer(x)  # 通过最后一层
        return x
