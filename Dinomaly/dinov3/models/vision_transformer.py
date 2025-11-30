import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from dinov3.layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from dinov3.utils import named_apply

# 创建日志记录器
logger = logging.getLogger("dinov3")

# 定义不同类型的FFN层（前馈网络）
ffn_layer_dict = {
    "mlp": Mlp,  # 标准MLP层
    "swiglu": SwiGLUFFN,  # SwiGLU激活函数的FFN层
    "swiglu32": partial(SwiGLUFFN, align_to=32),  # 对齐到32的SwiGLU FFN层
    "swiglu64": partial(SwiGLUFFN, align_to=64),  # 对齐到64的SwiGLU FFN层
    "swiglu128": partial(SwiGLUFFN, align_to=128),  # 对齐到128的SwiGLU FFN层
}

# 定义不同类型的归一化层
norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),  # 标准LayerNorm
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),  # 适用于bf16的LayerNorm
    "rmsnorm": RMSNorm,  # RMS归一化层
}

# 定义不同的数据类型
dtype_dict = {
    "fp32": torch.float32,  # 32位浮点数
    "fp16": torch.float16,  # 16位浮点数
    "bf16": torch.bfloat16,  # 脑浮点数16位
}


def init_weights_vit(module: nn.Module, name: str = ""):
    """
    初始化Vision Transformer模型中各层权重的函数
    Args:
        module (nn.Module): 需要初始化权重的神经网络模块
        name (str): 模块名称，目前未在函数中使用
    该函数根据不同的模块类型执行不同的初始化策略:
    1. 对于线性层(Linear)，使用截断正态分布初始化权重，偏置置零
    2. 对于各种归一化层(LayerNorm, LayerScale, PatchEmbed, RMSNorm)，调用其自身的reset_parameters方法进行初始化
    """
    if isinstance(module, nn.Linear):
        # 使用标准差为0.02的截断正态分布初始化线性层的权重
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        # 如果存在偏置项，则将其初始化为零
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        # 重置LayerNorm层的参数
        module.reset_parameters()
    if isinstance(module, LayerScale):
        # 重置LayerScale层的参数
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        # 重置PatchEmbed层的参数
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        # 重置RMSNorm层的参数
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
            self,
            *,
            img_size: int = 224,  # 输入图像的尺寸
            patch_size: int = 16,  # 图像块的尺寸
            in_chans: int = 3,  # 输入图像的通道数
            pos_embed_rope_base: float = 100.0,  # 旋转位置编码的基础值
            pos_embed_rope_min_period: float | None = None,  # 旋转位置编码的最小周期
            pos_embed_rope_max_period: float | None = None,  # 旋转位置编码的最大周期
            pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",  # 坐标归一化方法
            pos_embed_rope_shift_coords: float | None = None,  # 坐标偏移量
            pos_embed_rope_jitter_coords: float | None = None,  # 坐标抖动量
            pos_embed_rope_rescale_coords: float | None = None,  # 坐标重缩放量
            pos_embed_rope_dtype: str = "bf16",  # 旋转位置编码的数据类型
            embed_dim: int = 768,  # 嵌入维度
            depth: int = 12,  # Transformer层数
            num_heads: int = 12,  # 注意力头数
            ffn_ratio: float = 4.0,  # FFN层的扩展比例
            qkv_bias: bool = True,  # QKV是否使用偏置
            drop_path_rate: float = 0.0,  # 随机路径丢弃率
            layerscale_init: float | None = None,  # 层缩放初始化值
            norm_layer: str = "layernorm",  # 归一化层类型
            ffn_layer: str = "mlp",  # FFN层类型
            ffn_bias: bool = True,  # FFN层是否使用偏置
            proj_bias: bool = True,  # 投影层是否使用偏置
            n_storage_tokens: int = 0,  # 存储令牌数量
            mask_k_bias: bool = False,  # 掩码K偏置
            untie_cls_and_patch_norms: bool = False,  # 是否解绑CLS和补丁的归一化
            untie_global_and_local_cls_norm: bool = False,  # 是否解绑全局和局部CLS的归一化
            device: Any | None = None,  # 设备
            **kwargs,  # 其他关键字参数
    ):
        super().__init__()

        norm_layer_cls = norm_layer_dict[norm_layer]  # 获取归一化层类

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        # 初始化图像块嵌入层
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        # 初始化CLS令牌
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        # 记录旋转位置编码的配置信息
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")

        # 初始化旋转位置编码
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )

        # 记录使用的FFN层类型
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        # 创建Transformer块列表
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))
        self.layers_to_extract_from = kwargs.get('layers_to_extract_from', None)

    def init_weights(self):
        """
        初始化模型权重
        """
        self.rope_embed._init_weights()  # 初始化旋转位置嵌入(rope)的权重
        nn.init.normal_(self.cls_token, std=0.02)  # 使用均值为0，标准差为0.02的正态分布初始化分类令牌(cls_token)
        if self.n_storage_tokens > 0:  # 如果存储令牌数量大于0
            nn.init.normal_(self.storage_tokens, std=0.02)  # 使用均值为0，标准差为0.02的正态分布初始化存储令牌(storage_tokens)
        nn.init.zeros_(self.mask_token)  # 将掩码令牌(mask_token)初始化为零
        named_apply(init_weights_vit, self)  # 对Vision Transformer(ViT)模型应用特定的权重初始化函数

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        """
        准备带有掩码的令牌（tokens）函数
        该函数处理输入的张量x，添加类别令牌(cls_token)和存储令牌(storage_tokens)，
        并根据需要应用掩码。返回处理后的令牌和原始输入的空间维度。
        Args:
            x:
            masks:

        Returns:

        """
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token

        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1:])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1:]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        """
        前向传播特征提取函数
        该函数处理输入的特征张量，提取并返回特征字典列表。根据输入类型的不同，可以选择直接处理单个张量或批量处理多个张量。
        参数:
            x: 输入特征张量或张量列表，可以是单个torch.Tensor或Tensor列表
            masks: 可选的掩码张量，用于特征处理时的掩码操作
        返回:
            包含特征字典的列表，每个字典存储了不同名称的特征张量
        处理逻辑:
            1. 判断输入x是否为单个torch.Tensor
            2. 如果是单个张量，将其转换为列表形式，调用forward_features_list处理并返回第一个结果
            3. 如果是张量列表，直接调用forward_features_list处理
        """
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]  # 处理单个张量情况
        else:
            return self.forward_features_list(x, masks)  # 处理张量列表情况

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:

        """
        获取模型中间层的输出，不进行分块处理

        参数:
            x (Tensor): 输入张量
            n (int): 如果是整数，表示获取最后n个块的输出；如果是列表，表示获取指定索引的块的输出

        返回:
            List[Tensor]: 包含指定层输出的张量列表
        """
        x, (H, W) = self.prepare_tokens_with_masks(x)  # 准备输入token并获取高度和宽度信息
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
            self,
            x: torch.Tensor,  # 输入的张量
            *,
            n: Union[int, Sequence] = 1,  # Layers or n last layers to take
            reshape: bool = False,
            return_class_token: bool = False,
            return_extra_tokens: bool = False,
            norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1:])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1: self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1:] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=16, **kwargs):
    """
    创建一个小型的Vision Transformer (ViT) 模型
    参数:
        patch_size (int): 图像分割的块大小，默认为16
        **kwargs: 其他传递给DinoVisionTransformer的关键字参数
    返回:
        DinoVisionTransformer: 配置好的小型ViT模型实例
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,  # 设置图像块大小
        embed_dim=384,  # 嵌入维度，决定特征表示的向量大小
        depth=12,  # Transformer层数，即有多少个Transformer块
        num_heads=6,  # 多头注意力机制中的头数
        ffn_ratio=4,  # FFN层隐藏维度与嵌入维度的比例
        **kwargs,  # 接收其他可选参数
    )
    return model


def vit_base(patch_size=16, **kwargs):
    """
    创建并返回一个基础的Vision Transformer (ViT) 模型实例
    参数:
        patch_size (int): 图像分割的块大小，默认为16
        **kwargs: 其他传递给DinoVisionTransformer的参数
    返回:
        DinoVisionTransformer: 配置好的Vision Transformer模型实例
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,  # 设置图像块大小
        embed_dim=768,  # 嵌入维度，表示每个块的向量表示大小
        depth=12,  # Transformer层数，即块的堆叠数量
        num_heads=12,  # 多头注意力机制中的头数
        ffn_ratio=4,  # 前馈网络的扩展比例
        **kwargs,  # 其他可选参数
    )
    return model


def vit_large(patch_size=16, **kwargs):
    """
    创建一个大型Vision Transformer (ViT) 模型的函数
    参数:
        patch_size (int): 图像块的大小，默认为16
        **kwargs: 其他传递给DinoVisionTransformer的关键字参数
    返回:
        DinoVisionTransformer: 配置好的大型ViT模型实例
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,  # 设置图像块大小
        embed_dim=1024,  # 嵌入维度，设置为1024
        depth=24,  # Transformer层数，设置为24层
        num_heads=16,  # 注意力头的数量，设置为16
        ffn_ratio=4,  # 前馈网络的扩展比例，设置为4
        **kwargs,  # 接收其他可选参数
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    """
    创建并返回一个Vision Transformer (ViT) SO-400M模型实例

    参数:
        patch_size (int): 图像块的大小，默认为16
        **kwargs: 其他传递给DinoVisionTransformer的关键字参数

    返回:
        DinoVisionTransformer: 配置好的SO-400m模型实例
    """
    model = DinoVisionTransformer(  # 实例化DinoVisionTransformer模型
        patch_size=patch_size,  # 设置图像块大小
        embed_dim=1152,  # 设置嵌入维度为1152
        depth=27,  # 设置网络深度为27层
        num_heads=18,  # 设置注意力头数量为18
        ffn_ratio=3.777777778,  # 设置FFN层的比例因子
        **kwargs,  # 传递其他参数
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    """
    创建一个huge的DinoVisionTransformer模型实例
    参数:
        patch_size (int): 图像分块的大小，默认为16
        **kwargs: 其他传递给DinoVisionTransformer的参数
    返回:
        DinoVisionTransformer: 配置好的巨大模型实例
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,  # 设置图像分块大小
        embed_dim=1280,  # 设置嵌入维度为1280
        depth=32,  # 设置网络深度为32层
        num_heads=20,  # 设置注意力头的数量为20
        ffn_ratio=4,  # 设置FFN层的比例因子为4
        **kwargs,  # 传递其他未指定的参数
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    """
    创建一个7B版本的Vision Transformer (ViT) 模型
    参数:
        patch_size (int): 图像分割的块大小，默认为16
        **kwargs: 其他传递给DinoVisionTransformer的关键字参数
    返回:
        DinoVisionTransformer: 配置好的7B参数规模的ViT模型
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,  # 图像块大小，将图像分割成patch进行处理
        embed_dim=4096,  # 嵌入维度，决定特征表示的向量大小
        depth=40,  # Transformer编码器的层数
        num_heads=32,  # 多头注意力机制中的头数
        ffn_ratio=3,  # 前馈网络的扩展比例
        **kwargs,  # 其他可选参数
    )
    return model
