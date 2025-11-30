import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from dinov3.layers.fp8_linear import convert_linears_to_fp8
from . import vision_transformer as vits

logger = logging.getLogger("dinov3")


def init_fp8(model: nn.Module, args) -> nn.Module:
    """
    初始化FP8（8位浮点）功能，根据配置决定是否启用FP8矩阵乘法
    参数:
        model (nn.Module): 要转换的神经网络模型
        args: 配置参数对象，包含FP8相关配置
    返回:
        nn.Module: 处理后的模型，如果FP8启用则转换为FP8格式，否则返回原模型
    """
    # 检查配置中是否启用了FP8
    if not args.fp8_enabled:
        logger.info("fp8 matmuls: OFF (disabled in config)")
        return model
    # 如果FP8已启用，记录日志信息
    logger.info("fp8 matmuls: ON")
    # Multi-kernel makes Inductor auto-tune between a regular "streaming"-based
    # reduction kernel and a "persistent" reduction kernel. Since fp8 has some
    # multi-pass steps (e.g., first get amax, then scale), persistent kernels
    # should perform better.
    torch._inductor.config.triton.multi_kernel = 1
    return convert_linears_to_fp8(model, filter=args.fp8_filter)


def build_model(args, only_teacher=False, img_size=224, device=None):
    """
    根据给定的参数构建模型
    参数:
        args: 配置参数对象，包含模型架构和各项参数
        only_teacher: 是否只构建教师模型，默认为False
        img_size: 输入图像的尺寸，默认为224
        device: 指定模型运行的设备，如CPU或GPU
    返回:
        如果only_teacher为True，返回教师模型和其嵌入维度
        否则返回学生模型、教师模型和嵌入维度
    异常:
        当架构不支持时抛出NotImplementedError
    """
    if "vit" in args.arch:
        # 准备Vision Transformer模型的关键参数
        vit_kwargs = dict(
            img_size=img_size,  # 输入图像尺寸
            patch_size=args.patch_size,  # 图像块大小
            pos_embed_rope_base=args.pos_embed_rope_base,  # 旋转位置编码的基础值
            pos_embed_rope_min_period=args.pos_embed_rope_min_period,  # 旋转位置编码的最小周期
            pos_embed_rope_max_period=args.pos_embed_rope_max_period,  # 旋转位置编码的最大周期
            pos_embed_rope_normalize_coords=args.pos_embed_rope_normalize_coords,  # 是否归一化坐标
            pos_embed_rope_shift_coords=args.pos_embed_rope_shift_coords,  # 是否偏移坐标
            pos_embed_rope_jitter_coords=args.pos_embed_rope_jitter_coords,  # 是否抖动坐标
            pos_embed_rope_rescale_coords=args.pos_embed_rope_rescale_coords,  # 是否重新缩放坐标
            qkv_bias=args.qkv_bias,  # QKV矩阵是否使用偏置
            layerscale_init=args.layerscale,  # 层级缩放的初始值
            norm_layer=args.norm_layer,  # 归一化层类型
            ffn_layer=args.ffn_layer,  # 前馈网络层类型
            ffn_bias=args.ffn_bias,  # 前馈网络是否使用偏置
            proj_bias=args.proj_bias,  # 投影层是否使用偏置
            n_storage_tokens=args.n_storage_tokens,  # 存储令牌数量
            mask_k_bias=args.mask_k_bias,  # 掩码K偏置
            untie_cls_and_patch_norms=args.untie_cls_and_patch_norms,  # 是否解耦类别和块归一化
            untie_global_and_local_cls_norm=args.untie_global_and_local_cls_norm,  # 是否解耦全局和局部类别归一化
            device=device,  # 运行设备
        )
        # 创建教师模型
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        teacher = init_fp8(teacher, args)  # 初始化FP8精度
        # 如果只需要教师模型，则直接返回
        if only_teacher:
            return teacher, teacher.embed_dim
        # 创建学生模型，添加额外的dropout路径率参数
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,  # 随机深度率
        )
        embed_dim = student.embed_dim  # 获取嵌入维度
    else:
        # 不支持的架构抛出异常
        raise NotImplementedError(f"Unrecognized architecture {args.arch}")
    # 初始化学生模型的FP8精度
    student = init_fp8(student, args)
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher: bool = False):
    outputs = build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=cfg.crops.global_crops_size
        if isinstance(cfg.crops.global_crops_size, int)
        else max(cfg.crops.global_crops_size),
        device="meta",
    )
    if only_teacher:
        teacher, embed_dim = outputs
        return teacher, embed_dim
    else:
        student, teacher, embed_dim = outputs
        return student, teacher, embed_dim


def build_model_for_eval(
        config,
        pretrained_weights: Union[str, Path] | None,
        shard_unsharded_model: bool = False,
        # If the model is not sharded, shard it. No effect if already sharded on disk
):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if pretrained_weights is None or pretrained_weights == "":
        logger.info("No pretrained weights")
        model.init_weights()
    elif Path(pretrained_weights).is_dir():
        logger.info("PyTorch DCP checkpoint")
        from dinov3.checkpointer import load_checkpoint
        from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize

        moduledict = nn.ModuleDict({"backbone": model})
        # Wrap with FSDP
        ac_compile_parallelize(moduledict, inference_only_models=[], cfg=config)
        # Move to CUDA
        model.to_empty(device="cuda")
        # Load checkpoint
        load_checkpoint(pretrained_weights, model=moduledict, strict_loading=True)
        shard_unsharded_model = False
    else:
        logger.info("PyTorch consolidated checkpoint")
        from dinov3.checkpointer import init_model_from_checkpoint_for_evals

        # consolidated checkpoint codepath
        model.to_empty(device="cuda")
        init_model_from_checkpoint_for_evals(model, pretrained_weights, "teacher")
    if shard_unsharded_model:
        logger.info("Sharding model")
        moduledict = nn.ModuleDict({"backbone": model})
        ac_compile_parallelize(moduledict, inference_only_models=[], cfg=config)
    model.eval()
    return model
