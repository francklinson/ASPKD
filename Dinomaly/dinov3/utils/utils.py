import logging
import os
import random
import subprocess
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger("dinov3")


def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int]], List[int]]:
    """
    将多个张量拼接成一个张量，同时保留它们的原始形状和token数量信息
    参数:
        x_list: 待拼接的张量列表
    返回:
        Tuple[Tensor, List[Tuple[int]], List[int]]:
            - 拼接后的扁平化张量
            - 每个原始张量的形状列表
            - 每个张量的token数量列表
    """
    # 获取输入列表中每个张量的形状信息
    shapes = [x.shape for x in x_list]
    # 计算每个张量的token数量（选择最后一个维度的第一个索引，然后计算元素总数）
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    # 将每个张量扁平化（从第0维到倒数第2维），然后拼接所有扁平化后的张量
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: List[Tuple[int]], num_tokens: List[int]) -> List[Tensor]:
    """
    将一个展平的张量根据给定的形状和token数量重新分割并重塑为多个张量
    参数:
        flattened: 输入的展平张量
        shapes: 每个输出张量的目标形状列表(最后一个维度除外)
        num_tokens: 每个输出张量在第一个维度上的大小列表
    返回:
        重新分割和重塑后的张量列表
    """
    # 根据num_tokens将输入张量分割为多个小张量
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    # 调整形状，将原始形状的最后一个维度替换为输入张量的最后一个维度大小
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    # 将分割后的每个张量调整为调整后的形状
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped


def named_replace(
        fn: Callable,
        module: nn.Module,
        name: str = "",
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Module:
    """
    递归遍历神经网络模块，并对每个模块应用指定的函数
    参数:
        fn: 应用于每个模块的函数，接收module和name作为参数
        module: 要处理的神经网络模块
        name: 当前模块的名称，用于构建层次名称
        depth_first: 是否深度优先处理，默认为True
        include_root: 是否包含根模块，默认为False
    返回:
        处理后的神经网络模块
    """
    if not depth_first and include_root:
        # 如果不是深度优先且包含根模块，则先处理当前模块
        module = fn(module=module, name=name)
    for child_name_o, child_module in list(module.named_children()):
        # 遍历所有子模块
        child_name = ".".join((name, child_name_o)) if name else child_name_o  # 构建子模块的完整名称
        new_child = named_replace(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
        setattr(module, child_name_o, new_child)  # 更新模块

    if depth_first and include_root:
        # 如果是深度优先且包含根模块，则在处理完所有子模块后处理当前模块
        module = fn(module=module, name=name)
    return module


def named_apply(
        fn: Callable,  # 可调用函数，将应用于每个模块
        module: nn.Module,  # 要遍历的PyTorch神经网络模块
        name: str = "",  # 当前模块的名称，用于构建层次化名称
        depth_first: bool = True,  # 是否深度优先遍历模块
        include_root: bool = False,  # 是否在遍历过程中应用函数到根模块
) -> nn.Module:  # 返回处理后的模块
    if not depth_first and include_root:
        # 如果不是深度优先遍历且包含根模块，则先对当前模块应用函数
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        # 遍历当前模块的所有子模块
        child_name = ".".join((name, child_name)) if name else child_name  # 构建子模块的完整名称
        named_apply(
            fn=fn,  # 传递可调用函数
            module=child_module,  # 处理子模块
            name=child_name,  # 传递子模块名称
            depth_first=depth_first,  # 保持遍历策略
            include_root=True,  # 在递归调用中包含根模块
        )
    if depth_first and include_root:
        # 如果是深度优先遍历且包含根模块，则在处理完所有子模块后对当前模块应用函数
        fn(module=module, name=name)
    return module  # 返回处理后的模块


def fix_random_seeds(seed: int = 31):
    """
    Fix random seeds.
    该函数用于设置随机数种子，确保实验的可重复性。通过设置不同库的随机种子，
    可以保证每次运行程序时生成的随机数序列是一致的，这对于实验结果的复现非常重要。
    参数:
        seed (int, optional): 随机数种子，默认为31
    使用说明:
        在程序开始处调用此函数，可以确保后续所有使用随机数的操作结果可复现
    """
    # 设置PyTorch的CPU随机种子
    torch.manual_seed(seed)
    # 设置PyTorch的所有GPU的随机种子
    torch.cuda.manual_seed_all(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置Python内置的随机种子
    random.seed(seed)


def get_sha() -> str:
    # 获取当前文件所在目录的绝对路径
    cwd = os.path.dirname(os.path.abspath(__file__))

    # 定义内部函数，用于执行命令并返回输出结果
    def _run(command):
        # 执行命令，获取输出并解码为ASCII字符串，去除首尾空白字符
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    # 初始化变量，默认值分别为"N/A"、"clean"和"N/A"
    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        # 获取当前git commit的SHA值
        sha = _run(["git", "rev-parse", "HEAD"])
        # 检查是否有未提交的更改
        subprocess.check_output(["git", "diff"], cwd=cwd)
        # 获取与HEAD的差异索引
        diff = _run(["git", "diff-index", "HEAD"])
        # 如果有差异，则设置diff为"has uncommited changes"，否则为"clean"
        diff = "has uncommited changes" if diff else "clean"
        # 获取当前分支名称
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        # 如果发生任何异常，则忽略（保持默认值）
        pass
    # 格式化并返回包含SHA、状态和分支信息的消息
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def get_conda_env() -> Tuple[Optional[str], Optional[str]]:
    """
    获取当前conda环境的相关信息
    该函数通过读取系统环境变量来获取当前激活的conda环境名称和路径。
    它会返回一个包含两个元素的元组，分别是环境名称和环境路径。
    Returns:
        Tuple[Optional[str], Optional[str]]: 返回一个元组，包含两个元素:
            - 第一个元素是conda环境名称(如果有的话)，否则为None
            - 第二个元素是conda环境路径(如果有的话)，否则为None
    """
    # 从系统环境变量中获取当前激活的conda环境名称
    conda_env_name = os.environ.get("CONDA_DEFAULT_ENV")
    # 从系统环境变量中获取当前conda环境的基础路径
    conda_env_path = os.environ.get("CONDA_PREFIX")
    # 返回环境名称和路径组成的元组
    return conda_env_name, conda_env_path


def count_parameters(module: nn.Module) -> int:
    """
    计算神经网络模型中参数的总数量

    参数:
        module (nn.Module): PyTorch神经网络模块

    返回:
        int: 模型中所有参数的总数量
    """
    c = 0  # 参数计数器初始化为0
    for m in module.parameters():  # 遍历模型中的所有参数
        c += m.nelement()  # 累加当前参数的元素数量
    return c  # 返回总参数数量


def has_batchnorms(model: nn.Module) -> bool:
    """
    检查模型中是否包含任何批归一化(BatchNorm)层
    参数:
        model (nn.Module): 要检查的PyTorch模型
    返回:
        bool: 如果模型中包含任何BatchNorm层则返回True，否则返回False
    该函数通过遍历模型的所有模块(named_modules)，
    检查是否存在任何类型的BatchNorm层
    """
    # 定义所有可能的BatchNorm层类型元组
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    # 遍历模型中的所有模块
    for _, module in model.named_modules():
        # 检查当前模块是否为BatchNorm类型
        if isinstance(module, bn_types):
            # 如果找到任何BatchNorm层，立即返回True
            return True
    # 如果遍历完所有模块都没有找到BatchNorm层，返回False
    return False
