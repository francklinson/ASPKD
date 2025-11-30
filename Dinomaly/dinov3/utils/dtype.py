from typing import Dict, Union

import numpy as np
import torch

TypeSpec = Union[str, np.dtype, torch.dtype]

_NUMPY_TO_TORCH_DTYPE: Dict[np.dtype, torch.dtype] = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}


def as_torch_dtype(dtype: TypeSpec) -> torch.dtype:
    """
    将输入的数据类型转换为PyTorch的dtype类型

    参数:
        dtype: 输入的数据类型，可以是torch.dtype、字符串或numpy.dtype类型

    返回:
        torch.dtype: 转换后的PyTorch数据类型

    异常:
        AssertionError: 当输入既不是torch.dtype、字符串也不是numpy.dtype时抛出
    """
    if isinstance(dtype, torch.dtype):  # 如果已经是torch.dtype类型，直接返回
        return dtype
    if isinstance(dtype, str):  # 如果是字符串类型，先转换为numpy.dtype
        dtype = np.dtype(dtype)
    assert isinstance(dtype, np.dtype), f"Expected an instance of nunpy dtype, got {type(dtype)}"  # 确保转换后是numpy.dtype类型
    return _NUMPY_TO_TORCH_DTYPE[dtype]  # 通过预定义的映射关系将numpy.dtype转换为torch.dtype
