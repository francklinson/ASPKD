import warnings
import torch.nn.functional as F


def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    """
    调整输入张量的大小
    参数:
        input (Tensor): 输入张量
        size (tuple or int): 目标输出大小 (height, width)
        scale_factor (float or tuple): 缩放因子
        mode (str): 插值模式，可选 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        align_corners (bool): 是否对齐角点
        warning (bool): 是否显示警告信息
    返回:
        Tensor: 调整大小后的张量
    """
    if warning:
        if size is not None and align_corners:
            # 获取输入和输出的高度和宽度
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            # 检查输出尺寸是否大于输入尺寸
            if output_h > input_h or output_w > output_h:
                # 检查尺寸是否满足特定条件
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    # 显示警告信息
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    # 使用F.interpolate进行插值调整大小
    return F.interpolate(input, size, scale_factor, mode, align_corners)
