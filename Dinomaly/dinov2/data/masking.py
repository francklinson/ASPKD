import math
import random

import numpy as np


class MaskingGenerator:
    def __init__(
        self,
        input_size,          # 输入尺寸，可以是单个整数或元组
        num_masking_patches=None,  # 要生成的掩码块数量
        min_num_patches=4,    # 最小掩码块数量
        max_num_patches=None, # 最大掩码块数量
        min_aspect=0.3,       # 最小宽高比
        max_aspect=None,     # 最大宽高比
    ):
        """
        初始化函数，用于设置掩码生成器的参数
        参数:
            input_size: 输入尺寸，可以是单个整数或元组
            num_masking_patches: 要生成的掩码块数量
            min_num_patches: 最小掩码块数量，默认为4
            max_num_patches: 最大掩码块数量，默认为None
            min_aspect: 最小宽高比，默认为0.3
            max_aspect: 最大宽高比，默认为None
        """
        # 如果输入不是元组，则转换为相同高度的方形
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        # 计算总块数
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        # 设置最小和最大掩码块数
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        # 计算宽高比的对数范围
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        # 返回生成器的字符串表示形式
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        # 返回生成器的高度和宽度
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        # 内部方法，用于生成单个掩码区域
        delta = 0
        for _ in range(10):  # 最多尝试10次
            # 随机生成目标区域大小和宽高比
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))  # 计算高度
            w = int(round(math.sqrt(target_area / aspect_ratio)))  # 计算宽度
            # 确保生成的区域不超过图像边界
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)  # 随机选择起始行
                left = random.randint(0, self.width - w)  # 随机选择起始列

                # 计算当前区域中已被掩码的像素数量
                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        """
        该方法用于生成一个布尔类型的掩码矩阵，用于指定需要被遮盖的图像块位置。
        参数:
            num_masking_patches (int, optional): 需要遮盖的图像块数量，默认为0。
        返回:
            numpy.ndarray: 一个与图像形状相同的布尔类型数组，True表示需要被遮盖的位置。
        """
    # 初始化一个与图像形状相同的全False布尔数组作为掩码
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
    # 记录已遮盖的图像块数量
        mask_count = 0
    # 循环直到达到目标遮盖数量或无法再添加遮盖块
        while mask_count < num_masking_patches:
        # 计算本次循环中最多可以遮盖的图像块数量
            max_mask_patches = num_masking_patches - mask_count
        # 确保不超过单次最大遮盖限制
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

        # 调用内部方法生成遮盖，并返回实际遮盖的图像块数量
            delta = self._mask(mask, max_mask_patches)
        # 如果无法再添加遮盖块，则退出循环
            if delta == 0:
                break
            else:
            # 更新已遮盖的图像块数量
                mask_count += delta

    # 返回生成的掩码
        return mask
