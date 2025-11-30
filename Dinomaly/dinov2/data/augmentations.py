import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
            self,
            global_crops_scale,  # 全局裁剪的缩放范围
            local_crops_scale,  # 局部裁剪的缩放范围
            local_crops_number,  # 局部裁剪的数量
            global_crops_size=224,  # 全局裁剪的尺寸大小，默认为224
            local_crops_size=96,  # 局部裁剪的尺寸大小，默认为96
    ):
        """
        初始化函数，设置数据增强的相关参数和转换方法
        参数:
            global_crops_scale: 全局裁剪的缩放范围
            local_crops_scale: 局部裁剪的缩放范围
            local_crops_number: 局部裁剪的数量
            global_crops_size: 全局裁剪的尺寸大小，默认为224
            local_crops_size: 局部裁剪的尺寸大小，默认为96
        """
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        # 记录数据增强参数的日志信息
        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip - 几何增强方法
        # 为全局裁剪创建几何增强转换
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # 为局部裁剪创建几何增强转换
        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring - 颜色失真/模糊处理
        # 创建颜色抖动转换
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        # 全局转换1的额外增强（高斯模糊）
        global_transfo1_extra = GaussianBlur(p=1.0)

        # 全局转换2的额外增强（高斯模糊和随机过曝）
        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        # 局部转换的额外增强（高斯模糊）
        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization - 标准化处理
        # 创建标准化转换
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        # 组合各种转换，形成完整的全局转换1
        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        # 组合各种转换，形成完整的全局转换2
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        # 组合各种转换，形成完整的局部转换
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        """
        对输入图像进行数据增强处理，生成全局和局部裁剪图像
        参数:
            image: 输入图像，将被进行各种数据增强处理
        返回:
            output: 包含各种增强处理后图像的字典，包括全局裁剪、教师模型使用的全局裁剪和局部裁剪
        """
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
