from typing import Any, Tuple

from torchvision.datasets import VisionDataset

from .decoders import TargetDecoder, ImageDataDecoder


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:

        """
        初始化ExtendedVisionDataset类
        继承自VisionDataset类
        :param args: 可变位置参数
        :param kwargs: 可变关键字参数
        """
        super().__init__(*args, **kwargs)  # type: ignore  # 调用父类的初始化方法

    def get_image_data(self, index: int) -> bytes:

        """
        获取指定索引的图像数据
        :param index: 图像数据的索引
        :return: 图像数据的字节表示
        """
        raise NotImplementedError  # 子类必须实现此方法

    def get_target(self, index: int) -> Any:

        """
        获取指定索引的目标数据
        :param index: 目标数据的索引
        :return: 目标数据
        """
        raise NotImplementedError  # 子类必须实现此方法

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        获取指定索引的数据样本
        :param index: 数据样本的索引
        :return: 包含图像和目标的元组
        """
        try:
            # 获取并解码图像数据
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            # 如果读取图像失败，抛出运行时错误
            raise RuntimeError(f"can not read image for sample {index}") from e
        # 获取并解码目标数据
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        # 如果定义了数据增强变换，则应用变换
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target  # 返回处理后的图像和目标

    def __len__(self) -> int:

        """
        返回数据集的大小
        :return: 数据集中的样本数量
        """
        raise NotImplementedError  # 子类必须实现此方法
