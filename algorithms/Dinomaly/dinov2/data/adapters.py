from typing import Any, Tuple

from torch.utils.data import Dataset


class DatasetWithEnumeratedTargets(Dataset):
    def __init__(self, dataset):
        """
        初始化函数，接收一个数据集对象
        :param dataset: 原始数据集对象
        """
        self._dataset = dataset  # 保存传入的数据集对象

    def get_image_data(self, index: int) -> bytes:
        """
        获取指定索引的图像数据
        :param index: 图像的索引
        :return: 图像数据的字节表示
        """
        return self._dataset.get_image_data(index)  # 调用原始数据集的方法获取图像数据

    def get_target(self, index: int) -> Tuple[Any, int]:
        """
        获取指定索引的目标数据，并添加索引信息
        :param index: 目标数据的索引
        :return: 包含索引和原始目标值的元组
        """
        target = self._dataset.get_target(index)  # 获取原始目标数据
        return (index, target)  # 返回包含索引和目标值的元组

    def __getitem__(self, index: int) -> Tuple[Any, Tuple[Any, int]]:
        """
        获取指定索引的数据项，包括图像和目标
        :param index: 数据项的索引
        :return: 包含图像和增强目标(包含索引)的元组
        """
        image, target = self._dataset[index]  # 从原始数据集获取图像和目标
        target = index if target is None else target  # 如果目标为None，使用索引作为目标
        return image, (index, target)  # 返回图像和包含索引的目标元组

    def __len__(self) -> int:
        """
        获取数据集的长度
        :return: 数据集中的样本数量
        """
        return len(self._dataset)  # 返回原始数据集的长度
