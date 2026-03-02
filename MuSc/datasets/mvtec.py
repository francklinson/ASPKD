import os
from enum import Enum
import PIL
import torch
from torchvision import transforms
import random

_CLASSNAMES = ["bottle", "cable", "capsule", "carpet", "grid",
               "hazelnut", "leather", "metal_nut", "pill", "screw",
               "tile", "toothbrush", "transistor", "wood", "zipper"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            source,  # 数据来源
            classname,  # 类别名称
            resize=256,  # 调整大小的目标尺寸
            imagesize=224,  # 裁剪后的图像尺寸
            split=DatasetSplit.TRAIN,  # 数据集划分（训练/验证/测试）
            clip_transformer=None,  # CLIP模型的图像转换器
            k_shot=0,  # 小样本学习的样本数量
            random_seed=42,  # 随机种子，确保可重复性
            divide_num=1,  # 数据集划分的子集数量
            divide_iter=0,  # 当前划分的子集索引
            **kwargs,  # 其他可选参数
    ):
        super().__init__()  # 调用父类的初始化方法

        # 基本属性初始化
        self.source = source  # 设置数据来源
        self.split = split  # 设置数据集划分
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES  # 设置要使用的类别

        # 数据处理
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()  # 获取图像数据
        print(self.imgpaths_per_class)
        print(self.data_to_iterate)

        # 如果需要将数据集划分为多个子集
        if divide_num > 1:
            # divide into subsets
            self.data_to_iterate = self.sub_datasets(self.data_to_iterate, divide_num, divide_iter, random_seed)

        # 如果使用小样本学习
        if k_shot > 0:
            # few-shot
            torch.manual_seed(random_seed)  # 设置随机种子
            if k_shot >= len(self.data_to_iterate):
                pass  # 如果请求的样本数大于等于总数据量，则不做处理
            else:
                # 随机选择指定数量的样本
                indices = torch.randint(0, len(self.data_to_iterate), (k_shot,))
                self.data_to_iterate = [self.data_to_iterate[i] for i in indices]
        # 图像转换设置
        if clip_transformer is None:
            # 使用默认的图像预处理流程
            self.transform_img = [
                transforms.Resize((resize, resize)),  # 调整图像大小
                transforms.CenterCrop(imagesize),  # 中心裁剪
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # 标准化
            ]
            self.transform_img = transforms.Compose(self.transform_img)  # 组合转换操作
        else:
            self.transform_img = clip_transformer  # 使用提供的CLIP转换器

        # 掩码转换设置
        self.transform_mask = [
            transforms.Resize((resize, resize)),  # 调整掩码大小
            transforms.CenterCrop(imagesize),  # 中心裁剪
            transforms.ToTensor(),  # 转换为张量
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)  # 组合转换操作

        # 设置图像尺寸信息
        self.imagesize = (3, imagesize, imagesize)  # (通道数, 高度, 宽度)

    def sub_datasets(self, full_datasets, divide_num, divide_iter, random_seed=42):
        # uniform division
        if divide_num == 0:
            return full_datasets
        random.seed(random_seed)

        id_dict = {}
        for i in range(len(full_datasets)):
            anomaly_type = full_datasets[i][2].split('/')[-2]
            if anomaly_type not in id_dict.keys():
                id_dict[anomaly_type] = []
            id_dict[anomaly_type].append(i)

        sub_id_list = []
        for k in id_dict.keys():
            type_id_list = id_dict[k]
            random.shuffle(type_id_list)
            devide_list = [type_id_list[i:i + divide_num] for i in range(0, len(type_id_list), divide_num)]
            sub_list = [devide_list[i][divide_iter] for i in range(len(devide_list)) if
                        len(devide_list[i]) > divide_iter]
            sub_id_list.extend(sub_list)

        return [full_datasets[id] for id in sub_id_list]

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        """
        获取图像数据及其对应的掩码路径信息，按类别和异常类型组织
        返回:
            tuple: 包含两个元素的元组
                - imgpaths_per_class: 字典，按类别和异常类型组织的图像路径
                - data_to_iterate: 列表，包含可迭代的数据元组，每个元组包含类别、异常类型、图像路径和掩码路径
        """
        # 初始化按类别存储图像路径和掩码路径的字典
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        # 遍历每个要使用的类别
        for classname in self.classnames_to_use:
            # 构建类别数据路径和掩码路径
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            # 获取该类别下的所有异常类型
            anomaly_types = os.listdir(classpath)

            # 初始化当前类别的图像路径和掩码路径字典
            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            # 遍历每个异常类型
            for anomaly in anomaly_types:
                # 构建异常类型路径并获取该路径下的所有文件
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                # 存储当前异常类型的所有图像路径
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                # 如果是测试集且异常类型不是"good"，则获取对应的掩码文件
                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    # 对于非测试集或"good"类型，掩码设为None
                    maskpaths_per_class[classname]["good"] = None

        # 准备最终要迭代的数据结构
        data_to_iterate = []
        # 按类别名称排序遍历
        for classname in sorted(imgpaths_per_class.keys()):
            # 按异常类型排序遍历
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                # 遍历当前类别和异常类型下的所有图像
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    # 创建数据元组，包含类别、异常类型、图像路径和掩码路径
                    data_tuple = [classname, anomaly, image_path]
                    # 如果是测试集且异常类型不是"good"，则添加对应的掩码路径
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        # 否则添加None作为掩码路径
                        data_tuple.append(None)
                    # 将数据元组添加到迭代列表中
                    data_to_iterate.append(data_tuple)

        # 返回按类别组织的图像路径和可迭代的数据列表
        return imgpaths_per_class, data_to_iterate
