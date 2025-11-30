import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json

# import imgaug.augmenters as iaa

# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop((isize, isize), scale=(0.6, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    return data_transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        """
        初始化MVTec数据集类
        Args:
            root: 数据集根目录路径
            transform: 图像数据的预处理转换
            gt_transform: 真实标签数据的预处理转换
            phase: 数据集阶段，'train'或'test'
        """
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')  # 训练集图像路径
        else:
            self.img_path = os.path.join(root, 'test')  # 测试集图像路径
            self.gt_path = os.path.join(root, 'ground_truth')  # 测试集真实标签路径
        self.transform = transform  # 图像预处理转换
        self.gt_transform = gt_transform  # 真实标签预处理转换
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, input_img_pth, transform):
        """
        用于预测推理的数据集，没有gt
        Args:
            transform: 图像数据的预处理转换
            img_pth: 输入图像。可能是文件地址、文件列表、文件目录

        """
        self.transform = transform  # 图像预处理转换
        # load dataset
        self.img_paths = None
        self.load_dataset(input_img_pth)  # self.labels => good : 0, anomaly : 1

    def load_dataset(self, input_img_pth):
        """
        加载数据集的函数，根据输入的参数类型加载不同的图片路径

        参数:
            self: 类实例
            input_img_pth: 输入的图片路径，可以是文件路径、目录路径或路径列表

        返回:
            无返回值，但会将加载的图片路径存储在self.img_paths中

        异常:
            ValueError: 当输入既不是文件路径、目录路径也不是列表时抛出
        """
        img_to_pre_paths = []  # 用于存储待处理的图片路径列表

        # 判断输入是单个文件
        if os.path.isfile(input_img_pth):
            img_to_pre_paths = [input_img_pth]  # 如果是单个文件，将其放入列表中
        # 判断输入是目录
        elif os.path.isdir(input_img_pth):
            # 获取目录中所有.png、.JPG和.bmp格式的图片路径
            img_to_pre_paths = glob.glob(os.path.join(input_img_pth, "*.png")) + \
                               glob.glob(os.path.join(input_img_pth, "*.JPG")) + \
                               glob.glob(os.path.join(input_img_pth, "*.bmp"))
        # 判断输入是列表
        elif isinstance(input_img_pth, list):
            img_to_pre_paths = input_img_pth  # 如果是列表，直接使用
        else:
            raise ValueError("input_img_pth should be a file path or a directory path!")  # 输入类型不符合要求时抛出异常

        self.img_paths = np.array(img_to_pre_paths)  # 将图片路径列表转换为numpy数组并存储
        return

    def __len__(self):

        """返回数据集中图像路径列表的长度
        这是一个特殊方法，使得我们可以使用len()函数获取数据集的大小

        Returns:
            int: 图像路径列表的长度，即数据集中样本的数量
        """
        return len(self.img_paths)  # 返回存储图像路径列表的长度

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的图像及其路径
        参数:
            idx (int): 数据集中样本的索引值
        返回:
            tuple: 包含两个元素的元组
                - img (torch.Tensor): 经过转换处理后的图像张量
                - img_path (str): 图像文件的路径
        """
        # 根据索引获取对应的图像文件路径
        img_path = self.img_paths[idx]
        # 打开图像文件并转换为RGB模式
        img = Image.open(img_path).convert('RGB')
        # 对图像进行预定义的转换处理
        img = self.transform(img)
        # 返回处理后的图像及其原始路径
        return img, img_path


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class LOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*/000.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        size = (img.size[1], img.size[0])
        img = self.transform(img)
        type = self.types[idx]
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, type, size


class InsPLADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        self.transform = transform
        self.phase = phase
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == 'train':
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img, label, img_path


class AeBADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        defect_types = [i for i in defect_types if i[0] != '.']
        for defect_type in defect_types:
            if defect_type == 'good':
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
            else:
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, domain_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.phase == 'train':
            return img, label
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):

        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
            img_tot_paths.extend(img_paths)
            tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path, label = self.img_paths[idx], self.labels[idx]
            img = Image.open(img_path).convert('RGB')
        except:
            img_path, label = self.img_paths[idx - 1], self.labels[idx - 1]
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label


class MVTecDRAEMDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, strong_transform, phase, anomaly_source_path, anomaly_ratio=0.5,
                 size=256):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.strong_transform = strong_transform
        self.anomaly_ratio = anomaly_ratio
        self.size = size
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        no_anomaly = random.random()
        if no_anomaly > self.anomaly_ratio:
            return image, 0
        else:
            aug = self.randAugmenter()

            perlin_scale = 6
            min_perlin_scale = 0
            anomaly_source_img = Image.open(anomaly_source_path).convert('RGB').resize((self.size, self.size))
            anomaly_source_img = np.asarray(anomaly_source_img)
            anomaly_img_augmented = aug(image=anomaly_source_img)

            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

            perlin_noise = rand_perlin_2d_np((self.size, self.size),
                                             (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            threshold = 0.5
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)

            img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr

            beta = random.random() * 0.7 + 0.1

            image = image.resize((self.size, self.size))
            image = np.asarray(image)
            augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
            # augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image

            return Image.fromarray(np.uint8(augmented_image)), 1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')

        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        a_img, label = self.augment_image(img, self.anomaly_source_paths[anomaly_source_idx])

        img = self.transform(img)
        a_img = self.strong_transform(a_img)

        assert img.size()[1:] == a_img.size()[1:], "image.size != a_img.size !!!"

        return img, a_img, label


class MVTecSimplexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform

        self.simplexNoise = Simplex_CLASS()
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img_normal = self.transform(img)

        if random.random() > 0.5:
            return img_normal, img_normal
        ## simplex_noise
        size = 256
        img = img.resize((size, size))
        img = np.asarray(img)
        h_noise = np.random.randint(10, int(size // 8))
        w_noise = np.random.randint(10, int(size // 8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256, 256, 3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise + w_noise,
        :] = 0.2 * simplex_noise.transpose(1, 2, 0)
        img_noise = img + init_zero * 255
        img_noise = Image.fromarray(np.uint8(img_noise))
        img_noise = self.transform(img_noise)

        return img_normal, img_noise
