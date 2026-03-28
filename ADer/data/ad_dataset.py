import os
import glob
import json
import random
import pickle
from torchvision import transforms
from ADer.util.data import get_img_loader
import torch.utils.data as data
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
import torch
import cv2
import math
import copy
import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
# from . import DATA
from ADer.data import DATA
from ADer.data.noise import Simplex_CLASS

# data
# ├── mvtec
#     ├── meta.json
#     ├── bottle
#         ├── train
#             └── good
#                 ├── 000.png
#         ├── test
#             ├── good
#                 ├── 000.png
#             ├── anomaly1
#                 ├── 000.png
#         └── ground_truth
#             ├── anomaly1
#                 ├── 000.png

@DATA.register_module
class DefaultAD(data.Dataset):
    def __init__(self, cfg, train=True, transform=None, target_transform=None):
        self.root = cfg.data.root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.loader = get_img_loader(cfg.data.loader_type)
        self.loader_target = get_img_loader(cfg.data.loader_type_target)

        self.data_all = []
        name = self.root.split('/')[-1]
        if name in ['mvtec', 'coco', 'visa', 'medical', 'btad', 'mpdd', 'mad_sim', 'mad_real', 'spk', ]:
            meta_info = json.load(open(f'{self.root}/{cfg.data.meta}', 'r'))
            meta_info = meta_info['train' if self.train else 'test']
            self.cls_names = cfg.data.cls_names
            if not isinstance(self.cls_names, list):
                self.cls_names = [self.cls_names]
            self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names
        elif name in ['mvtec3d', 'mvtec_loco']:
            meta_info = json.load(open(f'{self.root}/{cfg.data.meta}', 'r'))
            if self.train:
                meta_info, meta_info_val = meta_info['train'], meta_info['validation']
                for k in meta_info.keys():
                    meta_info[k].extend(meta_info_val[k])
            else:
                meta_info = meta_info['test']
            self.cls_names = cfg.data.cls_names
            if not isinstance(self.cls_names, list):
                self.cls_names = [self.cls_names]
            self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names
        elif name in ['realiad']:
            self.cls_names = cfg.data.cls_names
            if not isinstance(self.cls_names, list):
                self.cls_names = [self.cls_names]
            if len(self.cls_names) == 0:
                cls_names = os.listdir(self.root)
                real_cls_names = []
                for cls_name in cls_names:
                    if cls_name.split('.')[0] not in real_cls_names:
                        real_cls_names.append(cls_name.split('.')[0])
                real_cls_names.sort()
                self.cls_names = real_cls_names
            meta_info = dict()
            for cls_name in self.cls_names:
                data_cls_all = []
                cls_info = json.load(open(f'{self.root}/{cls_name}.json', 'r'))
                data_cls = cls_info['train' if self.train else 'test']
                for data in data_cls:
                    if data['anomaly_class'] == 'OK':
                        info_img = dict(
                            img_path=f"{cls_name}/{data['image_path']}",
                            mask_path='',
                            cls_name=cls_name,
                            specie_name='',
                            anomaly=0,
                        )
                    else:
                        info_img = dict(
                            img_path=f"{cls_name}/{data['image_path']}",
                            mask_path=f"{cls_name}/{data['mask_path']}",
                            cls_name=cls_name,
                            specie_name=data['anomaly_class'],
                            anomaly=1,
                        )
                    data_cls_all.append(info_img)
                meta_info[cls_name] = data_cls_all

        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        random.shuffle(self.data_all) if self.train else None
        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
        data['specie_name'], data['anomaly']
        img_path = f'{self.root}/{img_path}'
        img = self.loader(img_path)
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            img_mask = np.array(self.loader_target(f'{self.root}/{mask_path}')) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': img_path}


class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0, 1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image


def get_data_transforms(size, isize):
    data_transforms = transforms.Compose([Normalize(), ToTensor()])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms



if __name__ == '__main__':
    from argparse import Namespace as _Namespace

    cfg = _Namespace()
    data = _Namespace()
    data.sampler = 'naive'
    # ========== MVTec ==========
    # data.root = 'data/mvtec'
    # data.meta = 'meta.json'
    # # data.cls_names = ['bottle']
    # data.cls_names = []
    # data.loader_type = 'pil'
    # data.loader_type_target = 'pil_L'
    # data_fun = DefaultAD

    # data.root = 'data/mvtec3d'
    # data.meta = 'meta.json'
    # # data.cls_names = ['bagel']
    # data.cls_names = []
    # data.loader_type = 'pil'
    # data.loader_type_target = 'pil_L'
    # data_fun = DefaultAD

    # data.root = 'data/coco'
    # data.meta = 'meta_20_0.json'
    # data.cls_names = ['coco']
    # data.loader_type = 'pil'
    # data.loader_type_target = 'pil_L'
    # data_fun = DefaultAD

    # data.root = 'data/visa'
    # data.meta = 'meta.json'
    # # data.cls_names = ['candle']
    # data.cls_names = []
    # data.loader_type = 'pil'
    # data.loader_type_target = 'pil_L'
    # data_fun = DefaultAD

    # ========== Cifar ==========
    # data.type = 'DefaultAD'
    # data.root = 'data/cifar'
    # data.type_cifar = 'cifar10'
    # data.cls_names = ['cifar']
    # data.uni_setting = True
    # data.one_cls_train = True
    # data.split_idx = 0
    # data_fun = CifarAD

    # ========== Tiny ImageNet ==========
    # data.root = 'data/tiny-imagenet-200'
    # data.cls_names = ['tin']
    # data.loader_type = 'pil'
    # data.split_idx = 0
    # data_fun = TinyINAD

    # ========== Real-IAD ==========
    data.root = 'data/realiad/explicit_full'
    # data.cls_names = ['audiojack']
    data.cls_names = []
    data.loader_type = 'pil'
    data.loader_type_target = 'pil_L'
    data.views = ['C1', 'C2']
    # data.views = []
    data.use_sample = True
    data_fun = RealIAD

    cfg.data = data
    data_debug = data_fun(cfg, train=True)
