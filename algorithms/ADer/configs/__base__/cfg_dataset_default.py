from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F


class cfg_dataset_default(Namespace):

    def __init__(self):
        Namespace.__init__(self)
        self.data = Namespace()
        self.data.sampler = 'naive'
        self.data.loader_type = 'pil'
        self.data.loader_type_target = 'pil_L'

        # ---------- MUAD ----------
        self.data.type = 'DefaultAD'
        self.data.root = 'data/mvtec'  # ['mvtec', 'visa', 'mvtec3d', 'medical']
        self.data.meta = 'meta.json'
        self.data.cls_names = []

        mvtec = [
            'carpet', 'grid', 'leather', 'tile', 'wood',
            'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
            'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
        ]
        visa = [
            'pcb1', 'pcb2', 'pcb3', 'pcb4',
            'macaroni1', 'macaroni2', 'capsules', 'candle',
            'cashew', 'chewinggum', 'fryum', 'pipe_fryum',
        ]
        mvtec3d = [
            'bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
            'foam', 'peach', 'potato', 'rope', 'tire',
        ]
        medical = [
            'brain', 'liver', 'retinal',
        ]

        realiad = [
            'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser',
            'fire_hood', 'mint', 'mounts', 'pcb', 'phone_battery',
            'plastic_nut', 'plastic_plug', 'porcelain_doll', 'regulator', 'rolled_strip_base',
            'sim_card_set', 'switch', 'tape', 'terminalblock', 'toothbrush',
            'toy', 'toy_brick', 'transistor1', 'u_block', 'usb',
            'usb_adaptor', 'vcpill', 'wooden_beads', 'woodstick', 'zipper',
        ]

        self.data.train_transforms = [
            dict(type='Resize', size=(256, 256), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(256, 256)),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        ]
        self.data.test_transforms = self.data.train_transforms
        self.data.target_transforms = [
            dict(type='Resize', size=(256, 256), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(256, 256)),
            dict(type='ToTensor'),
        ]