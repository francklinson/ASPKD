from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F

from dataset import get_data_transforms, PredictDataset
from dinov3.hub.backbones import load_dinov3_model
from models import vit_encoder
from models.uad import ViTill, ViTillDinoV2
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import get_gaussian_kernel, cal_anomaly_maps, visualize_when_predict


class ModelConfig:
    """模型配置类，统一管理不同模型的配置参数"""

    DINOV3_CONFIGS = {
        'small': {
            'target_layers': [2, 3, 4, 5, 6, 7, 8, 9],
            'encoder_name': 'dinov3_vits16',
            'encoder_weight': 'weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
            'embed_dim': 384,
            'num_heads': 6
        },
        'base': {
            'target_layers': [2, 3, 4, 5, 6, 7, 8, 9],
            'encoder_name': 'dinov3_vitb16',
            'encoder_weight': 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
            'embed_dim': 768,
            'num_heads': 12
        },
        'large': {
            'target_layers': [4, 6, 8, 10, 12, 14, 16, 18],
            'encoder_name': 'dinov3_vitl16',
            'encoder_weight': 'weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
            'embed_dim': 1024,
            'num_heads': 16
        }
    }

    DINOV2_CONFIGS = {
        'small': {
            'encoder_name': "dinov2reg_vit_small_14",
            'target_layers': [2, 3, 4, 5, 6, 7, 8, 9],
            'embed_dim': 384,
            'num_heads': 6
        },
        'base': {
            'encoder_name': "dinov2reg_vit_base_14",
            'target_layers': [2, 3, 4, 5, 6, 7, 8, 9],
            'embed_dim': 768,
            'num_heads': 12
        },
        'large': {
            'encoder_name': "dinov2reg_vit_large_14",
            'target_layers': [4, 6, 8, 10, 12, 14, 16, 18],
            'embed_dim': 1024,
            'num_heads': 16
        }
    }


class DinomalyBaseInferencer(ABC):
    """异常检测器基类"""

    def __init__(self, model_path: str, model_size: str, device: str = 'cuda:0', threshold: float = 0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_size = model_size
        self.model = self._load_model(model_path)
        self.model.eval()
        self.batch_size = 8
        self.transform = get_data_transforms(512, 448)
        self.vis_dir = "visualize"
        self.threshold = threshold

    @abstractmethod
    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型的抽象方法，需要子类实现"""
        pass

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """预处理输入图像"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(
            self,
            image_path: str,
            image_size: int = 512,
            crop_size: int = 448,
            max_ratio: float = 0.01,
            resize_mask: int = 256
    ):
        """预测图像的异常分数"""
        data_transform, _ = get_data_transforms(image_size, crop_size)
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(self.device)

        pred_data = PredictDataset(input_img_pth=image_path, transform=data_transform)
        pred_dataloader = torch.utils.data.DataLoader(
            pred_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

        sp_score_list = []
        img_path_list = []
        with torch.no_grad():
            for img, img_path in pred_dataloader:
                img = img.to(self.device)
                output = self.model(img)
                en, de = output[0], output[1]

                anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])

                if resize_mask is not None:
                    anomaly_map = F.interpolate(
                        anomaly_map,
                        size=resize_mask,
                        mode='bilinear',
                        align_corners=False
                    )

                anomaly_map = gaussian_kernel(anomaly_map)

                if max_ratio == 0:
                    sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
                else:
                    anomaly_map = anomaly_map.flatten(1)
                    sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][
                        :, :int(anomaly_map.shape[1] * max_ratio)
                    ].mean(dim=1)
                img_path_list.extend(list(img_path))
                sp_score_list.extend(sp_score.tolist())

            visualize_save_dir = f"{self.__class__.__name__.lower()}_{self.model_size}_predict"
            visualize_when_predict(
                self.model,
                dataloader=pred_dataloader,
                device=self.device,
                _class_="predict",
                save_name=visualize_save_dir
            )
            print(f"Visualization done!! Saved to ./visualize/{visualize_save_dir}")
        pred_res_dict = dict()
        for i in range(len(sp_score_list)):
            score = sp_score_list[i]
            pred_res_dict[img_path_list[i]] = (score, score > self.threshold)
        self.format_output_pred_result(pred_res_dict)
        return pred_res_dict

    def format_output_pred_result(self, pred_res_dict):
        """
        格式化输出预测结果
        Parameters
        ----------
        pred_res_dict : dict
        """
        print("Predict Result: ")
        print('*' * 40)
        print("File  :  Score  Label")
        for k, v in pred_res_dict.items():
            print(f"{k} : {v[0]:.4f} {v[1]}")


class DinomalyDinoV2Inference(DinomalyBaseInferencer):
    """DINOv2异常检测器实现"""

    def __init__(self, model_path: str, model_size: str, device: str = 'cuda:0', threshold: float = 0.15):
        super().__init__(model_path, model_size, device, threshold)

    def _create_model(self, encoder: nn.Module, config: Dict, model_path) -> nn.Module:
        """创建模型实例"""
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        bottleneck = nn.ModuleList([
            bMlp(config['embed_dim'], config['embed_dim'] * 4, config['embed_dim'], drop=0.2)
        ])

        decoder = nn.ModuleList([
            VitBlock(
                dim=config['embed_dim'],
                num_heads=config['num_heads'],
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2
            ) for _ in range(8)
        ])

        model = ViTillDinoV2(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            target_layers=config['target_layers'],
            mask_neighbor_size=0,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder
        )

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        print(f"Load model from {model_path} successfully!!")
        return model

    def _load_model(self, model_path: str) -> nn.Module:
        """加载DINOv2模型"""
        config = ModelConfig.DINOV2_CONFIGS[self.model_size]

        encoder = vit_encoder.load(config['encoder_name'])

        return self._create_model(encoder, config, model_path)


class DinomalyDinoV3Inference(DinomalyBaseInferencer):
    """DINOv3异常检测器实现"""

    def __init__(self, model_path: str, model_size: str, device: str = 'cuda:0', threshold: float = 0.05):
        super().__init__(model_path, model_size, device, threshold)

    def _create_model(self, encoder: nn.Module, config: Dict, model_path) -> nn.Module:
        """创建模型实例"""
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        bottleneck = nn.ModuleList([
            bMlp(config['embed_dim'], config['embed_dim'] * 4, config['embed_dim'], drop=0.2)
        ])

        decoder = nn.ModuleList([
            VitBlock(
                dim=config['embed_dim'],
                num_heads=config['num_heads'],
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2
            ) for _ in range(8)
        ])

        model = ViTill(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            target_layers=config['target_layers'],
            mask_neighbor_size=0,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder
        )

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        print(f"Load model from {model_path} successfully!!")
        return model

    def _load_model(self, model_path: str) -> nn.Module:
        """加载DINOv3模型"""
        config = ModelConfig.DINOV3_CONFIGS[self.model_size]

        encoder = load_dinov3_model(
            config['encoder_name'],
            layers_to_extract_from=config['target_layers'],
            pretrained_weight_path=config['encoder_weight']
        )

        return self._create_model(encoder, config, model_path)


if __name__ == '__main__':
    # 初始化检测器
    detectorv2 = DinomalyDinoV2Inference(
        model_path='saved_results/dinomaly_dinov2_small_carpet_grid_epoch_500_Sat Dec 20 19:52:36 2025.pth',
        model_size='small')

    detectorv3 = DinomalyDinoV3Inference(
        model_path='saved_results/dinomaly_dinov3_small_carpet_grid_epoch_500_Sat Dec 20 19:55:45 2025.pth',
        model_size='small')

    # 单张图像检测
    # image = r'minitest/003.png'
    # detectorv2.predict(image)
    # detectorv3.predict(image)


    # 检测文件夹
    image = r'../data/mvtec/hazelnut/test/crack'
    detectorv2.predict(image)
    detectorv3.predict(image)

    # 批量检测
    # image_list = [Image.open(f) for f in ['test1.jpg', 'test2.jpg']]
    # anomaly_maps, anomaly_scores = detector.batch_detect(image_list)
