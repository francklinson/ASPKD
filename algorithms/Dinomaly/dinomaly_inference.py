import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
import pandas as pd
import yaml

from .dataset import get_data_transforms, PredictDataset
from .dinov3.hub.backbones import load_dinov3_model
from .models import vit_encoder
from .models.uad import ViTill, ViTillDinoV2
from .models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from .utils import get_gaussian_kernel, cal_anomaly_maps, visualize_when_predict


class ModelConfig:
    """
    模型配置类，从YAML配置文件加载不同模型的配置参数
    """
    _config = None
    
    @classmethod
    def _load_config(cls):
        """加载YAML配置文件"""
        if cls._config is None:
            # 获取项目根目录 (从 algorithms/Dinomaly/dinomaly_inference.py 向上3层)
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            config_path = os.path.join(project_root, 'config', 'asd_gui_config.yaml')
            
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f)
        return cls._config
    
    @classmethod
    def get_dinov2_config(cls, model_size: str) -> Dict:
        """获取DINOv2配置"""
        config = cls._load_config()
        arch_config = config['model_architectures']['dinov2'][model_size].copy()
        return arch_config
    
    @classmethod
    def get_dinov3_config(cls, model_size: str) -> Dict:
        """获取DINOv3配置"""
        config = cls._load_config()
        print(f"[DEBUG] get_dinov3_config: model_size={model_size}")
        print(f"[DEBUG] config keys: {list(config.keys())}")
        print(f"[DEBUG] model_architectures keys: {list(config.get('model_architectures', {}).keys())}")
        
        dinov3_config = config.get('model_architectures', {}).get('dinov3', {})
        arch_config = dinov3_config.get(model_size, {}).copy()
        
        print(f"[DEBUG] dinov3_config keys: {list(dinov3_config.keys())}")
        print(f"[DEBUG] arch_config before: {arch_config}")
        
        if not arch_config:
            raise ValueError(f"未找到DINOv3 {model_size}配置，请检查 config/asd_gui_config.yaml")
        
        # 拼接完整的权重路径
        weights_dir = dinov3_config.get('weights_dir')
        encoder_weight = arch_config.get('encoder_weight')
        
        print(f"[DEBUG] weights_dir={weights_dir}")
        print(f"[DEBUG] encoder_weight={encoder_weight}")
        
        if not weights_dir:
            raise ValueError(f"DINOv3 weights_dir 未配置")
        if not encoder_weight:
            raise ValueError(f"DINOv3 {model_size} encoder_weight 未配置")
        
        arch_config['encoder_weight'] = os.path.join(weights_dir, encoder_weight)
        print(f"[DEBUG] arch_config after: {arch_config}")
        return arch_config


class DinomalyBaseInferencer(ABC):
    """异常检测器基类"""

    def __init__(self, model_path: str, model_size: str, device: str = 'cuda:0', threshold: float = 0.5):
        print(f"[DEBUG] DinomalyBaseInferencer.__init__: model_path={model_path}, type={type(model_path)}")
        if not model_path:
            raise ValueError(f"DinomalyBaseInferencer 接收到无效的 model_path: {model_path}")
        
        # 设备选择逻辑：检查指定设备是否可用
        if torch.cuda.is_available():
            # 如果指定了cuda设备，检查该设备是否存在
            if device.startswith('cuda:'):
                gpu_id = int(device.split(':')[1])
                if gpu_id >= torch.cuda.device_count():
                    # 回退到 cuda:0
                    print(f"[WARNING] 请求的 {device} 不可用，回退到 cuda:0")
                    device = 'cuda:0'
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')
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
            image_path,
            image_size: int = 512,
            crop_size: int = 448,
            max_ratio: float = 0.01,
            resize_mask: int = 256,
            model_name=None,
            save_to_xlsx=False,
    ):
        """
        预测图像的异常分数
        返回 {文件名:(异常分数，是否异常) ,}字典 、 异常热力图路径列表
        """
        data_transform, _ = get_data_transforms(image_size, crop_size)
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(self.device)

        pred_data = PredictDataset(input_img_pth=image_path, transform=data_transform)
        pred_dataloader = torch.utils.data.DataLoader(
            pred_data,
            batch_size=1,
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
                        :, :int(anomaly_map.shape[1] * max_ratio)].mean(dim=1)
                img_path_list.extend(list(img_path))
                sp_score_list.extend(sp_score.tolist())

            visualize_save_dir = f"{self.__class__.__name__.lower()}_{self.model_size}_predict"
            save_img_path_list = visualize_when_predict(
                self.model,
                dataloader=pred_dataloader,
                device=self.device,
                _class_="predict",
                save_name=visualize_save_dir,
                overlay_on_image=True  # 热力图叠加原图显示
            )
            print(f"Visualization done!! Saved to ./visualize/{visualize_save_dir}")
        pred_res_dict = dict()
        for i in range(len(sp_score_list)):
            score = sp_score_list[i]
            pred_res_dict[img_path_list[i].split(os.sep)[-1]] = (score, score > self.threshold)

        pred_res_col_dict = dict()
        pred_res_col_dict["filepath"] = [img_path.split(os.sep)[-1] for img_path in img_path_list]
        pred_res_col_dict["score"] = sp_score_list
        pred_res_col_dict["pred"] = [score > self.threshold for score in sp_score_list]

        self.format_output_pred_result(pred_res_dict)
        if save_to_xlsx:
            df = pd.DataFrame(pred_res_col_dict)
            df.to_excel(f"dinomaly_{model_name}_results.xlsx")
        return pred_res_dict, save_img_path_list

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

    def __init__(self, model_path: str, model_size: str, device: str = 'cuda:0', threshold: float = 0.02):
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
        config = ModelConfig.get_dinov2_config(self.model_size)

        encoder = vit_encoder.load(config['encoder_name'])

        return self._create_model(encoder, config, model_path)


class DinomalyDinoV3Inference(DinomalyBaseInferencer):
    """DINOv3异常检测器实现"""

    def __init__(self, model_path: str, model_size: str, device: str = 'cuda:0', threshold: float = 0.033):
        super().__init__(model_path, model_size, device, threshold)

    def _create_model(self, encoder: nn.Module, config: Dict, model_path) -> nn.Module:
        """创建模型实例"""
        print(f"[DEBUG] _create_model: model_path={model_path}, type={type(model_path)}")
        if not model_path:
            raise ValueError(f"_create_model 接收到无效的 model_path: {model_path}")
            
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
        config = ModelConfig.get_dinov3_config(self.model_size)

        encoder = load_dinov3_model(
            config['encoder_name'],
            layers_to_extract_from=config['target_layers'],
            pretrained_weight_path=config['encoder_weight']
        )

        return self._create_model(encoder, config, model_path)


if __name__ == '__main__':
    # 初始化检测器
    # detectorv2 = DinomalyDinoV2Inference(
    #     model_path="/mnt/test/scripts/asd_for_spk/Dinomaly/saved_results/dinomaly_dinov2_small_dk_22050_qzgy_22050_epoch_10000_Mon Jan  5 19:59:12 2026.pth",
    #     model_size='small')

    detectorv3 = DinomalyDinoV3Inference(
        model_path="/mnt/test/scripts/asd_for_spk/Dinomaly/saved_results/dinomaly_dinov3_small_dk_qzgy_epoch_10000_Fri Jan 23 17:02:55 2026.pth",
        model_size='small')

    # 单张图像检测
    # image = r'minitest/003.png'
    # detectorv2.predict(image)
    # detectorv3.predict(image)

    # 检测文件夹
    image = "/mnt/test/scripts/asd_for_spk/data/spk_260123_with_manual/qzgy/test/good/"
    # detectorv2.predict(image, model_name="dinov2")
    detectorv3.predict(image, model_name="dinov3")
    # 批量检测
    # image_list = [Image.open(f) for f in ['test1.jpg', 'test2.jpg']]
    # detectorv3.predict(image, model_name="dinov3")
