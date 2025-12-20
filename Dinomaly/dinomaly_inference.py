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


# class DinomalyDinoV3Inference:
#     def __init__(self, model_path, model_size='base', device='cuda:0'):
#         """
#         初始化异常检测器
#         Args:
#             model_path: 训练好的模型权重路径
#             model_size: 模型大小，'base' 或 'large'
#             device: 运行设备
#         """
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         self.model_size = model_size
#         self.model = self._load_model(model_path)
#         self.model.eval()
#         self.batch_size = 8
#
#         # 获取数据预处理方法
#         self.transform = get_data_transforms(448, 392)
#         self.vis_dir = "visualize"
#
#     def _load_model(self, model_path):
#         """
#         加载模型
#         Args:
#             model_path: 模型权重路径
#         Returns:
#             加载好的模型
#         """
#         encoder_name, encoder_weight, target_layers, fuse_layer_encoder, fuse_layer_decoder = None, None, None, None, None
#         if self.model_size == "base":
#             target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
#             fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
#             fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
#             encoder_name = 'dinov3_vitb16'
#             encoder_weight = 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
#             embed_dim, num_heads = 768, 12
#
#         elif self.model_size == "large":
#             target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
#             fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
#             fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
#             encoder_name = 'dinov3_vitl16'
#             encoder_weight = 'weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
#             embed_dim, num_heads = 1024, 16
#
#         elif self.model_size == "small":
#             # 设置模型参数
#             target_layers = [2, 3, 4, 5, 6, 7, 8, 9]  # 目标层列表
#             fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 编码器融合层
#             fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 解码器融合层
#             batch_size = 16  # 批次大小
#             # 设置编码器名称和权重路径
#             encoder_name = 'dinov3_vits16'
#             encoder_weight = 'weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
#
#         # 加载预训练编码器
#         encoder = load_dinov3_model(encoder_name, layers_to_extract_from=target_layers,
#                                     pretrained_weight_path=encoder_weight)
#
#         # 根据编码器名称设置模型参数
#         if 'vits' in encoder_name:
#             embed_dim, num_heads = 384, 6
#         elif 'vitb' in encoder_name:
#             embed_dim, num_heads = 768, 12
#         elif 'vitl' in encoder_name:
#             embed_dim, num_heads = 1024, 16
#         else:
#             raise "Architecture not in vits, vitb, vitl."
#
#         # 初始化瓶颈层和解码器
#         bottleneck = []
#         decoder = []
#
#         # 添加瓶颈层模块
#         bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
#         bottleneck = nn.ModuleList(bottleneck)
#
#         # 添加解码器模块
#         for i in range(8):
#             blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
#                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
#                            attn=LinearAttention2)
#             decoder.append(blk)
#         decoder = nn.ModuleList(decoder)
#
#         # 创建完整模型
#         model = ViTill(
#             encoder=encoder,
#             bottleneck=bottleneck,
#             decoder=decoder,
#             target_layers=target_layers,
#             mask_neighbor_size=0,
#             fuse_layer_encoder=fuse_layer_encoder,
#             fuse_layer_decoder=fuse_layer_decoder
#         )
#         # 加载权重
#         model.load_state_dict(torch.load(model_path, map_location=self.device))
#         model = model.to(self.device)
#         print(f"Load model from {model_path} successfully!!")
#         return model
#
#     def preprocess_image(self, image):
#         """
#         预处理输入图像
#         Args:
#             image: PIL Image 或 numpy array
#         Returns:
#             预处理后的tensor
#         """
#         if isinstance(image, np.ndarray):
#             image = Image.fromarray(image)
#         return self.transform(image).unsqueeze(0).to(self.device)
#
#     def predict_anomaly_score(self, image_path, image_size=512, crop_size=448, max_ratio=0.01, resize_mask=256):
#         """
#         预测图像的异常分数
#         参数:
#             image_path (str): 输入图像的路径
#             image_size (int): 输入图像的目标大小，默认为512
#             crop_size (int): 裁剪大小，默认为448
#             max_ratio (float): 用于计算异常分数的最大比例，默认为0.01
#             resize_mask (int): 调整异常图大小的目标尺寸，默认为256
#         返回:
#             torch.Tensor: 图像的异常分数
#         """
#         # 获取数据变换
#         data_transform, _ = get_data_transforms(image_size, crop_size)
#
#         # 获取高斯核
#         gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(self.device)  # 创建高斯核并移动到设备
#
#         pred_data = PredictDataset(input_img_pth=image_path, transform=data_transform)
#         pred_dataloader = torch.utils.data.DataLoader(pred_data, batch_size=self.batch_size, shuffle=False,
#                                                       num_workers=4)
#         # 预测
#         sp_score_list = []
#         with torch.no_grad():  # 关闭梯度计算以节省内存
#             for img, img_path in pred_dataloader:
#                 img = img.to(self.device)  # 确保图像在正确的设备上
#                 # starter.record()
#                 output = self.model(img)  # 通过模型获取输出
#                 # ender.record()
#                 # torch.cuda.synchronize()
#                 # curr_time = starter.elapsed_time(ender)
#                 en, de = output[0], output[1]  # 分离编码器和解码器的输出
#                 # 计算anomaly_maps
#                 anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])  # 计算异常图
#                 if resize_mask is not None:  # 如果需要调整异常图大小
#                     anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear',
#                                                 align_corners=False)  # 使用双线性插值调整大小
#                 anomaly_map = gaussian_kernel(anomaly_map)  # 应用高斯核平滑异常图
#                 if max_ratio == 0:  # 如果max_ratio为0，使用最大值作为异常分数
#                     sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
#                 else:  # 否则，使用前max_ratio比例的最大值的平均值作为异常分数
#                     anomaly_map = anomaly_map.flatten(1)
#                     sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
#                                :int(anomaly_map.shape[1] * max_ratio)]
#                     sp_score = sp_score.mean(dim=1)
#                 print(img_path)
#                 sp_score_list.extend(sp_score.tolist())
#             # 执行可视化
#             visualize_when_predict(self.model, dataloader=pred_dataloader, device=self.device, _class_="predict",
#                                    save_name=f"dinomaly_dinov3_{self.model_size}_predict")
#
#         return sp_score_list  # 返回计算得到的异常分数
#
# class DinomalyDinoV2Inference:
#     def __init__(self, model_path, model_size='base', device='cuda:0'):
#         """
#         初始化异常检测器
#         Args:
#             model_path: 训练好的模型权重路径
#             model_size: 模型大小，'base' 或 'large'
#             device: 运行设备
#         """
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         self.model_size = model_size
#         self.model = self._load_model(model_path)
#         self.model.eval()
#         self.batch_size = 8
#
#         # 获取数据预处理方法
#         self.transform = get_data_transforms(size=512, isize=448)
#         self.vis_dir = "visualize"
#
#     def _load_model(self, model_path):
#         """
#         加载模型
#         Args:
#             model_path: 模型权重路径
#         Returns:
#             加载好的模型
#         """
#         encoder_name, target_layers, fuse_layer_encoder, fuse_layer_decoder = None, None, None, None
#         if self.model_size == "small":
#             encoder_name = "dinov2reg_vit_small_14"
#             target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
#             embed_dim, num_heads = 384, 6
#
#         elif self.model_size == "base":
#             encoder_name = "dinov2reg_vit_base_14"
#             target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
#             embed_dim, num_heads = 768, 12
#
#         elif self.model_size == "large":
#             encoder_name = "dinov2reg_vit_large_14"
#             target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
#             embed_dim, num_heads = 1024, 16
#
#         fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
#         fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
#
#         # 加载预训练编码器
#         encoder = vit_encoder.load(encoder_name)
#
#         # 初始化瓶颈层和解码器
#         bottleneck = []
#         decoder = []
#
#         # 添加瓶颈层模块
#         bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
#         bottleneck = nn.ModuleList(bottleneck)
#
#         # 添加解码器模块
#         for i in range(8):
#             blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
#                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
#                            attn=LinearAttention2)
#             decoder.append(blk)
#         decoder = nn.ModuleList(decoder)
#
#         # 创建完整模型
#         model = ViTillDinoV2(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
#                              mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder,
#                              fuse_layer_decoder=fuse_layer_decoder)
#         # 加载权重
#         model.load_state_dict(torch.load(model_path, map_location=self.device))
#         model = model.to(self.device)
#         print(f"Load model from {model_path} successfully!!")
#         return model
#
#     def preprocess_image(self, image):
#         """
#         预处理输入图像
#         Args:
#             image: PIL Image 或 numpy array
#         Returns:
#             预处理后的tensor
#         """
#         if isinstance(image, np.ndarray):
#             image = Image.fromarray(image)
#         return self.transform(image).unsqueeze(0).to(self.device)
#
#     def predict_anomaly_score(self, image_path, image_size=512, crop_size=448, max_ratio=0.01, resize_mask=256):
#         """
#         预测图像的异常分数
#         参数:
#             image_path (str): 输入图像的路径
#             image_size (int): 输入图像的目标大小，默认为512
#             crop_size (int): 裁剪大小，默认为448
#             max_ratio (float): 用于计算异常分数的最大比例，默认为0.01
#             resize_mask (int): 调整异常图大小的目标尺寸，默认为256
#         返回:
#             torch.Tensor: 图像的异常分数
#         """
#         # 获取数据变换
#         data_transform, _ = get_data_transforms(image_size, crop_size)
#
#         # 获取高斯核
#         gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(self.device)  # 创建高斯核并移动到设备
#
#         pred_data = PredictDataset(input_img_pth=image_path, transform=data_transform)
#         pred_dataloader = torch.utils.data.DataLoader(pred_data, batch_size=self.batch_size, shuffle=False,
#                                                       num_workers=4)
#         # 预测
#         sp_score_list = []
#         with torch.no_grad():  # 关闭梯度计算以节省内存
#             for img, img_path in pred_dataloader:
#                 img = img.to(self.device)  # 确保图像在正确的设备上
#                 output = self.model(img)  # 通过模型获取输出
#                 en, de = output[0], output[1]  # 分离编码器和解码器的输出
#                 # 计算anomaly_maps
#                 anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])  # 计算异常图
#                 if resize_mask is not None:  # 如果需要调整异常图大小
#                     anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear',
#                                                 align_corners=False)  # 使用双线性插值调整大小
#                 anomaly_map = gaussian_kernel(anomaly_map)  # 应用高斯核平滑异常图
#                 if max_ratio == 0:  # 如果max_ratio为0，使用最大值作为异常分数
#                     sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
#                 else:  # 否则，使用前max_ratio比例的最大值的平均值作为异常分数
#                     anomaly_map = anomaly_map.flatten(1)
#                     sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
#                                :int(anomaly_map.shape[1] * max_ratio)]
#                     sp_score = sp_score.mean(dim=1)
#                 sp_score_list.extend(sp_score.tolist())
#             # 执行可视化
#             visualize_save_dir = f"dinomaly_dinov2_{self.model_size}_predict"
#             visualize_when_predict(self.model, dataloader=pred_dataloader, device=self.device, _class_="predict",
#                                    save_name=visualize_save_dir)
#             print(f"Visualization done!! Saved to ./visualize/{visualize_save_dir}")
#
#         return sp_score_list  # 返回计算得到的异常分数


class ModelConfig:
    """模型配置类，统一管理不同模型的配置参数"""

    V3_CONFIGS = {
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

    V2_CONFIGS = {
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

    def __init__(self, model_path: str, model_size: str, device: str = 'cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_size = model_size
        self.model = self._load_model(model_path)
        self.model.eval()
        self.batch_size = 8
        self.transform = get_data_transforms(512, 448)
        self.vis_dir = "visualize"

    @abstractmethod
    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型的抽象方法，需要子类实现"""
        pass

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """预处理输入图像"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict_anomaly_score(
            self,
            image_path: str,
            image_size: int = 512,
            crop_size: int = 448,
            max_ratio: float = 0.01,
            resize_mask: int = 256
    ) -> List[float]:
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

        return sp_score_list


class DinomalyDinoV2Inference(DinomalyBaseInferencer):
    """DINOv2异常检测器实现"""

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
        config = ModelConfig.V2_CONFIGS[self.model_size]

        encoder = vit_encoder.load(config['encoder_name'])

        return self._create_model(encoder, config, model_path)


class DinomalyDinoV3Inference(DinomalyBaseInferencer):
    """DINOv3异常检测器实现"""

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
        config = ModelConfig.V3_CONFIGS[self.model_size]

        encoder = load_dinov3_model(
            config['encoder_name'],
            layers_to_extract_from=config['target_layers'],
            pretrained_weight_path=config['encoder_weight']
        )

        return self._create_model(encoder, config, model_path)


if __name__ == '__main__':
    # 初始化检测器
    detector = DinomalyDinoV2Inference(
        model_path='saved_results/dinomaly_dinov2_small_carpet_grid_epoch_500_Sat Dec 20 19:52:36 2025.pth',
        model_size='small')

    # detector = DinomalyDinoV3Inference(
    #     model_path='saved_results/dinomaly_dinov3_small_carpet_grid_epoch_500_Sat Dec 20 19:55:45 2025.pth',
    #     model_size='small')

    # 单张图像检测
    # image = r'C:\Users\W0401544_ZCH\PycharmProjects\ASPKD\data\spk_251210\dk_22050\test\bad\0a1438fbe30547ddf944665f4e17cd4f.png'
    # anomaly_score = detector.predict_anomaly_score(image)
    # print(anomaly_score)

    # 检测文件夹
    image = r'minitest'
    anomaly_score = detector.predict_anomaly_score(image)
    print(anomaly_score)

    # 批量检测
    # image_list = [Image.open(f) for f in ['test1.jpg', 'test2.jpg']]
    # anomaly_maps, anomaly_scores = detector.batch_detect(image_list)
