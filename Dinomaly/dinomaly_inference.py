import os
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import TensorDataset, Dataset
from torch.nn import functional as F

from dataset import get_data_transforms, PredictDataset
from dinov3.hub.backbones import load_dinov3_model
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import visualize, get_gaussian_kernel, cal_anomaly_maps, show_cam_on_image, min_max_norm, cvt2heatmap, \
    visualize_when_predict


class DinomalyInference:
    def __init__(self, model_path, model_size='base', device='cuda:0'):
        """
        初始化异常检测器
        Args:
            model_path: 训练好的模型权重路径
            model_size: 模型大小，'base' 或 'large'
            device: 运行设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_size = model_size
        self.model = self._load_model(model_path)
        self.model.eval()
        self.batch_size = 8

        # 获取数据预处理方法
        self.transform = get_data_transforms(size=512, isize=448)[0]
        self.vis_dir = "visualize"

    def _load_model(self, model_path):
        """
        加载模型
        Args:
            model_path: 模型权重路径
        Returns:
            加载好的模型
        """
        if self.model_size == "base":
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
            fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
            fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
            encoder_name = 'dinov3_vitb16'
            encoder_weight = 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
            embed_dim, num_heads = 768, 12

        elif self.model_size == "large":
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
            fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
            fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
            encoder_name = 'dinov3_vitl16'
            encoder_weight = 'weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
            embed_dim, num_heads = 1024, 16

        # 加载预训练编码器
        encoder = load_dinov3_model(encoder_name, layers_to_extract_from=target_layers,
                                    pretrained_weight_path=encoder_weight)

        # 初始化瓶颈层和解码器
        bottleneck = []
        decoder = []

        # 添加瓶颈层模块
        bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
        bottleneck = nn.ModuleList(bottleneck)

        # 添加解码器模块
        for i in range(8):
            blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                           qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                           attn=LinearAttention2)
            decoder.append(blk)
        decoder = nn.ModuleList(decoder)

        # 创建完整模型
        model = ViTill(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            target_layers=target_layers,
            mask_neighbor_size=0,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder
        )
        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        return model

    def preprocess_image(self, image):
        """
        预处理输入图像
        Args:
            image: PIL Image 或 numpy array
        Returns:
            预处理后的tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict_anomaly_score(self, image_path, image_size=512, crop_size=448, max_ratio=0.01, resize_mask=256):
        """
        预测图像的异常分数
        参数:
            image_path (str): 输入图像的路径
            image_size (int): 输入图像的目标大小，默认为512
            crop_size (int): 裁剪大小，默认为448
            max_ratio (float): 用于计算异常分数的最大比例，默认为0.01
            resize_mask (int): 调整异常图大小的目标尺寸，默认为256
        返回:
            torch.Tensor: 图像的异常分数
        """
        # 获取数据变换
        data_transform, _ = get_data_transforms(image_size, crop_size)

        # 获取高斯核
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(self.device)  # 创建高斯核并移动到设备

        pred_data = PredictDataset(input_img_pth=image_path, transform=data_transform)
        pred_dataloader = torch.utils.data.DataLoader(pred_data, batch_size=self.batch_size, shuffle=False,
                                                      num_workers=4)

        # 预测
        with torch.no_grad():  # 关闭梯度计算以节省内存
            for img, img_path in pred_dataloader:
                img = img.to(self.device)  # 确保图像在正确的设备上
                # starter.record()
                output = self.model(img)  # 通过模型获取输出
                # ender.record()
                # torch.cuda.synchronize()
                # curr_time = starter.elapsed_time(ender)
                en, de = output[0], output[1]  # 分离编码器和解码器的输出

                # 计算anomaly_maps
                anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])  # 计算异常图

                if resize_mask is not None:  # 如果需要调整异常图大小
                    anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear',
                                                align_corners=False)  # 使用双线性插值调整大小

                anomaly_map = gaussian_kernel(anomaly_map)  # 应用高斯核平滑异常图
                if max_ratio == 0:  # 如果max_ratio为0，使用最大值作为异常分数
                    sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
                else:  # 否则，使用前max_ratio比例的最大值的平均值作为异常分数
                    anomaly_map = anomaly_map.flatten(1)
                    sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                               :int(anomaly_map.shape[1] * max_ratio)]
                    sp_score = sp_score.mean(dim=1)
            # 执行可视化
            visualize_when_predict(self.model, dataloader=pred_dataloader, device=self.device, _class_="predict",
                                   save_name=f"dinomaly_model_size_{self.model_size}_predict")

        return sp_score  # 返回计算得到的异常分数


if __name__ == '__main__':
    # 初始化检测器
    detector = DinomalyInference(
        model_path='saved_results/vitill_mvtec_uni_dinov3/Dinomaly_base_epoch_100_Sun Nov 30 15:03:54 2025.pth',
        model_size='base')

    # 单张图像检测
    image = '004.png'
    anomaly_score = detector.predict_anomaly_score(image)
    print(anomaly_score)

    # 批量检测
    # image_list = [Image.open(f) for f in ['test1.jpg', 'test2.jpg']]
    # anomaly_maps, anomaly_scores = detector.batch_detect(image_list)
