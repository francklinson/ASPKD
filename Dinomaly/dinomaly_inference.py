import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from dataset import get_data_transforms
from dinov3.hub.backbones import load_dinov3_model
from models.uad import ViTill
from utils import visualize


class AnomalyDetector:
    def __init__(self, model_path, model_size='base', device='cuda'):
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

        # 获取数据预处理方法
        self.transform = get_data_transforms(image_size=512, crop_size=448)[0]

    def _load_model(self, model_path):
        """
        加载模型
        Args:
            model_path: 模型权重路径
        Returns:
            加载好的模型
        """
        if self.model_size == 'base':
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
            fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
            fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
            encoder_name = 'dinov3_vitb16'
            embed_dim, num_heads = 768, 12
        else:  # large
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
            fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
            fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
            encoder_name = 'dinov3_vitl16'
            embed_dim, num_heads = 1024, 16

        # 加载预训练编码器
        encoder = load_dinov3_model(encoder_name, layers_to_extract_from=target_layers)

        # 初始化瓶颈层和解码器
        bottleneck = nn.ModuleList([nn.Linear(embed_dim, embed_dim * 4)])
        decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True
            ) for _ in range(8)
        ])

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

    def detect(self, image):
        """
        检测图像中的异常
        Args:
            image: 输入图像(PIL Image或numpy array)
        Returns:
            异常分数图和异常区域图
        """
        with torch.no_grad():
            # 预处理图像
            img_tensor = self.preprocess_image(image)

            # 模型推理
            en, de = self.model(img_tensor)

            # 计算异常分数
            anomaly_map = torch.mean((en - de) ** 2, dim=1, keepdim=True)
            anomaly_score = torch.max(anomaly_map)

            # 将结果转换为numpy数组
            anomaly_map = anomaly_map.squeeze().cpu().numpy()
            anomaly_score = anomaly_score.item()

            return anomaly_map, anomaly_score

    def visualize_results(self, image, anomaly_map, save_path=None):
        """
        可视化检测结果
        Args:
            image: 原始图像
            anomaly_map: 异常分数图
            save_path: 保存路径(可选)
        """
        visualize(self.model, image, self.device, anomaly_map, save_path)

    def batch_detect(self, image_list):
        """
        批量检测图像
        Args:
            image_list: 图像列表
        Returns:
            异常分数图列表和异常分数列表
        """
        anomaly_maps = []
        anomaly_scores = []

        with torch.no_grad():
            for image in image_list:
                anomaly_map, anomaly_score = self.detect(image)
                anomaly_maps.append(anomaly_map)
                anomaly_scores.append(anomaly_score)

        return anomaly_maps, anomaly_scores


if __name__ == '__main__':
    # 初始化检测器
    detector = AnomalyDetector(
        model_path='path/to/model.pth',
        model_size='base',
        device='cuda'
    )

    # 单张图像检测
    image = Image.open('test.jpg')
    anomaly_map, anomaly_score = detector.detect(image)

    # 可视化结果
    detector.visualize_results(image, anomaly_map, save_path='result.jpg')

    # 批量检测
    image_list = [Image.open(f) for f in ['test1.jpg', 'test2.jpg']]
    anomaly_maps, anomaly_scores = detector.batch_detect(image_list)
