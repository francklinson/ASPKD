"""
DiAD 算法适配器
"DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection"
AAAI 2024 — 浙江大学 & 腾讯优图

基于 Stable Diffusion v1.5 + 微调 Autoencoder + Semantic-Guided Network (SG)
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from typing import Optional, List

_algorithms_dir = os.path.dirname(os.path.abspath(__file__))
_DIAD_DIR = os.path.join(_algorithms_dir, "DiAD")
if _DIAD_DIR not in sys.path:
    sys.path.insert(0, _DIAD_DIR)

from backend.core import BaseDetector, DetectionResult, register_algorithm


class DiADDetector(BaseDetector):
    """
    DiAD 异常检测器

    DiAD 使用扩散模型进行异常检测:
    1. AutoencoderKL: 将图像编码到潜空间
    2. Semantic-Guided Network (SG): 基于输入图像引导去噪过程
    3. SD UNet: 在潜空间去噪重建
    4. ResNet50: 提取特征，计算异常图

    模型构建:
    - 需要 Stable Diffusion v1.5 权重 (models/v1-5-pruned.ckpt)
    - 需要微调 Autoencoder (models/mvtecad_fs.ckpt)
    - 运行 build_model.py 生成 models/diad.ckpt
    - 运行 train.py 训练 SG 网络
    """

    def __init__(self, model_path: str, device: str = 'auto',
                 threshold: float = 0.5, **kwargs):
        super().__init__(model_path, device, threshold, **kwargs)
        self._model = None
        self._pretrained = None  # ResNet50 feature extractor
        self._input_size = 256

    # ========================================================================
    # 模型加载
    # ========================================================================

    def load_model(self) -> None:
        """加载 DiAD 模型 + ResNet50 特征提取器"""
        import timm
        from sgn.model import create_model, load_state_dict

        start_time = time.time()
        print(f"[DiAD] Loading model...")

        # 1. 构建/加载 DiAD 模型
        config_path = os.path.join(_DIAD_DIR, "models", "diad.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"DiAD config not found: {config_path}")

        # 检查是否需要先构建模型
        built_model = self.model_path
        if not os.path.exists(built_model):
            # 尝试默认路径
            built_model = os.path.join(_DIAD_DIR, "models", "diad.ckpt")

        if not os.path.exists(built_model):
            print(f"[DiAD] Built model not found, attempting to build...")
            built_model = self._build_model()

        print(f"[DiAD] Loading checkpoint: {built_model}")
        self._model = create_model(config_path).cpu()
        state_dict = load_state_dict(built_model, location='cpu')
        # 过滤不兼容的键
        model_dict = self._model.state_dict()
        compatible = {k: v for k, v in state_dict.items()
                     if k in model_dict and v.shape == model_dict[k].shape}
        self._model.load_state_dict(compatible, strict=False)
        print(f"[DiAD] Loaded {len(compatible)}/{len(model_dict)} parameter tensors")

        self._model.to(self.device)
        self._model.eval()

        # 2. 加载 ResNet50 特征提取器
        print(f"[DiAD] Loading ResNet50 feature extractor...")
        self._pretrained = timm.create_model(
            "resnet50", pretrained=True, features_only=True
        )
        self._pretrained.to(self.device)
        self._pretrained.eval()

        self.is_loaded = True
        elapsed = time.time() - start_time
        print(f"[DiAD] Load complete in {elapsed:.2f}s")

    def _build_model(self) -> str:
        """构建完整 DiAD 模型 (合并 SD + Autoencoder 权重)

        参考 algorithms/DiAD/build_model.py
        """
        from sgn.model import create_model, load_state_dict

        sd_path = os.path.join(_DIAD_DIR, "models", "v1-5-pruned.ckpt")
        ae_path = os.path.join(_DIAD_DIR, "models", "mvtecad_fs.ckpt")
        output_path = os.path.join(_DIAD_DIR, "models", "diad.ckpt")
        config_path = os.path.join(_DIAD_DIR, "models", "diad.yaml")

        for path, name in [(sd_path, "SD v1.5"), (ae_path, "Autoencoder")]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{name} weights not found: {path}\n"
                    f"Download from the DiAD GitHub release page."
                )

        print(f"[DiAD] Building model from {sd_path} + {ae_path}...")

        model = create_model(config_path).cpu()
        ae_state = load_state_dict(ae_path)
        sd_state = torch.load(sd_path, map_location='cpu', weights_only=True)
        if 'state_dict' in sd_state:
            sd_state = sd_state['state_dict']

        scratch_dict = model.state_dict()
        target_dict = {}

        for k in scratch_dict:
            is_control = k.startswith('control_')
            is_first_stage = k.startswith('first_stage_model.')

            if is_control:
                # SG network: 从 SD UNet 复制结构
                copy_k = 'model.diffusion_' + k[len('control_'):]
            elif is_first_stage:
                # Autoencoder: 从微调权重加载
                ae_key = k[len('first_stage_model.'):]
                target_dict[k] = ae_state[ae_key].clone()
                continue
            else:
                copy_k = k

            if copy_k in sd_state:
                target_dict[k] = sd_state[copy_k].clone()
            else:
                target_dict[k] = scratch_dict[k].clone()

        model.load_state_dict(target_dict, strict=True)
        torch.save(model.state_dict(), output_path)
        print(f"[DiAD] Model built and saved to {output_path}")

        del model, ae_state, sd_state
        torch.cuda.empty_cache()

        return output_path

    # ========================================================================
    # 推理
    # ========================================================================

    def predict(self, image_path: str) -> DetectionResult:
        """单张图像异常检测"""
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 预处理 (匹配 MVTecDataset)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((self._input_size, self._input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)  # [1, 3, 256, 256]

        # 构造 batch (匹配 test.py 的 input 格式)
        batch = {
            'jpg': img_tensor,
            'txt': '',
            'hint': img_tensor,  # DiAD 使用图片本身作为 condition
            'filename': [os.path.basename(image_path)],
            'mask': torch.zeros(1, 1, self._input_size, self._input_size),
        }

        # 移动到设备
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            # 1. 提取输入特征 (ResNet50 layers 1-3)
            input_features = self._pretrained(img_tensor)
            input_features = input_features[1:4]  # 取中间3层

            # 2. DiAD 重建
            output = self._model.log_images_test(batch)
            output_img = output['samples']  # [1, 3, 256, 256]

            # 3. 提取重建特征
            output_features = self._pretrained(output_img.to(self.device))
            output_features = output_features[1:4]

            # 4. 计算异常图 (余弦相似度)
            anomaly_map, _ = self._cal_anomaly_map(
                input_features, output_features,
                input_size=self._input_size, amap_mode='a'
            )

            # 5. Gaussian 平滑
            from scipy.ndimage import gaussian_filter
            anomaly_map = gaussian_filter(anomaly_map, sigma=5)

        score = float(anomaly_map.max())
        anomaly_map = anomaly_map.squeeze()

        # 归一化
        if score > 0:
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

        inference_time = (time.time() - start_time) * 1000

        return DetectionResult(
            is_anomaly=score > self.threshold,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=inference_time,
            metadata={
                'model_name': 'diad',
                'mode': 'diffusion',
            }
        )

    @staticmethod
    def _cal_anomaly_map(fs_list, ft_list, input_size=256, amap_mode='a'):
        """计算异常图 (参考 utils/util.py)

        对输入特征和重建特征的各层计算余弦相似度，加权合成异常图。
        """
        anomaly_map = np.zeros([input_size, input_size])
        if amap_mode == 'a':
            # L2 归一化后计算内积
            for i in range(len(ft_list)):
                fs = fs_list[i]
                ft = ft_list[i]
                # L2 normalize
                fs_norm = torch.norm(fs, p=2, dim=1, keepdim=True)
                ft_norm = torch.norm(ft, p=2, dim=1, keepdim=True)
                fs = fs / (fs_norm + 1e-8)
                ft = ft / (ft_norm + 1e-8)
                # 内积 (余弦相似度)
                a_map = 1 - torch.sum(fs * ft, dim=1, keepdim=True)
                a_map = a_map.squeeze().cpu().numpy()
                # 上采样到输入尺寸
                import torch.nn.functional as F
                a_map_t = torch.from_numpy(a_map).unsqueeze(0).unsqueeze(0)
                a_map = F.interpolate(
                    a_map_t, size=(input_size, input_size),
                    mode='bilinear', align_corners=True
                ).squeeze().cpu().numpy()
                anomaly_map += a_map
            anomaly_map = anomaly_map / len(ft_list)
        else:
            # 直接计算差异
            for i in range(len(ft_list)):
                fs = fs_list[i]
                ft = ft_list[i]
                a_map = 1 - torch.nn.functional.cosine_similarity(fs, ft, dim=1)
                a_map = a_map.squeeze().cpu().numpy()
                import torch.nn.functional as F
                a_map_t = torch.from_numpy(a_map).unsqueeze(0).unsqueeze(0)
                a_map = F.interpolate(
                    a_map_t, size=(input_size, input_size),
                    mode='bilinear', align_corners=True
                ).squeeze().cpu().numpy()
                anomaly_map += a_map
            anomaly_map = anomaly_map / len(ft_list)

        return anomaly_map, anomaly_map.max()

    # ========================================================================
    # 训练
    # ========================================================================

    def fit(self, dataset_name: str = 'mvtec', category: str = 'bottle',
            max_epochs: int = 500, learning_rate: float = 1e-5,
            only_mid_control: bool = True) -> dict:
        """
        训练 DiAD 模型

        Args:
            dataset_name: 数据集名称 ('mvtec' 或 'visa')
            category: 类别名称
            max_epochs: 最大训练轮数
            learning_rate: 学习率
            only_mid_control: 只训练中间层的 SG 网络

        Returns:
            训练结果摘要
        """
        if not self.is_loaded:
            self.load_model()

        print(f"[DiAD] Starting training: dataset={dataset_name}, category={category}")

        # DiAD 训练脚本在 algorithms/DiAD/train.py
        # 需要通过子进程运行或直接调用训练逻辑
        # 由于训练涉及复杂的 PyTorch Lightning 配置，推荐通过子进程运行
        import subprocess

        train_script = os.path.join(_DIAD_DIR, "train.py")
        if not os.path.exists(train_script):
            raise FileNotFoundError(f"Training script not found: {train_script}")

        cmd = [
            sys.executable, train_script,
            "--dataset", dataset_name,
            "--category", category,
            "--max_epochs", str(max_epochs),
            "--learning_rate", str(learning_rate),
        ]

        print(f"[DiAD] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=_DIAD_DIR)

        if result.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {result.returncode}")

        return {'status': 'completed'}

    # ========================================================================
    # 资源管理
    # ========================================================================

    def release(self) -> None:
        """释放资源"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._pretrained is not None:
            del self._pretrained
            self._pretrained = None
        torch.cuda.empty_cache()
        super().release()


# ============================================================================
# 注册算法
# ============================================================================

@register_algorithm("diad")
class DiADAdapter(DiADDetector):
    """DiAD 适配器 (注册用)"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
