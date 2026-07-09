"""
Dinomaly2 算法适配器
"One Dinomaly2 Detect Them All" — 统一全频谱无监督异常检测
GitHub: https://github.com/guojiajeremy/Dinomaly2

支持: DINOv2 (reg) / DINOv3 Small / Base / Large 骨干网络
"""

import os
import sys
import time
import math
import numpy as np
from typing import Optional
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from backend.core import BaseDetector, DetectionResult, register_algorithm

# 确保 Dinomaly2 模块在路径中（内部文件使用平铺导入如 from dinov1 import ...）
_algorithms_dir = os.path.dirname(os.path.abspath(__file__))
_DINOMALY2_DIR = os.path.join(_algorithms_dir, "Dinomaly2")
if _algorithms_dir not in sys.path:
    sys.path.insert(0, _algorithms_dir)
if _DINOMALY2_DIR not in sys.path:
    sys.path.insert(0, _DINOMALY2_DIR)

# 在模块级别导入 Dinomaly2 的内部模块（必须在 DiAD 适配器之前，避免 models 命名冲突）
# 关键: vit_encoder.load() 内部有 flat imports (from dinov1 import ..., from dinov2.models import ...)
# 这些需要在 sys.path 中有 Dinomaly2 目录，且不能被 DiAD 的 models 缓存污染
_DINOMALY2_DIR = os.path.join(_algorithms_dir, "Dinomaly2")
if _DINOMALY2_DIR not in sys.path:
    sys.path.insert(0, _DINOMALY2_DIR)

# 在模块级别预先导入所有 Dinomaly2 内部模块（此时 models 命名空间尚无冲突）
try:
    from models.uad import Dinomaly as _D2_Dinomaly
    from models import vit_encoder as _D2_vit_encoder
    from models.vision_transformer import Block as _D2_VitBlock
    from models.vision_transformer import LinearAttention2 as _D2_LinearAttention2
    from models.vision_transformer import Attention as _D2_Attention
    from dataset import get_data_transforms as _D2_get_data_transforms
    _D2_MODELS = {
        'Dinomaly': _D2_Dinomaly,
        'vit_encoder': _D2_vit_encoder,
        'VitBlock': _D2_VitBlock,
        'LinearAttention2': _D2_LinearAttention2,
        'Attention': _D2_Attention,
        'get_data_transforms': _D2_get_data_transforms,
    }
except ImportError as e:
    print(f"[Dinomaly2Adapter] Pre-import warning (will retry at load time): {e}")
    _D2_MODELS = None

def _get_d2_models():
    """获取 Dinomaly2 内部模块（已在模块级别预加载）"""
    global _D2_MODELS
    if _D2_MODELS is not None:
        return _D2_MODELS
    # 回退: 运行时加载
    import sys as _sys
    _sys.path.insert(0, _DINOMALY2_DIR)
    from models.uad import Dinomaly as UADDinomaly
    from models import vit_encoder
    from models.vision_transformer import Block as VitBlock, LinearAttention2, Attention
    from dataset import get_data_transforms
    _D2_MODELS = {
        'Dinomaly': UADDinomaly,
        'vit_encoder': vit_encoder,
        'VitBlock': VitBlock,
        'LinearAttention2': LinearAttention2,
        'Attention': Attention,
        'get_data_transforms': get_data_transforms,
    }
    return _D2_MODELS


# 骨干网络配置: {key: (embed_dim, num_heads, target_layers, use_get_intermediate)}
_BACKBONE_CONFIG = {
    # --- DINOv2 with registers ---
    "dinov2reg_vit_small_14":  (384,  6,  [2, 3, 4, 5, 6, 7, 8, 9],   False),
    "dinov2reg_vit_base_14":   (768,  12, [2, 3, 4, 5, 6, 7, 8, 9],   False),
    "dinov2reg_vit_large_14":  (1024, 16, [4, 6, 8, 10, 12, 14, 16, 18], False),
    # --- DINOv3 (命名需兼容 vit_encoder.load: xxx_arch_patchsize, arch=small/base/large) ---
    "dinov3_vit_small_16":     (384,  6,  [2, 3, 4, 5, 6, 7, 8, 9],   True),
    "dinov3_vit_base_16":      (768,  12, [2, 3, 4, 5, 6, 7, 8, 9],   True),
    "dinov3_vit_large_16":     (1024, 16, [4, 6, 8, 10, 12, 14, 16, 18], True),
}


def _resolve_backbone(backbone: str):
    """解析骨干网络名称，返回完整配置"""
    if backbone in _BACKBONE_CONFIG:
        return _BACKBONE_CONFIG[backbone]

    # 模糊匹配: 通过 small/base/large + v2/v3 关键词
    is_v3 = "v3" in backbone.lower() or "dinov3" in backbone.lower()
    is_large = "large" in backbone.lower()
    is_base = "base" in backbone.lower()

    if is_v3:
        embed_dim, heads, layers, use_gi = (1024, 16, [4, 6, 8, 10, 12, 14, 16, 18], True) if is_large else \
                                           (768, 12, [2, 3, 4, 5, 6, 7, 8, 9], True) if is_base else \
                                           (384, 6, [2, 3, 4, 5, 6, 7, 8, 9], True)
    else:
        embed_dim, heads, layers, use_gi = (1024, 16, [4, 6, 8, 10, 12, 14, 16, 18], False) if is_large else \
                                           (768, 12, [2, 3, 4, 5, 6, 7, 8, 9], False) if is_base else \
                                           (384, 6, [2, 3, 4, 5, 6, 7, 8, 9], False)

    return embed_dim, heads, layers, use_gi


class Dinomaly2Inference:
    """Dinomaly2 推理引擎"""

    def __init__(
        self,
        model_path: str,
        backbone: str = "dinov2reg_vit_small_14",
        device: str = "cuda",
        image_size: int = 448,
        crop_size: int = 392,
        lc: int = 2,
        la: bool = True,
        cr: bool = True,
        dropout: float = 0.4,
    ):
        self.model_path = model_path
        self.backbone = backbone
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.image_size = image_size
        self.crop_size = crop_size
        self.lc = lc
        self.la = la
        self.cr = cr
        self.dropout = dropout

        # 解析骨干网络参数
        self._embed_dim, self._num_heads, self._target_layers, self._use_intermediate = \
            _resolve_backbone(backbone)

        self._model = None
        self._transform = None
        self._gt_transform = None

    def _build_model(self):
        """构建 Dinomaly2 模型（自动适配 DINOv2/v3）"""
        d2 = _get_d2_models()
        Dinomaly = d2['Dinomaly']
        vit_encoder = d2['vit_encoder']
        VitBlock = d2['VitBlock']
        LinearAttention2 = d2['LinearAttention2']
        Attention = d2['Attention']

        # 直接用预解析的参数
        embed_dim = self._embed_dim
        num_heads = self._num_heads
        target_layers = self._target_layers
        use_intermediate = self._use_intermediate

        # 1. Fuse layer 配置
        fuse_presets = {
            0: ([[0], [1], [2], [3], [4], [5], [6], [7]],  [[0], [1], [2], [3], [4], [5], [6], [7]]),
            1: ([[0, 1, 2, 3, 4, 5, 6, 7]],            [[0, 1, 2, 3, 4, 5, 6, 7]]),
            2: ([[0, 1, 2, 3], [4, 5, 6, 7]],            [[0, 1, 2, 3], [4, 5, 6, 7]]),
            3: ([[0, 1, 2], [3, 4, 5], [6, 7]],          [[0, 1, 2], [3, 4, 5], [6, 7]]),
            4: ([[0, 1], [2, 3], [4, 5], [6, 7]],        [[0, 1], [2, 3], [4, 5], [6, 7]]),
        }
        fuse_layer_encoder, fuse_layer_decoder = fuse_presets.get(
            self.lc, ([[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 2, 3], [4, 5, 6, 7]]))

        # 2. 构建编码器 — 使用项目预训练模型目录
        # __file__ = .../ASD_for_SPK/algorithms/dinomaly2_adapter.py
        # dirname x 2 = .../ASD_for_SPK (项目根目录)
        pretrained_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "pre_trained"
        )
        encoder = vit_encoder.load(self.backbone, WEIGHTS_DIR=pretrained_dir)

        # 3. 构建瓶颈层 (Noisy Bottleneck)
        bottleneck = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=self.dropout)),
            nn.Sequential(
                nn.Linear(256, embed_dim * 4), nn.GELU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(p=self.dropout)
            )
        ])

        # 4. 构建解码器
        decoder = nn.ModuleList([
            VitBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=partial(LinearAttention2, eps=1e-8) if self.la else Attention
            )
            for _ in range(8)
        ])

        # 5. 组合模型 — DINOv3 自动开 use_get_intermediate
        model = Dinomaly(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            target_layers=target_layers,
            remove_class_token=False,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            context_aware_recenter=self.cr,
            use_get_intermediate=use_intermediate,
        )

        return model

    def load_model(self):
        """加载模型权重"""
        model = self._build_model()
        model = model.to(self.device)

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # 过滤不匹配的键
            model_dict = model.state_dict()
            filtered_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
            model.load_state_dict(filtered_dict, strict=False)
            print(f"[Dinomaly2] Loaded {len(filtered_dict)}/{len(model_dict)} parameter tensors from {self.model_path}")
        else:
            print(f"[Dinomaly2] Warning: model path {self.model_path} not found, using random init")

        model.eval()
        model.init_weights()
        self._model = model

        # 准备数据变换
        get_data_transforms = _get_d2_models()['get_data_transforms']
        self._transform, self._gt_transform = get_data_transforms(self.image_size, self.crop_size)

        print(f"[Dinomaly2] Model ready: {self.backbone}, device={self.device}")

    def predict(self, image_path: str) -> tuple:
        """单张图片推理，返回 (score, anomaly_map)"""
        import torchvision.transforms as T

        img = Image.open(image_path).convert("RGB")
        img_tensor = self._transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            en, de = self._model(img_tensor)

        # 计算异常图
        anomaly_map = self._compute_anomaly_map(en, de, img.size)
        score = float(anomaly_map.max())

        # 平滑和归一化
        if score > 0:
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

        return score, anomaly_map

    def _compute_anomaly_map(self, en, de, original_size):
        """计算异常图 —— 编码器/解码器特征余弦距离"""
        anomaly_maps = []
        for i in range(len(de)):
            a_map = 1 - F.cosine_similarity(en[i], de[i], dim=1)
            # 确保 a_map 是 4D: [B, C, H, W]
            while a_map.dim() < 4:
                a_map = a_map.unsqueeze(0)
            if a_map.dim() == 4:
                a_map = F.interpolate(
                    a_map, size=(original_size[1], original_size[0]),
                    mode="bilinear", align_corners=True
                )
            anomaly_maps.append(a_map)

        anomaly_map = torch.cat(anomaly_maps, dim=1).mean(dim=1, keepdim=True)
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        return anomaly_map


# ============================================================================
# 适配器基类
# ============================================================================

class Dinomaly2BaseAdapter(BaseDetector):
    """Dinomaly2 统一适配器基类"""

    def __init__(self, model_path: str, device: str = "auto",
                 threshold: float = 0.5, backbone: str = "dinov2reg_vit_small_14",
                 model_size: str = "small", **kwargs):
        super().__init__(model_path, device, threshold, **kwargs)
        self.backbone = backbone
        self.model_size = model_size
        self._inferencer = None

    def load_model(self) -> None:
        start = time.time()
        print(f"[Dinomaly2] Loading model: {self.backbone}")

        self._inferencer = Dinomaly2Inference(
            model_path=self.model_path,
            backbone=self.backbone,
            device=self.device,
            image_size=448,
            crop_size=392,
        )
        self._inferencer.load_model()
        self.is_loaded = True

        print(f"[Dinomaly2] Load complete in {time.time() - start:.2f}s")

    def predict(self, image_path: str) -> DetectionResult:
        if not self.is_loaded:
            self.load_model()

        start = time.time()
        score, anomaly_map = self._inferencer.predict(image_path)
        inference_time = (time.time() - start) * 1000

        is_anomaly = score > self.threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=inference_time,
            metadata={"backbone": self.backbone, "threshold": self.threshold}
        )

    def release(self) -> None:
        if self._inferencer and self._inferencer._model:
            del self._inferencer._model
            self._inferencer = None
        super().release()


# ============================================================================
# 注册 Dinomaly2 变体
# ============================================================================

DINOMALY2_VARIANTS = {
    # --- DINOv2 with registers ---
    "dinomaly2_dinov2_small": {
        "name": "Dinomaly2 DINOv2 Small",
        "backbone": "dinov2reg_vit_small_14",
        "model_size": "small",
        "threshold": 0.5,
    },
    "dinomaly2_dinov2_base": {
        "name": "Dinomaly2 DINOv2 Base",
        "backbone": "dinov2reg_vit_base_14",
        "model_size": "base",
        "threshold": 0.5,
    },
    "dinomaly2_dinov2_large": {
        "name": "Dinomaly2 DINOv2 Large",
        "backbone": "dinov2reg_vit_large_14",
        "model_size": "large",
        "threshold": 0.5,
    },
    # --- DINOv3 ---
    "dinomaly2_dinov3_small": {
        "name": "Dinomaly2 DINOv3 Small",
        "backbone": "dinov3_vit_small_16",
        "model_size": "small",
        "threshold": 0.5,
    },
    "dinomaly2_dinov3_base": {
        "name": "Dinomaly2 DINOv3 Base",
        "backbone": "dinov3_vit_base_16",
        "model_size": "base",
        "threshold": 0.5,
    },
    "dinomaly2_dinov3_large": {
        "name": "Dinomaly2 DINOv3 Large",
        "backbone": "dinov3_vit_large_16",
        "model_size": "large",
        "threshold": 0.5,
    },
}

def _make_dinomaly2_adapter(variant: str, info: dict):
    """工厂函数 — 捕获 variant/info 避免 Python 闭包循环变量 bug"""
    @register_algorithm(variant)
    class Adapter(Dinomaly2BaseAdapter):
        def __init__(self, model_path: str, **kwargs):
            kwargs.setdefault("threshold", info["threshold"])
            kwargs.setdefault("backbone", info["backbone"])
            kwargs.setdefault("model_size", info["model_size"])
            super().__init__(model_path, **kwargs)

    # 生成唯一类名
    clean_name = info['name'].replace(' ', '').replace('DINOv2', 'DINOv2')
    Adapter.__name__ = f"Dinomaly2{clean_name}"
    return Adapter

for variant, info in DINOMALY2_VARIANTS.items():
    _make_dinomaly2_adapter(variant, info)
