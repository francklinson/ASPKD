"""
BaseASD基础算法适配器
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image

# 添加 algorithms 目录到路径，使用完整包名导入
_algorithms_dir = os.path.dirname(os.path.abspath(__file__))
if _algorithms_dir not in sys.path:
    sys.path.insert(0, _algorithms_dir)

from core import BaseDetector, DetectionResult, register_algorithm


class BaseASDAdapter(BaseDetector):
    """
    BaseASD统一适配器
    """
    
    def __init__(self, model_path: str, method: str, device: str = 'auto',
                 threshold: float = 0.5, **kwargs):
        super().__init__(model_path, device, threshold, **kwargs)
        self.method = method
        self._interface = None
        
    def load_model(self) -> None:
        """加载BaseASD模型"""
        import time
        start_time = time.time()
        
        print(f"[BaseASD] {'='*60}")
        print(f"[BaseASD] [MODEL LOAD START] BaseASD Traditional Detector")
        print(f"[BaseASD] {'='*60}")
        print(f"[BaseASD] Configuration:")
        print(f"[BaseASD]   - Method: {self.method}")
        print(f"[BaseASD]   - Device: {self.device}")
        
        method_names = {
            'denseae': 'Dense Autoencoder',
            'cae': 'Convolutional Autoencoder',
            'vae': 'Variational Autoencoder',
            'aegan': 'AE+GAN',
            'differnet': 'DifferNet (Normalizing Flow)'
        }
        print(f"[BaseASD]   - Full name: {method_names.get(self.method, self.method)}")
        
        print(f"[BaseASD] Loading model interface...")
        interface_start = time.time()
        
        if self.method == 'denseae':
            from BaseASD.DenseAE.DenseAE_interface import DenseAEInterface
            self._interface = DenseAEInterface()
        elif self.method == 'cae':
            from BaseASD.ConvolutionalAE.CAE_interface import CAEInterface
            self._interface = CAEInterface()
        elif self.method == 'vae':
            from BaseASD.VAE.VAE_interface import VAEInterface
            self._interface = VAEInterface()
        elif self.method == 'aegan':
            from BaseASD.AEGAN.AeGan_interface import AEGANInterface
            self._interface = AEGANInterface()
        elif self.method == 'differnet':
            from BaseASD.DifferNet.DifferNet_interface import DifferNetInterface
            self._interface = DifferNetInterface()
        
        interface_time = time.time() - interface_start
        print(f"[BaseASD] ✓ Interface loaded in {interface_time:.2f}s")
        
        self.is_loaded = True
        total_time = time.time() - start_time
        print(f"[BaseASD] {'='*60}")
        print(f"[BaseASD] [MODEL LOAD COMPLETE] Total time: {total_time:.2f}s")
        print(f"[BaseASD] {'='*60}")
        
    def predict(self, image_path: str) -> DetectionResult:
        """单张图像推理"""
        if not self.is_loaded:
            self.load_model()
            
        start_time = time.time()
        
        # BaseASD接口直接支持图像路径
        result = self._interface.judge_is_normal(image_path)
        
        inference_time = (time.time() - start_time) * 1000
        
        # BaseASD返回True/False，需要转换
        is_normal = result if isinstance(result, bool) else result[0]
        is_anomaly = not is_normal
        
        # 模拟异常分数 (BaseASD有些方法不直接输出分数)
        score = 1.0 if is_anomaly else 0.0
        
        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=None,
            inference_time=inference_time,
            metadata={'method': self.method}
        )
    
    def release(self) -> None:
        """释放资源"""
        self._interface = None
        super().release()


# 注册BaseASD各算法
@register_algorithm("denseae")
class DenseAEAdapter(BaseASDAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='denseae', **kwargs)


@register_algorithm("cae")
class CAEAdapter(BaseASDAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='cae', **kwargs)


@register_algorithm("vae")
class VAEAdapter(BaseASDAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='vae', **kwargs)


@register_algorithm("aegan")
class AEGANAdapter(BaseASDAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='aegan', **kwargs)


@register_algorithm("differnet")
class DifferNetAdapter(BaseASDAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='differnet', **kwargs)
