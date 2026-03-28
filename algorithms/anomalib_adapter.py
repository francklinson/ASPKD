"""
Anomalib库算法适配器
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from typing import Optional

# 添加 algorithms 目录到路径，使用完整包名导入
_algorithms_dir = os.path.dirname(os.path.abspath(__file__))
if _algorithms_dir not in sys.path:
    sys.path.insert(0, _algorithms_dir)

from core import BaseDetector, DetectionResult, register_algorithm


class AnomalibAdapter(BaseDetector):
    """
    Anomalib统一适配器
    支持所有Anomalib图像算法
    """
    
    def __init__(self, model_path: str, model_name: str, device: str = 'auto',
                 threshold: float = 0.5, **kwargs):
        super().__init__(model_path, device, threshold, **kwargs)
        self.model_name = model_name
        self._model = None
        self._engine = None
        
    def load_model(self) -> None:
        """加载Anomalib模型"""
        from Anomalib.models import get_model
        from Anomalib.engine import Engine
        
        # 创建模型
        self._model = get_model(self.model_name)
        
        # 加载权重
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self._model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        
        self._model.to(self.device)
        self._model.eval()
        
        # 创建引擎
        self._engine = Engine()
        self.is_loaded = True
        
    def predict(self, image_path: str) -> DetectionResult:
        """单张图像推理"""
        if not self.is_loaded:
            self.load_model()
            
        start_time = time.time()
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # 预处理
        input_tensor = self.preprocess(image_np)
        
        # 推理
        with torch.no_grad():
            output = self._model(input_tensor)
        
        # 后处理
        result = self.postprocess(output)
        result.inference_time = (time.time() - start_time) * 1000
        
        return result
    
    def postprocess(self, output: torch.Tensor) -> DetectionResult:
        """后处理"""
        # Anomalib输出格式处理
        if isinstance(output, dict):
            anomaly_score = output.get('anomaly_score', 0.0)
            anomaly_map = output.get('anomaly_map', None)
        else:
            anomaly_score = output.item() if output.numel() == 1 else output.mean().item()
            anomaly_map = None
        
        # 转换为numpy
        if anomaly_map is not None:
            anomaly_map = anomaly_map.squeeze().cpu().numpy()
        
        is_anomaly = anomaly_score > self.threshold
        
        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            anomaly_map=anomaly_map,
            inference_time=0.0,  # 由predict填充
            metadata={'model_name': self.model_name}
        )
    
    def release(self) -> None:
        """释放资源"""
        self._model = None
        self._engine = None
        super().release()


# 注册Anomalib各算法
# 注意: dinomaly使用Dinomaly路径下的原生实现，不在此处注册
ANOMALIB_MODELS = [
    'patchcore', 'cfa', 'csflow', 'dfkde', 'dfm', 'draem', 
    'dsr', 'efficient_ad', 'fastflow', 'fre',
    'padim', 'reverse_distillation', 'stfpm', 'ganomaly',
    'supersimplenet', 'uflow', 'uninet', 'vlm_ad', 'winclip'
]

for model_name in ANOMALIB_MODELS:
    @register_algorithm(model_name)
    class AdapterClass(AnomalibAdapter):
        def __init__(self, model_path: str, **kwargs):
            super().__init__(model_path, model_name=model_name, **kwargs)
    
    # 重命名类
    AdapterClass.__name__ = f"{model_name.capitalize()}Adapter"
