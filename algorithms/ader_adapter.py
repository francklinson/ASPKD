"""
ADer框架算法适配器
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


class ADerBaseAdapter(BaseDetector):
    """ADer框架基础适配器"""
    
    def __init__(self, model_path: str, method: str, device: str = 'auto', 
                 threshold: float = 0.5, config_path: Optional[str] = None, **kwargs):
        super().__init__(model_path, device, threshold, **kwargs)
        self.method = method
        self.config_path = config_path
        self._assigner = None
        
    def load_model(self) -> None:
        """加载ADer模型"""
        from ADer import ADerTaskAssigner
        
        self._assigner = ADerTaskAssigner(method=self.method)
        # ADer在test/train时动态加载模型
        self.is_loaded = True
        
    def predict(self, image_path: str) -> DetectionResult:
        """单张图像推理"""
        if not self.is_loaded:
            self.load_model()
            
        start_time = time.time()
        
        # ADer使用inference模式
        # 需要将图像放到指定目录
        import shutil
        inference_dir = 'inference_dir_temp'
        os.makedirs(inference_dir, exist_ok=True)
        
        # 复制图像到推理目录
        dst_path = os.path.join(inference_dir, os.path.basename(image_path))
        shutil.copy(image_path, dst_path)
        
        # 执行推理
        try:
            self._assigner.inference(inference_dir=inference_dir)
            # TODO: 从vis目录或结果中读取异常分数
            # 这里需要根据ADer实际输出格式解析
            score = 0.5  # 占位
            is_anomaly = score > self.threshold
        finally:
            # 清理临时文件
            if os.path.exists(dst_path):
                os.remove(dst_path)
        
        inference_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=None,
            inference_time=inference_time,
            metadata={'method': self.method}
        )
    
    def release(self) -> None:
        """释放资源"""
        self._assigner = None
        super().release()


# 注册ADer各算法
@register_algorithm("mambaad")
class MambaADAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='MambaAD', **kwargs)


@register_algorithm("invad")
class InVadAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='InVad', **kwargs)


@register_algorithm("vitad")
class ViTADAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='ViTAD', **kwargs)


@register_algorithm("unad")
class UniADAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='UniAD', **kwargs)


@register_algorithm("cflow")
class CFlowAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='CFlow', **kwargs)


@register_algorithm("pyramidflow")
class PyramidFlowAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='PyramidFlow', **kwargs)


@register_algorithm("simplenet")
class SimpleNetAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='SimpleNet', **kwargs)
