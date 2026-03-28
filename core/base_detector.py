"""
统一异常检测算法接口基类
所有检测算法必须继承此类并实现抽象方法
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import torch


@dataclass
class DetectionResult:
    """检测结果数据类"""
    is_anomaly: bool                      # 是否异常
    anomaly_score: float                  # 异常分数
    anomaly_map: Optional[np.ndarray]     # 异常热力图 (H, W)
    inference_time: float                 # 推理时间(ms)
    metadata: Dict = None                 # 额外元数据
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseDetector(ABC):
    """
    异常检测算法统一接口基类
    
    所有异常检测算法必须继承此类并实现以下方法：
    - load_model: 加载模型
    - predict: 单张图像推理
    - predict_batch: 批量推理
    
    使用示例:
        >>> detector = SomeDetector(model_path="path/to/model")
        >>> detector.load_model()
        >>> result = detector.predict(image_path="path/to/image.png")
        >>> print(f"异常分数: {result.anomaly_score}, 是否异常: {result.is_anomaly}")
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto',
                 threshold: float = 0.5,
                 **kwargs):
        """
        初始化检测器
        
        Args:
            model_path: 模型权重路径
            device: 运行设备 ('cuda', 'cpu', 'auto')
            threshold: 异常判定阈值
            **kwargs: 算法特定参数
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.threshold = threshold
        self.config = kwargs
        self.model = None
        self.is_loaded = False
        
    def _setup_device(self, device: str) -> torch.device:
        """设置运行设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    @abstractmethod
    def load_model(self) -> None:
        """
        加载模型权重
        必须实现模型加载逻辑
        """
        pass
    
    @abstractmethod
    def predict(self, image_path: str) -> DetectionResult:
        """
        单张图像推理
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            DetectionResult: 检测结果
        """
        pass
    
    def predict_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """
        批量推理 (默认实现为串行处理，子类可覆盖优化)
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            List[DetectionResult]: 检测结果列表
        """
        results = []
        for path in image_paths:
            result = self.predict(path)
            results.append(result)
        return results
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        图像预处理 (子类可覆盖)
        
        Args:
            image: 输入图像 (H, W, C)
            
        Returns:
            torch.Tensor: 预处理后的张量
        """
        # 默认实现：归一化到[0,1]并转为tensor
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def postprocess(self, output: torch.Tensor) -> DetectionResult:
        """
        后处理 (子类必须覆盖)
        
        Args:
            output: 模型输出
            
        Returns:
            DetectionResult: 处理后的检测结果
        """
        raise NotImplementedError("子类必须实现postprocess方法")
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            Dict: 包含模型名称、版本、输入尺寸等信息
        """
        return {
            'name': self.__class__.__name__,
            'device': str(self.device),
            'threshold': self.threshold,
            'is_loaded': self.is_loaded,
            'model_path': self.model_path
        }
    
    def set_threshold(self, threshold: float) -> None:
        """动态调整阈值"""
        self.threshold = threshold
        
    def release(self) -> None:
        """释放资源"""
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        self.is_loaded = False
