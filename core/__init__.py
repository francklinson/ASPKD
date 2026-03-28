"""
核心模块 - 提供统一的异常检测算法接口和基础设施
"""

from .base_detector import BaseDetector, DetectionResult
from .algorithm_registry import AlgorithmRegistry, register_algorithm
from .config_manager import ConfigManager

__all__ = [
    'BaseDetector',
    'DetectionResult', 
    'AlgorithmRegistry',
    'register_algorithm',
    'ConfigManager'
]
