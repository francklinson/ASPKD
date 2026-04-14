"""
统一算法适配器模块
所有异常检测算法的统一调用入口

使用示例:
    >>> from algorithms import create_detector
    >>> detector = create_detector("dinomaly_dinov3_small")
    >>> detector.load_model()
    >>> result = detector.predict("path/to/image.png")
"""

from core import BaseDetector, DetectionResult, AlgorithmRegistry

__all__ = [
    'BaseDetector',
    'DetectionResult',
    'AlgorithmRegistry',
]

# 延迟导入工厂函数，避免启动时加载所有适配器（会触发 torch 初始化）
def create_detector(algorithm_name: str, *args, **kwargs):
    """创建检测器 - 延迟加载工厂模块"""
    from .factory import create_detector as _create_detector
    return _create_detector(algorithm_name, *args, **kwargs)

def list_available_algorithms():
    """列出可用算法 - 延迟加载工厂模块"""
    from .factory import list_available_algorithms as _list_algorithms
    return _list_algorithms()

__all__.extend(['create_detector', 'list_available_algorithms'])
