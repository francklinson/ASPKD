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
from .factory import create_detector, list_available_algorithms

# 自动导入所有算法适配器，完成注册
try:
    from . import dinomaly_adapter
except Exception as e:
    print(f"[algorithms] dinomaly_adapter 导入失败: {e}")

try:
    from . import ader_adapter
except Exception as e:
    print(f"[algorithms] ader_adapter 导入失败: {e}")

try:
    from . import anomalib_adapter
except Exception as e:
    print(f"[algorithms] anomalib_adapter 导入失败: {e}")

try:
    from . import baseasd_adapter
except Exception as e:
    print(f"[algorithms] baseasd_adapter 导入失败: {e}")

try:
    from . import other_adapters
except Exception as e:
    print(f"[algorithms] other_adapters 导入失败: {e}")

try:
    from . import musc_adapter
except Exception as e:
    print(f"[algorithms] musc_adapter 导入失败: {e}")

try:
    from . import subspacead_adapter
except Exception as e:
    print(f"[algorithms] subspacead_adapter 导入失败: {e}")

__all__ = [
    'BaseDetector',
    'DetectionResult',
    'AlgorithmRegistry',
    'create_detector',
    'list_available_algorithms'
]
