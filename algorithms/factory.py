"""
算法工厂 - 统一创建检测器实例
"""

from typing import Optional, Dict, List
from core import AlgorithmRegistry, ConfigManager
from core.base_detector import BaseDetector


# 延迟导入适配器模块的标志
_adapters_imported = False

def _import_adapters():
    """延迟导入所有适配器模块，避免启动时触发 torch 初始化"""
    global _adapters_imported
    if _adapters_imported:
        return
    
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
    
    _adapters_imported = True


def create_detector(algorithm_name: str, 
                   model_path: Optional[str] = None,
                   config_manager: Optional[ConfigManager] = None,
                   **kwargs) -> BaseDetector:
    """
    工厂函数 - 创建检测器实例
    
    Args:
        algorithm_name: 算法名称
        model_path: 模型路径(如为None则从配置中读取)
        config_manager: 配置管理器实例
        **kwargs: 额外参数
        
    Returns:
        BaseDetector: 检测器实例
        
    Raises:
        ValueError: 算法不存在或配置错误
        
    使用示例:
        >>> detector = create_detector("dinomaly_dinov3_small")
        >>> detector.load_model()
        >>> result = detector.predict("image.png")
    """
    import os
    
    # 延迟导入适配器（确保 CUDA 环境变量已设置）
    _import_adapters()
    
    # 使用配置管理器获取配置
    if config_manager is None:
        # 尝试使用绝对路径加载配置
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        config_path = os.path.join(project_root, "config", "config.yaml")
        print(f"[DEBUG] create_detector: 使用默认配置路径: {config_path}")
        config_manager = ConfigManager(config_path)
    
    # 如果未提供模型路径，从配置中获取
    if model_path is None:
        # 尝试多种方式解析 algorithm_name
        # 方式1: 完整匹配 (如 dinomaly_dinov3_small -> dinomaly + dinov3_small)
        model_path = _resolve_model_path(config_manager, algorithm_name)
        
        if model_path is None:
            print(f"[ERROR] 无法解析算法 '{algorithm_name}' 的模型路径")
            print(f"[ERROR] 配置中的 models: {list(config_manager.config.get('models', {}).keys())}")
            raise ValueError(f"未找到算法 '{algorithm_name}' 的模型路径配置，请检查 config/config.yaml")
        
        print(f"[DEBUG] create_detector: 解析到的模型路径: {model_path}")
        
    # 确保 model_path 不为 None 或空字符串
    if not model_path:
        raise ValueError(f"模型路径无效: {model_path}")
    
    # 从算法配置中获取阈值（如果 kwargs 中未提供）
    if 'threshold' in kwargs:
        threshold = kwargs.pop('threshold')
    else:
        threshold = config_manager.get_threshold(algorithm_name)
    
    # 创建实例
    detector = AlgorithmRegistry.create(
        name=algorithm_name,
        model_path=model_path,
        threshold=threshold,
        **kwargs
    )
    
    if detector is None:
        raise ValueError(f"创建检测器失败: {algorithm_name}")
    
    return detector


def list_available_algorithms() -> List[str]:
    """
    列出所有可用算法
    
    Returns:
        算法名称列表
    """
    return AlgorithmRegistry.list_algorithms()


def get_algorithm_info(algorithm_name: str) -> Optional[Dict]:
    """
    获取算法详细信息
    
    Args:
        algorithm_name: 算法名称
        
    Returns:
        算法信息字典
    """
    return AlgorithmRegistry.get_algorithm_info(algorithm_name)


def _resolve_model_path(config_manager: ConfigManager, algorithm_name: str) -> Optional[str]:
    """
    解析算法名称获取模型路径
    
    支持多种命名格式:
    - dinomaly_dinov3_small -> (dinomaly, dinov3_small)
    - mambaad -> (mambaad, default)
    """
    print(f"[DEBUG] _resolve_model_path: algorithm_name={algorithm_name}")
    
    # 尝试1: 完整名称作为algorithm，尝试常见variant
    variants_to_try = ['dinov3_small', 'dinov3_large', 'dinov2_small', 'dinov2_large', 
                       'small', 'large', 'base', 'default']
    
    for variant in variants_to_try:
        if algorithm_name.endswith(f'_{variant}'):
            algo = algorithm_name[:-len(f'_{variant}')]
            print(f"[DEBUG] Trying algo='{algo}', variant='{variant}'")
            path = config_manager.get_model_path(algo, variant)
            print(f"[DEBUG] Path result: {path}")
            if path:
                return path
    
    # 尝试2: 分割成两部分 (如 dinomaly_dinov3_small -> dinomaly + dinov3_small)
    parts = algorithm_name.split('_', 1)
    if len(parts) == 2:
        algo, variant = parts
        print(f"[DEBUG] Trying split: algo='{algo}', variant='{variant}'")
        path = config_manager.get_model_path(algo, variant)
        print(f"[DEBUG] Path result: {path}")
        if path:
            return path
    
    # 尝试3: 作为单一算法名，使用default variant
    print(f"[DEBUG] Trying default variant for '{algorithm_name}'")
    path = config_manager.get_model_path(algorithm_name, 'default')
    print(f"[DEBUG] Path result: {path}")
    if path:
        return path
    
    # 尝试4: 使用完整名称作为variant，尝试常见算法名
    common_algos = ['dinomaly', 'mambaad', 'invad', 'vitad', 'unad', 
                    'cflow', 'pyramidflow', 'simplenet', 'patchcore', 
                    'efficient_ad', 'padim', 'denseae', 'cae', 'vae', 'aegan', 'differnet',
                    'hiad', 'multiads', 'musc', 'dictas', 'subspacead', 'diad']
    for algo in common_algos:
        path = config_manager.get_model_path(algo, algorithm_name)
        if path:
            print(f"[DEBUG] Found via common algo '{algo}': {path}")
            return path
    
    print(f"[DEBUG] No path found for '{algorithm_name}'")
    return None
