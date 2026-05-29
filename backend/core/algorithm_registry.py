"""
算法注册表 - 实现算法自动发现和动态加载
使用装饰器模式实现算法注册
"""

import inspect
from typing import Dict, Type, Optional, List
from .base_detector import BaseDetector


class AlgorithmRegistry:
    """
    算法注册表单例类
    
    用于管理所有可用的异常检测算法，支持动态发现和实例化
    """
    _instance = None
    _algorithms: Dict[str, Type[BaseDetector]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, name: str, algorithm_class: Type[BaseDetector]) -> None:
        """
        注册算法
        
        Args:
            name: 算法标识名
            algorithm_class: 算法类(必须继承BaseDetector)
        """
        if not issubclass(algorithm_class, BaseDetector):
            raise TypeError(f"算法类必须继承BaseDetector: {algorithm_class}")
        
        cls._algorithms[name] = algorithm_class
        print(f"[Registry] 注册算法: {name} -> {algorithm_class.__name__}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseDetector]]:
        """
        获取算法类
        
        Args:
            name: 算法标识名
            
        Returns:
            算法类或None
        """
        return cls._algorithms.get(name)
    
    @classmethod
    def create(cls, 
               name: str, 
               model_path: str,
               **kwargs) -> Optional[BaseDetector]:
        """
        创建算法实例
        
        Args:
            name: 算法标识名
            model_path: 模型路径
            **kwargs: 其他参数
            
        Returns:
            算法实例或None
        """
        algorithm_class = cls.get(name)
        if algorithm_class is None:
            raise ValueError(f"未找到算法: {name}。可用算法: {list(cls._algorithms.keys())}")
        
        return algorithm_class(model_path=model_path, **kwargs)
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """
        列出所有已注册算法
        
        Returns:
            算法名称列表
        """
        return list(cls._algorithms.keys())
    
    @classmethod
    def get_algorithm_info(cls, name: str) -> Optional[Dict]:
        """
        获取算法信息
        
        Args:
            name: 算法名
            
        Returns:
            算法信息字典
        """
        algo_class = cls.get(name)
        if algo_class is None:
            return None
        
        return {
            'name': name,
            'class_name': algo_class.__name__,
            'doc': algo_class.__doc__,
            'parameters': cls._get_parameters(algo_class)
        }
    
    @classmethod
    def _get_parameters(cls, algorithm_class: Type[BaseDetector]) -> List[Dict]:
        """获取算法参数信息"""
        sig = inspect.signature(algorithm_class.__init__)
        params = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            params.append({
                'name': param_name,
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty
            })
        return params
    
    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        cls._algorithms.clear()


# 便捷装饰器
def register_algorithm(name: str):
    """
    算法注册装饰器
    
    使用示例:
        @register_algorithm("patchcore")
        class PatchCoreDetector(BaseDetector):
            pass
    """
    def decorator(cls: Type[BaseDetector]):
        AlgorithmRegistry.register(name, cls)
        return cls
    return decorator
