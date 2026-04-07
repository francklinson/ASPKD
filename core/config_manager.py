"""
配置管理器 - 统一管理算法配置
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    配置管理器
    
    统一管理：
    1. 算法配置
    2. 模型路径配置
    3. 推理参数配置
    """
    
    DEFAULT_CONFIG_PATH = "config/config.yaml"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.config_path)))
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        print(f"[DEBUG] _load_config: config_path={self.config_path}")
        print(f"[DEBUG] _load_config: exists={os.path.exists(self.config_path)}")
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                print(f"[DEBUG] _load_config: loaded keys={list(config.keys())}")
                return config
        print(f"[DEBUG] _load_config: file not found, using default")
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """默认配置"""
        return {
            'algorithms': {},
            'models': {},
            'inference': {
                'device': 'auto',
                'batch_size': 1,
                'num_workers': 4
            }
        }
    
    def save_config(self) -> None:
        """保存配置到文件"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
    
    def get_algorithm_config(self, name: str) -> Dict[str, Any]:
        """获取算法配置"""
        return self.config.get('algorithms', {}).get(name, {})
    
    def set_algorithm_config(self, name: str, config: Dict[str, Any]) -> None:
        """设置算法配置"""
        if 'algorithms' not in self.config:
            self.config['algorithms'] = {}
        self.config['algorithms'][name] = config
        self.save_config()
    
    def get_model_path(self, algorithm: str, variant: str = 'default') -> Optional[str]:
        """获取模型路径（返回绝对路径）"""
        models = self.config.get('models', {})
        print(f"[DEBUG] ConfigManager.get_model_path: algorithm='{algorithm}', variant='{variant}'")
        print(f"[DEBUG] Available models keys: {list(models.keys())}")
        
        algo_models = models.get(algorithm, {})
        print(f"[DEBUG] Algorithm '{algorithm}' models: {algo_models}")
        
        path = algo_models.get(variant)
        print(f"[DEBUG] Raw path from config: {path}")
        
        if path and not os.path.isabs(path):
            # 将相对路径转为绝对路径
            path = os.path.join(self.base_dir, path)
            print(f"[DEBUG] Converted to absolute: {path}")
        
        return path
    
    def set_model_path(self, algorithm: str, variant: str, path: str) -> None:
        """设置模型路径"""
        if 'models' not in self.config:
            self.config['models'] = {}
        if algorithm not in self.config['models']:
            self.config['models'][algorithm] = {}
        self.config['models'][algorithm][variant] = path
        self.save_config()
    
    def get_inference_config(self) -> Dict[str, Any]:
        """获取推理配置"""
        return self.config.get('inference', {})
    
    def get_threshold(self, algorithm: str, variant: str = 'default') -> float:
        """获取算法阈值"""
        algo_config = self.get_algorithm_config(algorithm)
        thresholds = algo_config.get('thresholds', {})
        return thresholds.get(variant, 0.5)
    
    def list_configured_algorithms(self) -> list:
        """列出所有已配置的算法"""
        return list(self.config.get('algorithms', {}).keys())
