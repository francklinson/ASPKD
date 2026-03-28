"""
Dinomaly算法适配器
统一接口封装
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Optional
from PIL import Image

# 在导入Dinomaly模块之前设置环境变量
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dinomaly_config_path = os.path.join(_project_root, "config", "asd_gui_config.yaml")
if os.path.exists(_dinomaly_config_path):
    import yaml
    with open(_dinomaly_config_path, 'r', encoding='utf-8') as f:
        dinomaly_config = yaml.safe_load(f) or {}
    env_config = dinomaly_config.get('environments', {})
    for key, value in env_config.items():
        if value:
            os.environ[key] = str(value)
            print(f"[DEBUG] [dinomaly_adapter] Set environment variable: {key}={value}")

from core import BaseDetector, DetectionResult, register_algorithm


class DinomalyBaseAdapter(BaseDetector):
    """Dinomaly基础适配器"""
    
    def __init__(self, model_path: str, device: str = 'auto', threshold: float = 0.033, 
                 model_size: str = 'small', dinov_version: str = 'v3', **kwargs):
        print(f"[DEBUG] DinomalyBaseAdapter.__init__: model_path={model_path}, type={type(model_path)}")
        if not model_path:
            raise ValueError(f"DinomalyBaseAdapter 接收到无效的 model_path: {model_path}")
        super().__init__(model_path, device, threshold, **kwargs)
        self.model_size = model_size
        self.dinov_version = dinov_version
        self._inferencer = None
        
    def load_model(self) -> None:
        """加载Dinomaly模型"""
        # 检查 model_path
        if not self.model_path:
            raise ValueError(f"模型路径无效: {self.model_path}。请检查配置文件中的模型路径设置。")
        
        print(f"[DEBUG] DinomalyBaseAdapter.load_model: model_path={self.model_path}")
        
        # 延迟导入，避免在注册时加载依赖
        # 添加 algorithms 目录到路径，使用完整包名导入
        algorithms_dir = os.path.dirname(os.path.abspath(__file__))
        if algorithms_dir not in sys.path:
            sys.path.insert(0, algorithms_dir)
        
        # 导入时捕获错误
        print("[DEBUG] Importing Dinomaly inference modules...")
        try:
            from Dinomaly.dinomaly_inference import DinomalyDinoV3Inference, DinomalyDinoV2Inference
            print("[DEBUG] Import successful")
        except Exception as e:
            print(f"[ERROR] Import failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"[DEBUG] About to create inferencer, dinov_version={self.dinov_version}")
        print(f"[DEBUG] model_path type: {type(self.model_path)}, value: {repr(self.model_path)}")
        print(f"[DEBUG] model_size: {self.model_size}")
        print(f"[DEBUG] threshold: {self.threshold}")
        
        if self.dinov_version == 'v3':
            print("[DEBUG] Creating DinomalyDinoV3Inference...")
            print(f"[DEBUG] Parameters: model_path={self.model_path}, model_size={self.model_size}, device={self.device}, threshold={self.threshold}")
            try:
                self._inferencer = DinomalyDinoV3Inference(
                    model_path=self.model_path,
                    model_size=self.model_size,
                    device=str(self.device),  # 添加 device 参数
                    threshold=self.threshold
                )
                print("[DEBUG] DinomalyDinoV3Inference created successfully")
            except Exception as e:
                print(f"[ERROR] Failed to create DinomalyDinoV3Inference: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print("[DEBUG] Creating DinomalyDinoV2Inference...")
            print(f"[DEBUG] Parameters: model_path={self.model_path}, model_size={self.model_size}, device={self.device}, threshold={self.threshold}")
            try:
                self._inferencer = DinomalyDinoV2Inference(
                    model_path=self.model_path,
                    model_size=self.model_size,
                    device=str(self.device),  # 添加 device 参数
                    threshold=self.threshold
                )
                print("[DEBUG] DinomalyDinoV2Inference created successfully")
            except Exception as e:
                print(f"[ERROR] Failed to create DinomalyDinoV2Inference: {e}")
                import traceback
                traceback.print_exc()
                raise
        self.is_loaded = True
        print("[DEBUG] Model loaded successfully")
        
    def predict(self, image_path: str) -> DetectionResult:
        """单张图像推理"""
        if not self.is_loaded:
            self.load_model()
            
        start_time = time.time()
        
        # 调用原始推理方法
        result_dict, _ = self._inferencer.predict([image_path])
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # 解析结果
        filename = os.path.basename(image_path)
        if filename in result_dict:
            score, is_anomaly = result_dict[filename]
        else:
            # 尝试匹配key
            key = list(result_dict.keys())[0] if result_dict else None
            if key:
                score, is_anomaly = result_dict[key]
            else:
                score, is_anomaly = 0.0, False
        
        return DetectionResult(
            is_anomaly=bool(is_anomaly),
            anomaly_score=float(score),
            anomaly_map=None,  # Dinomaly原始代码不直接输出热力图
            inference_time=inference_time,
            metadata={'model': f'dinov{self.dinov_version}_{self.model_size}'}
        )
    
    def predict_batch(self, image_paths: list) -> list:
        """批量推理"""
        if not self.is_loaded:
            self.load_model()
            
        start_time = time.time()
        result_dict, heatmap_paths = self._inferencer.predict(image_paths)
        total_time = (time.time() - start_time) * 1000
        
        results = []
        avg_time = total_time / len(image_paths) if image_paths else 0
        
        # 创建热力图路径字典（文件名（不含_heatmap） -> 热力图路径）
        heatmap_dict = {}
        if heatmap_paths:
            for heatmap_path in heatmap_paths:
                # 从热力图路径中提取文件名
                heatmap_filename = os.path.basename(heatmap_path)
                # 移除 _heatmap 后缀得到原始文件名
                original_filename = heatmap_filename.replace('_heatmap.png', '.png')
                heatmap_dict[original_filename] = heatmap_path
                print(f"[DEBUG] Heatmap mapping: {original_filename} -> {heatmap_path}")
        
        for path in image_paths:
            filename = os.path.basename(path)
            # 尝试匹配结果
            result_key = None
            for key in result_dict.keys():
                if filename in key or key in filename:
                    result_key = key
                    break
            
            if result_key:
                score, is_anomaly = result_dict[result_key]
            else:
                score, is_anomaly = 0.0, False
            
            # 查找对应的热力图路径
            heatmap_path = heatmap_dict.get(filename)
            if heatmap_path:
                print(f"[DEBUG] Found heatmap for {filename}: {heatmap_path}")
            else:
                print(f"[DEBUG] No heatmap found for {filename}")
                
            results.append(DetectionResult(
                is_anomaly=bool(is_anomaly),
                anomaly_score=float(score),
                anomaly_map=None,
                inference_time=avg_time,
                metadata={
                    'model': f'dinov{self.dinov_version}_{self.model_size}',
                    'heatmap_path': heatmap_path  # 在metadata中存储热力图路径
                }
            ))
        
        return results
    
    def release(self) -> None:
        """释放资源"""
        self._inferencer = None
        super().release()


# 注册各版本算法
@register_algorithm("dinomaly_dinov3_small")
class DinomalyDinoV3Small(DinomalyBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, dinov_version='v3', model_size='small', **kwargs)


@register_algorithm("dinomaly_dinov3_base")
class DinomalyDinoV3Base(DinomalyBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, dinov_version='v3', model_size='base', **kwargs)


@register_algorithm("dinomaly_dinov3_large")
class DinomalyDinoV3Large(DinomalyBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, dinov_version='v3', model_size='large', **kwargs)


@register_algorithm("dinomaly_dinov2_small")
class DinomalyDinoV2Small(DinomalyBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, dinov_version='v2', model_size='small', **kwargs)


@register_algorithm("dinomaly_dinov2_base")
class DinomalyDinoV2Base(DinomalyBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, dinov_version='v2', model_size='base', **kwargs)


@register_algorithm("dinomaly_dinov2_large")
class DinomalyDinoV2Large(DinomalyBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, dinov_version='v2', model_size='large', **kwargs)
