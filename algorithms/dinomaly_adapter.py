"""
Dinomaly算法适配器
统一接口封装
"""

import os
import sys
import time
import numpy as np
from typing import Optional
from PIL import Image

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
        import time
        start_time = time.time()
        
        print(f"[Dinomaly] {'='*60}")
        print(f"[Dinomaly] [MODEL LOAD START] Dinomaly Zero-Shot Detector")
        print(f"[Dinomaly] {'='*60}")
        
        # 检查 model_path
        if not self.model_path:
            raise ValueError(f"模型路径无效: {self.model_path}。请检查配置文件中的模型路径设置。")
        
        # 解析模型路径信息
        model_path_abs = os.path.abspath(self.model_path)
        model_exists = os.path.exists(self.model_path)
        
        print(f"[Dinomaly] Configuration:")
        print(f"[Dinomaly]   - DINOv version: {self.dinov_version}")
        print(f"[Dinomaly]   - Model size: {self.model_size}")
        print(f"[Dinomaly]   - Threshold: {self.threshold}")
        print(f"[Dinomaly]   - Device: {self.device}")
        print(f"[Dinomaly] Model path info:")
        print(f"[Dinomaly]   - Original path: {self.model_path}")
        print(f"[Dinomaly]   - Absolute path: {model_path_abs}")
        print(f"[Dinomaly]   - File exists: {model_exists}")
        
        if model_exists:
            model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            print(f"[Dinomaly]   - File size: {model_size_mb:.2f} MB")
        
        # 延迟导入，避免在注册时加载依赖
        algorithms_dir = os.path.dirname(os.path.abspath(__file__))
        if algorithms_dir not in sys.path:
            sys.path.insert(0, algorithms_dir)
            print(f"[Dinomaly] Added to sys.path: {algorithms_dir}")
        
        # 导入时捕获错误
        print(f"[Dinomaly] [1/2] Importing Dinomaly inference modules...")
        try:
            from Dinomaly.dinomaly_inference import DinomalyDinoV3Inference, DinomalyDinoV2Inference
            print(f"[Dinomaly] ✓ Import successful")
        except Exception as e:
            print(f"[Dinomaly] ✗ Import failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 创建推理器
        print(f"[Dinomaly] [2/2] Creating inference engine...")
        inferencer_start = time.time()
        
        if self.dinov_version == 'v3':
            print(f"[Dinomaly]   - Creating DinomalyDinoV3Inference")
            print(f"[Dinomaly]   - Loading DINOv3 {self.model_size} model weights...")
            try:
                self._inferencer = DinomalyDinoV3Inference(
                    model_path=self.model_path,
                    model_size=self.model_size,
                    device=str(self.device),
                    threshold=self.threshold
                )
                inferencer_time = time.time() - inferencer_start
                print(f"[Dinomaly] ✓ DinomalyDinoV3Inference created in {inferencer_time:.2f}s")
            except Exception as e:
                print(f"[Dinomaly] ✗ Failed to create DinomalyDinoV3Inference: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print(f"[Dinomaly]   - Creating DinomalyDinoV2Inference")
            print(f"[Dinomaly]   - Loading DINOv2 {self.model_size} model weights...")
            try:
                self._inferencer = DinomalyDinoV2Inference(
                    model_path=self.model_path,
                    model_size=self.model_size,
                    device=str(self.device),
                    threshold=self.threshold
                )
                inferencer_time = time.time() - inferencer_start
                print(f"[Dinomaly] ✓ DinomalyDinoV2Inference created in {inferencer_time:.2f}s")
            except Exception as e:
                print(f"[Dinomaly] ✗ Failed to create DinomalyDinoV2Inference: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        self.is_loaded = True
        total_time = time.time() - start_time
        print(f"[Dinomaly] {'='*60}")
        print(f"[Dinomaly] [MODEL LOAD COMPLETE] Total time: {total_time:.2f}s")
        print(f"[Dinomaly] {'='*60}")
        
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
        
        # 延迟导入Dinomaly模块，避免循环导入问题
        from Dinomaly.utils import visualize_when_predict_with_all_images
        from Dinomaly.dataset import get_data_transforms, PredictDataset
            
        start_time = time.time()
        
        # 使用新的可视化函数生成三种图像
        # 1. 准备数据加载器
        import torch
        data_transform, _ = get_data_transforms(512, 448)
        pred_data = PredictDataset(input_img_pth=image_paths, transform=data_transform)
        pred_dataloader = torch.utils.data.DataLoader(
            pred_data,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        # 2. 调用推理获取分数（禁用内部可视化，避免重复生成）
        result_dict, _ = self._inferencer.predict(image_paths, generate_visualization=False)

        # 3. 生成三种图像（原图、叠加图、纯热力图）
        visualize_save_dir = f"dinomaly_{self.dinov_version}_{self.model_size}_predict"
        image_paths_dict = visualize_when_predict_with_all_images(
            self._inferencer.model,
            dataloader=pred_dataloader,
            device=self._inferencer.device,
            _class_="predict",
            save_name=visualize_save_dir
        )
        
        total_time = (time.time() - start_time) * 1000
        
        results = []
        avg_time = total_time / len(image_paths) if image_paths else 0
        
        for path in image_paths:
            filename = os.path.basename(path)
            name = os.path.splitext(filename)[0]  # 不含扩展名的文件名
            
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
            
            # 获取三种图像路径
            paths = image_paths_dict.get(name, {})
            original_path = paths.get('original')
            overlay_path = paths.get('overlay')
            heatmap_path = paths.get('heatmap')
            
            if heatmap_path:
                print(f"[DEBUG] Found images for {filename}: original={original_path}, overlay={overlay_path}, heatmap={heatmap_path}")
            else:
                print(f"[DEBUG] No images found for {filename}")
                
            results.append(DetectionResult(
                is_anomaly=bool(is_anomaly),
                anomaly_score=float(score),
                anomaly_map=None,
                inference_time=avg_time,
                metadata={
                    'model': f'dinov{self.dinov_version}_{self.model_size}',
                    'original_path': original_path,
                    'overlay_path': overlay_path,
                    'heatmap_path': heatmap_path
                }
            ))
        
        return results
    
    def release(self) -> None:
        """释放资源"""
        import time
        print(f"[Dinomaly] [模型释放] 开始释放Dinomaly资源...")
        release_start = time.time()
        
        if self._inferencer is not None:
            print(f"[Dinomaly] [模型释放] 释放推理器对象...")
            self._inferencer = None
            print(f"[Dinomaly] [模型释放] 推理器对象已置为None")
        else:
            print(f"[Dinomaly] [模型释放] 推理器对象为None，无需释放")
        
        print(f"[Dinomaly] [模型释放] 调用父类释放方法...")
        super().release()
        
        total_time = time.time() - release_start
        print(f"[Dinomaly] [模型释放] Dinomaly资源释放完成，总耗时: {total_time:.3f}s")


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
