"""
SubspaceAD 算法适配器 - 少样本异常检测
基于 PCA 子空间建模，无需训练，仅需少量正常样本
CVPR 2026
"""

import os
import sys
import time
import numpy as np
from typing import Optional, List, Dict
from PIL import Image
import cv2

from core import BaseDetector, DetectionResult, register_algorithm


class SubspaceADBaseAdapter(BaseDetector):
    """SubspaceAD 基础适配器 - PCA 子空间异常检测"""
    
    def __init__(self, model_path: str, device: str = 'auto', threshold: float = 0.5,
                 backbone: str = 'dinov2-with-registers-large', img_size: int = 672, **kwargs):
        super().__init__(model_path, device, threshold, **kwargs)
        self.backbone = backbone
        self.img_size = img_size
        self.config = kwargs
        
        # SubspaceAD 特定参数
        # 根据 backbone 类型选择合适的层数
        if 'small' in backbone.lower():
            # DINOv2 Small 只有 12 层，使用最后 4 层
            default_layers = [-9, -10, -11, -12]
        elif 'base' in backbone.lower():
            # DINOv2 Base 有 12 层，使用最后 6 层
            default_layers = [-7, -8, -9, -10, -11, -12]
        else:
            # DINOv2 Large 有 24 层，使用最后 7 层 (默认)
            default_layers = [-12, -13, -14, -15, -16, -17, -18]
        self.layers = self.config.get('layers', default_layers)
        self.agg_method = self.config.get('agg_method', 'mean')
        self.pca_ev = self.config.get('pca_ev', 0.99)
        self.pca_dim = self.config.get('pca_dim', None)
        self.score_method = self.config.get('score_method', 'reconstruction')
        self.drop_k = self.config.get('drop_k', 0)
        self.img_score_agg = self.config.get('img_score_agg', 'mtop1p')
        self.use_clahe = self.config.get('use_clahe', False)
        self.k_shot = self.config.get('k_shot', 1)
        self.aug_count = self.config.get('aug_count', 30)
        
        # 内部组件
        self._extractor = None
        self._pca_model = None
        self._pca_params = None
        
    def load_model(self) -> None:
        """加载 SubspaceAD 模型组件"""
        import time
        start_time = time.time()
        
        print(f"[SubspaceAD] {'='*60}")
        print(f"[SubspaceAD] [MODEL LOAD START] SubspaceAD Few-Shot Detector")
        print(f"[SubspaceAD] {'='*60}")
        print(f"[SubspaceAD] Configuration:")
        print(f"[SubspaceAD]   - Backbone: {self.backbone}")
        print(f"[SubspaceAD]   - Image size: {self.img_size}x{self.img_size}")
        print(f"[SubspaceAD]   - Feature layers: {self.layers}")
        print(f"[SubspaceAD]   - Aggregation method: {self.agg_method}")
        print(f"[SubspaceAD]   - PCA variance ratio: {self.pca_ev}")
        print(f"[SubspaceAD]   - Score method: {self.score_method}")
        print(f"[SubspaceAD]   - Device: {self.device}")
        
        # 添加 SubspaceAD 目录到路径
        subspacead_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SubspaceAD')
        if subspacead_dir not in sys.path:
            sys.path.insert(0, subspacead_dir)
            print(f"[SubspaceAD] Added to sys.path: {subspacead_dir}")
        
        # 导入模块
        try:
            print(f"[SubspaceAD] [1/3] Importing SubspaceAD modules...")
            from src.subspacead.core.extractor import FeatureExtractor
            from src.subspacead.core.pca import PCAModel
            print(f"[SubspaceAD] ✓ Modules imported successfully")
        except Exception as e:
            print(f"[SubspaceAD] ✗ Failed to import modules: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 获取项目根目录和缓存目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, 'pre_trained')
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[SubspaceAD] [2/3] Cache directory ready: {cache_dir}")
        
        # 初始化特征提取器
        try:
            model_ckpt = f"facebook/{self.backbone}"
            print(f"[SubspaceAD] [3/3] Initializing feature extractor...")
            print(f"[SubspaceAD]   - Model checkpoint: {model_ckpt}")
            print(f"[SubspaceAD]   - Loading DINOv2 from HuggingFace...")
            
            extractor_start = time.time()
            self._extractor = FeatureExtractor(model_ckpt, cache_dir=cache_dir)
            extractor_time = time.time() - extractor_start
            
            print(f"[SubspaceAD] ✓ Feature extractor initialized in {extractor_time:.2f}s")
        except Exception as e:
            print(f"[SubspaceAD] ✗ Failed to initialize feature extractor: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        self.is_loaded = True
        total_time = time.time() - start_time
        print(f"[SubspaceAD] {'='*60}")
        print(f"[SubspaceAD] [MODEL LOAD COMPLETE] Total time: {total_time:.2f}s")
        print(f"[SubspaceAD] {'='*60}")
    
    def _fit_pca(self, image_paths: List[str]) -> None:
        """在参考图像上拟合 PCA 模型"""
        from src.subspacead.core.pca import PCAModel
        
        print(f"[SubspaceAD] Fitting PCA on {len(image_paths)} reference images...")
        
        # 加载并处理参考图像
        all_tokens = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_resized = img.resize((self.img_size, self.img_size), Image.BICUBIC)
            
            tokens, (h_p, w_p), _ = self._extractor.extract_tokens(
                [img_resized],
                self.img_size,
                self.layers,
                self.agg_method,
                [],
                docrop=False,
                use_clahe=self.use_clahe,
                dino_saliency_layer=6,
            )
            all_tokens.append(tokens)
        
        # 合并所有 tokens
        all_tokens = np.concatenate(all_tokens, axis=0)
        b, h, w, c = all_tokens.shape
        tokens_flat = all_tokens.reshape(b * h * w, c)
        
        print(f"[SubspaceAD] Total tokens for PCA: {tokens_flat.shape}")
        
        # 拟合 PCA
        self._pca_model = PCAModel(k=self.pca_dim, ev=self.pca_ev, whiten=False)
        
        # 创建特征生成器
        def feature_generator():
            yield tokens_flat
        
        self._pca_params = self._pca_model.fit(
            feature_generator,
            feature_dim=c,
            total_tokens=tokens_flat.shape[0],
            num_batches=1
        )
        
        print(f"[SubspaceAD] PCA fitted with {self._pca_params['k']} components")
        
        # 计算参考样本的异常分数用于归一化
        self._calibrate_normal_scores(image_paths)
    
    def _calibrate_normal_scores(self, image_paths: List[str]) -> None:
        """
        基于参考样本计算正常分数范围，用于后续归一化。
        使用参考样本的最大分数作为"正常上限"，将分数映射到 [0, 1] 范围。
        """
        from src.subspacead.post_process.scoring import calculate_anomaly_scores
        
        print(f"[SubspaceAD] Calibrating normal scores from {len(image_paths)} reference images...")
        
        normal_scores = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_resized = img.resize((self.img_size, self.img_size), Image.BICUBIC)
            
            tokens, (h_p, w_p), _ = self._extractor.extract_tokens(
                [img_resized],
                self.img_size,
                self.layers,
                self.agg_method,
                [],
                docrop=False,
                use_clahe=self.use_clahe,
                dino_saliency_layer=6,
            )
            
            b, h, w, c = tokens.shape
            tokens_flat = tokens.reshape(b * h * w, c)
            
            scores = calculate_anomaly_scores(
                tokens_flat,
                self._pca_params,
                self.score_method,
                self.drop_k,
            )
            anomaly_map = scores.reshape(h, w)
            
            # 后处理：上采样到原始尺寸（与预测阶段保持一致）
            from src.subspacead.post_process.scoring import post_process_map
            anomaly_map_resized = post_process_map(anomaly_map, self.img_size, blur=True)
            
            # 计算图像级分数（使用与预测相同的聚合方法）
            if self.img_score_agg == 'max':
                img_score = float(np.max(anomaly_map_resized))
            elif self.img_score_agg == 'p99':
                img_score = float(np.percentile(anomaly_map_resized, 99))
            elif self.img_score_agg == 'mtop5':
                img_score = float(np.mean(np.sort(anomaly_map_resized.flatten())[-5:]))
            elif self.img_score_agg == 'mtop1p':
                flat = anomaly_map_resized.ravel()
                k = max(1, int(len(flat) * 0.01))
                idx = np.argpartition(flat, -k)[-k:]
                img_score = float(np.mean(flat[idx]))
            else:
                img_score = float(np.mean(anomaly_map_resized))
            
            normal_scores.append(img_score)
        
        # 计算归一化参数
        normal_scores = np.array(normal_scores)
        self._normal_score_min = float(np.min(normal_scores))
        self._normal_score_max = float(np.max(normal_scores))
        self._normal_score_mean = float(np.mean(normal_scores))
        self._normal_score_std = float(np.std(normal_scores))
        
        # 处理只有一个参考样本的情况（std=0）
        # 使用基于均值的启发式标准差估计
        if self._normal_score_std < 1e-6:
            # 当标准差为0时，使用均值的10%作为估计标准差
            self._normal_score_std = self._normal_score_mean * 0.1
            print(f"[SubspaceAD] Single reference detected, using estimated std={self._normal_score_std:.4f}")
        
        # 使用均值 + 4*标准差作为"正常上限"
        # 这样可以确保正常样本分数在 0.5 左右，异常样本会明显高于这个值
        self._score_upper_bound = self._normal_score_mean + 4 * self._normal_score_std
        self._score_lower_bound = max(0, self._normal_score_mean - 2 * self._normal_score_std)
        
        # 确保 upper_bound > lower_bound 且有合理的差距
        if self._score_upper_bound <= self._score_lower_bound:
            self._score_upper_bound = self._score_lower_bound + self._normal_score_mean * 0.5
        
        print(f"[SubspaceAD] Normal score range: [{self._normal_score_min:.4f}, {self._normal_score_max:.4f}]")
        print(f"[SubspaceAD] Normal score stats: mean={self._normal_score_mean:.4f}, std={self._normal_score_std:.4f}")
        print(f"[SubspaceAD] Score bounds for normalization: [{self._score_lower_bound:.4f}, {self._score_upper_bound:.4f}]")
    
    def _normalize_score(self, raw_score: float) -> float:
        """
        将原始异常分数归一化到 [0, 1] 范围。
        映射策略：
        - lower_bound (参考样本正常范围下限) -> 0.3
        - mean (参考样本均值) -> 0.5
        - upper_bound (参考样本正常范围上限) -> 0.7
        - 超过 upper_bound 的分数 -> 0.7-1.0 (线性映射)
        """
        if not hasattr(self, '_score_upper_bound'):
            # 如果没有校准数据，使用简单的 min-max 归一化
            return min(1.0, max(0.0, raw_score / 100.0))
        
        mean = self._normal_score_mean
        lower = self._score_lower_bound
        upper = self._score_upper_bound
        
        # 分段映射
        if raw_score <= lower:
            # 低于下限 -> 0.0-0.3
            if lower > 0:
                normalized = 0.3 * (raw_score / lower)
            else:
                normalized = 0.0
            print(f"[SubspaceAD] Normalize: range=below_lower, raw={raw_score:.4f}, lower={lower:.4f}, result={normalized:.4f}")
        elif raw_score <= mean:
            # 下限到均值 -> 0.3-0.5
            if mean > lower:
                normalized = 0.3 + 0.2 * (raw_score - lower) / (mean - lower)
            else:
                normalized = 0.4
            print(f"[SubspaceAD] Normalize: range=lower_to_mean, raw={raw_score:.4f}, mean={mean:.4f}, result={normalized:.4f}")
        elif raw_score <= upper:
            # 均值到上限 -> 0.5-0.7
            if upper > mean:
                normalized = 0.5 + 0.2 * (raw_score - mean) / (upper - mean)
            else:
                normalized = 0.6
            print(f"[SubspaceAD] Normalize: range=mean_to_upper, raw={raw_score:.4f}, upper={upper:.4f}, result={normalized:.4f}")
        else:
            # 超过上限 -> 0.7-1.0 (使用对数压缩避免过度饱和)
            excess = raw_score - upper
            # 使用参考样本标准差作为缩放因子
            scale = self._normal_score_std if self._normal_score_std > 0 else mean * 0.1
            # 每增加一个标准差，分数增加约0.1
            ratio = excess / (3 * scale)
            normalized = 0.7 + 0.3 * min(1.0, ratio)
            print(f"[SubspaceAD] Normalize: range=above_upper, raw={raw_score:.4f}, excess={excess:.4f}, scale={scale:.4f}, ratio={ratio:.4f}, result={normalized:.4f}")
        
        # 确保在 [0, 1] 范围内
        return float(min(1.0, max(0.0, normalized)))
    
    def _calculate_anomaly_score(self, image_path: str) -> tuple:
        """计算单张图像的异常分数"""
        from src.subspacead.post_process.scoring import calculate_anomaly_scores, post_process_map
        
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        
        # 提取特征
        tokens, (h_p, w_p), _ = self._extractor.extract_tokens(
            [img_resized],
            self.img_size,
            self.layers,
            self.agg_method,
            [],
            docrop=False,
            use_clahe=self.use_clahe,
            dino_saliency_layer=6,
        )
        
        b, h, w, c = tokens.shape
        tokens_flat = tokens.reshape(b * h * w, c)
        
        # 计算异常分数
        scores = calculate_anomaly_scores(
            tokens_flat,
            self._pca_params,
            self.score_method,
            self.drop_k,
        )
        anomaly_map = scores.reshape(h, w)
        
        # 后处理：上采样到原始尺寸
        anomaly_map_resized = post_process_map(anomaly_map, self.img_size, blur=True)
        
        # 计算图像级原始分数
        if self.img_score_agg == 'max':
            raw_score = float(np.max(anomaly_map_resized))
        elif self.img_score_agg == 'p99':
            raw_score = float(np.percentile(anomaly_map_resized, 99))
        elif self.img_score_agg == 'mtop5':
            raw_score = float(np.mean(np.sort(anomaly_map_resized.flatten())[-5:]))
        elif self.img_score_agg == 'mtop1p':
            flat = anomaly_map_resized.ravel()
            k = max(1, int(len(flat) * 0.01))
            idx = np.argpartition(flat, -k)[-k:]
            raw_score = float(np.mean(flat[idx]))
        else:
            raw_score = float(np.mean(anomaly_map_resized))
        
        # 归一化分数到 [0, 1] 范围
        normalized_score = self._normalize_score(raw_score)
        
        # 调试输出
        print(f"[SubspaceAD] Image: raw_score={raw_score:.4f}, normalized={normalized_score:.4f}")
        if hasattr(self, '_normal_score_mean'):
            print(f"[SubspaceAD] Calibration: mean={self._normal_score_mean:.4f}, upper={self._score_upper_bound:.4f}")
        
        return normalized_score, anomaly_map_resized, img.size
    
    def predict(self, image_path: str, reference_paths: Optional[List[str]] = None) -> DetectionResult:
        """单张图像推理"""
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # 如果没有提供参考集，使用单图自身作为参考（退化情况）
            if reference_paths is None or len(reference_paths) < 1:
                print("[SubspaceAD] Warning: No reference images provided, using self-reference")
                reference_paths = [image_path]
            
            # 拟合 PCA（每次预测都重新拟合，因为是少样本场景）
            self._fit_pca(reference_paths)
            
            # 计算异常分数
            score, anomaly_map, orig_size = self._calculate_anomaly_score(image_path)
            
            # 判定是否异常
            is_anomaly = score > self.threshold
            
            inference_time = (time.time() - start_time) * 1000
            
            return DetectionResult(
                is_anomaly=bool(is_anomaly),
                anomaly_score=float(score),
                anomaly_map=anomaly_map,
                inference_time=inference_time,
                metadata={
                    'backbone': self.backbone,
                    'image_size': self.img_size,
                    'method': 'subspacead-pca',
                    'pca_components': self._pca_params['k'],
                    'reference_count': len(reference_paths)
                }
            )
            
        except Exception as e:
            print(f"[ERROR] SubspaceAD prediction failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict_batch(self, image_paths: List[str], reference_paths: Optional[List[str]] = None) -> List[DetectionResult]:
        """批量推理"""
        if not self.is_loaded:
            self.load_model()
        
        if len(image_paths) == 0:
            return []
        
        # 如果没有提供参考集，使用第一批图像作为参考集（少样本场景）
        if reference_paths is None:
            # 默认使用 k_shot 个样本作为参考集
            k = min(self.k_shot, len(image_paths))
            reference_paths = image_paths[:k]
            test_paths = image_paths
        else:
            test_paths = image_paths
        
        print(f"[SubspaceAD] Batch prediction: {len(test_paths)} test images, {len(reference_paths)} reference images")
        
        start_time = time.time()
        
        try:
            # 在参考集上拟合 PCA
            self._fit_pca(reference_paths)
            
            # 处理所有测试图像
            results = []
            for i, path in enumerate(test_paths):
                score, anomaly_map, orig_size = self._calculate_anomaly_score(path)
                is_anomaly = score > self.threshold
                
                results.append(DetectionResult(
                    is_anomaly=bool(is_anomaly),
                    anomaly_score=float(score),
                    anomaly_map=anomaly_map,
                    inference_time=0,  # 稍后统一计算
                    metadata={
                        'backbone': self.backbone,
                        'image_size': self.img_size,
                        'method': 'subspacead-pca',
                        'pca_components': self._pca_params['k'],
                        'reference_count': len(reference_paths)
                    }
                ))
            
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(test_paths)
            
            # 更新推理时间
            for r in results:
                r.inference_time = avg_time
            
            return results
            
        except Exception as e:
            print(f"[ERROR] SubspaceAD batch prediction failed: {e}")
            import traceback
            traceback.print_exc()
            # 回退到单张推理
            return [self.predict(path, reference_paths) for path in test_paths]
    
    def release(self) -> None:
        """释放资源"""
        self._extractor = None
        self._pca_model = None
        self._pca_params = None
        super().release()


# 注册不同 backbone 的 SubspaceAD 变体
@register_algorithm("subspacead_dinov2_large_672")
class SubspaceADDINOv2_Large_672(SubspaceADBaseAdapter):
    """SubspaceAD with DINOv2-Large@672px - 高精度（推荐）"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='dinov2-with-registers-large', img_size=672, **kwargs)


@register_algorithm("subspacead_dinov2_large_518")
class SubspaceADDINOv2_Large_518(SubspaceADBaseAdapter):
    """SubspaceAD with DINOv2-Large@518px - 高精度"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='dinov2-with-registers-large', img_size=518, **kwargs)


@register_algorithm("subspacead_dinov2_large_336")
class SubspaceADDINOv2_Large_336(SubspaceADBaseAdapter):
    """SubspaceAD with DINOv2-Large@336px - 均衡"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='dinov2-with-registers-large', img_size=336, **kwargs)


@register_algorithm("subspacead_dinov2_base_672")
class SubspaceADDINOv2_Base_672(SubspaceADBaseAdapter):
    """SubspaceAD with DINOv2-Base@672px - 轻量级"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='dinov2-with-registers-base', img_size=672, **kwargs)


@register_algorithm("subspacead_dinov2_base_518")
class SubspaceADDINOv2_Base_518(SubspaceADBaseAdapter):
    """SubspaceAD with DINOv2-Base@518px - 轻量均衡"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='dinov2-with-registers-base', img_size=518, **kwargs)


@register_algorithm("subspacead_dinov2_small_672")
class SubspaceADDINOv2_Small_672(SubspaceADBaseAdapter):
    """SubspaceAD with DINOv2-Small@672px - 快速"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='dinov2-with-registers-small', img_size=672, **kwargs)


def create_subspacead_detector(algorithm_name: str, model_path: str, **kwargs):
    """创建 SubspaceAD 检测器的工厂函数"""
    from algorithms import create_detector
    return create_detector(algorithm_name, model_path, **kwargs)
