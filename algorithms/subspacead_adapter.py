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

from backend.core import BaseDetector, DetectionResult, register_algorithm


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
        # 与 start_server.py 中 HF_HOME 保持一致：models/pre_trained/huggingface
        cache_dir = os.path.join(project_root, 'models', 'pre_trained', 'huggingface')
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[SubspaceAD] [2/3] Cache directory ready: {cache_dir}")

        # 初始化特征提取器
        try:
            # 从本地路径加载模型，绕开 huggingface_hub 缓存系统
            # 模型文件在 models/pre_trained/dinov2/{backbone_name}/
            model_local_dir = os.path.join(project_root, 'models', 'pre_trained', 'dinov2', self.backbone)
            if not os.path.isdir(model_local_dir):
                print(f"[SubspaceAD] [3/3] Local model path not found: {model_local_dir}")
                print(f"[SubspaceAD] [3/3] Models are expected at: {os.path.join(project_root, 'models', 'pre_trained', 'dinov2/')}")
                raise FileNotFoundError(f"Local model directory not found: {model_local_dir}")

            print(f"[SubspaceAD] [3/3] Loading model from: {model_local_dir}")

            extractor_start = time.time()
            self._extractor = FeatureExtractor(model_local_dir, cache_dir=cache_dir)
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

    def _compute_image_score(self, tokens: np.ndarray, pca_params: dict) -> float:
        """从 tokens 和 PCA 参数计算单张图像的原始异常分数"""
        from src.subspacead.post_process.scoring import calculate_anomaly_scores, post_process_map

        b, h, w, c = tokens.shape
        tokens_flat = tokens.reshape(b * h * w, c)

        scores = calculate_anomaly_scores(tokens_flat, pca_params, self.score_method, self.drop_k)
        anomaly_map = scores.reshape(h, w)
        anomaly_map_resized = post_process_map(anomaly_map, self.img_size, blur=True)

        if self.img_score_agg == 'max':
            return float(np.max(anomaly_map_resized))
        elif self.img_score_agg == 'p99':
            return float(np.percentile(anomaly_map_resized, 99))
        elif self.img_score_agg == 'mtop5':
            return float(np.mean(np.sort(anomaly_map_resized.flatten())[-5:]))
        elif self.img_score_agg == 'mtop1p':
            flat = anomaly_map_resized.ravel()
            k = max(1, int(len(flat) * 0.01))
            idx = np.argpartition(flat, -k)[-k:]
            return float(np.mean(flat[idx]))
        else:
            return float(np.mean(anomaly_map_resized))

    def _fit_pca(self, image_paths: List[str]) -> None:
        """
        拟合 PCA 并校准归一化参数。

        流程：
        1. 提取所有参考图的 tokens（每图独立保存）
        2. LOO 校准：对每张图，用其余 N-1 张图的 tokens 拟合 PCA，
           然后计算这张图的异常分数（真正的"未见样本"误差）
        3. 用全部参考图拟合最终 PCA（用于测试图像评分）
        """
        from src.subspacead.core.pca import PCAModel

        print(f"[SubspaceAD] Fitting PCA on {len(image_paths)} reference images...")

        # 步骤 1：提取 tokens（每张参考图独立保留）
        per_image_tokens = []
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
            per_image_tokens.append(tokens)

        self._patch_h, self._patch_w = per_image_tokens[0].shape[1], per_image_tokens[0].shape[2]

        # 步骤 2：留一法（LOO）校准
        self._calibrate_normal_scores(per_image_tokens)

        # 步骤 3：用全部 tokens 拟合最终 PCA
        all_tokens = np.concatenate(per_image_tokens, axis=0)
        b, h, w_feat, c = all_tokens.shape
        all_tokens_flat = all_tokens.reshape(b * h * w_feat, c)

        print(f"[SubspaceAD] Final PCA on all {b} images ({all_tokens_flat.shape[0]} tokens)...")
        self._pca_model = PCAModel(k=self.pca_dim, ev=self.pca_ev, whiten=False)

        def feature_generator():
            yield all_tokens_flat

        self._pca_params = self._pca_model.fit(
            feature_generator,
            feature_dim=c,
            total_tokens=all_tokens_flat.shape[0],
            num_batches=1
        )

        print(f"[SubspaceAD] Final PCA fitted with {self._pca_params['k']} components")

    def _calibrate_normal_scores(self, per_image_tokens: list) -> None:
        """
        留一法交叉验证（LOO）校准分数。

        对 N 张参考图的每张图 i：
          用其他 N-1 张拟合 PCA → 对第 i 张图计算异常分数
        这些是"未经 PCA 拟合的正常图"的真实误差估计。

        N<3 时回退到带启发式修正的原始校准。
        """
        from src.subspacead.core.pca import PCAModel

        N = len(per_image_tokens)
        print(f"[SubspaceAD] LOO calibration on {N} reference images...")

        if N >= 3:
            loo_scores = []
            for i in range(N):
                # 训练 tokens：除第 i 张外的所有图
                train_np = np.concatenate(
                    [t for j, t in enumerate(per_image_tokens) if j != i], axis=0
                )
                b, h, w_f, c = train_np.shape
                train_flat = train_np.reshape(b * h * w_f, c)

                # 拟合 PCA（用 N-1 张图）
                pca_model = PCAModel(k=self.pca_dim, ev=self.pca_ev, whiten=False)
                def gen(): yield train_flat
                loo_params = pca_model.fit(gen, c, train_flat.shape[0], 1)

                # 对第 i 张图评分（真正的"未见样本"误差）
                score = self._compute_image_score(per_image_tokens[i], loo_params)
                loo_scores.append(score)
                print(f"[SubspaceAD]   LOO[{i+1}/{N}]: score={score:.4f} "
                      f"(PCA on {b} images, {loo_params['k']} components)")

            normal_scores = np.array(loo_scores)
            print(f"[SubspaceAD] LOO scores: {[f'{s:.4f}' for s in loo_scores]}")
        else:
            # N < 3：用参考图自身的 PCA 重建误差作为校准基线
            # 避免使用任意因子估算，改用实际重建误差计算
            print(f"[SubspaceAD] N={N} < 3, using self-reconstruction calibration")

            if N >= 2:
                # N == 2：2 折交叉验证——每张图用另一张训练的 PCA 评分
                loo_scores = []
                for i in range(N):
                    j = 1 - i
                    train_np = per_image_tokens[j]
                    b, h, w_f, c = train_np.shape
                    train_flat = train_np.reshape(b * h * w_f, c)

                    pca_model = PCAModel(k=self.pca_dim, ev=self.pca_ev, whiten=False)
                    def _gen2(): yield train_flat
                    loo_params = pca_model.fit(_gen2, c, train_flat.shape[0], 1)
                    score = self._compute_image_score(per_image_tokens[i], loo_params)
                    loo_scores.append(score)

                normal_scores = np.array(loo_scores)
                print(f"[SubspaceAD] 2-fold LOO scores: {[f'{s:.4f}' for s in loo_scores]}")
            else:
                # N == 1：在单张图上拟合 PCA，评分自身作为基线
                # 这给出"图内补丁变异"的重建误差量级
                tokens = per_image_tokens[0]
                b, h, w_f, c = tokens.shape
                tokens_flat = tokens.reshape(b * h * w_f, c)

                pca_model = PCAModel(k=self.pca_dim, ev=self.pca_ev, whiten=False)
                def _gen1(): yield tokens_flat
                pca_params = pca_model.fit(_gen1, c, tokens_flat.shape[0], 1)
                score = self._compute_image_score(tokens, pca_params)
                normal_scores = np.array([score])
                print(f"[SubspaceAD] Self-reconstruction score: {score:.4f}")

            # 使用参考分数计算校准参数
            self._normal_score_mean = float(np.mean(normal_scores))
            self._normal_score_std = float(np.std(normal_scores))

            # 对于单样本，std 可能为 0，使用保守的默认值
            if self._normal_score_std < 1e-6:
                self._normal_score_std = max(self._normal_score_mean * 2.0, 0.1)
                print(f"[SubspaceAD] Low std, using {self._normal_score_std:.4f}")

            # N < 3 时放宽边界（因为校准样本太少）
            if N == 2:
                widen = 2.0
            else:
                widen = 3.0  # N==1 时更宽松
            self._score_lower_bound = max(0, self._normal_score_mean - widen * self._normal_score_std)
            self._score_upper_bound = self._normal_score_mean + 2 * widen * self._normal_score_std

            if self._score_upper_bound <= self._score_lower_bound:
                self._score_upper_bound = self._score_lower_bound + self._normal_score_mean * 0.5 + 0.1

            print(f"[SubspaceAD] N<3 calibration: mean={self._normal_score_mean:.4f}, std={self._normal_score_std:.4f}")
            print(f"[SubspaceAD] Bounds: [{self._score_lower_bound:.4f}, {self._score_upper_bound:.4f}]")
            print(f"[SubspaceAD] Mapping: lower→0.3, mean→0.5, upper→0.7, above→0.7-1.0")
            return

        # 从 LOO 分数计算归一化参数
        self._normal_score_min = float(np.min(normal_scores))
        self._normal_score_max = float(np.max(normal_scores))
        self._normal_score_mean = float(np.mean(normal_scores))
        self._normal_score_std = float(np.std(normal_scores))

        if self._normal_score_std < 1e-6:
            self._normal_score_std = self._normal_score_mean * 0.5 if self._normal_score_mean > 0 else 0.01
            print(f"[SubspaceAD] Low std detected, using 0.5*mean as std={self._normal_score_std:.4f}")

        self._score_upper_bound = self._normal_score_mean + 4 * self._normal_score_std
        self._score_lower_bound = max(0, self._normal_score_mean - 2 * self._normal_score_std)

        if self._score_upper_bound <= self._score_lower_bound:
            self._score_upper_bound = self._score_lower_bound + self._normal_score_mean * 0.5

        print(f"[SubspaceAD] LOO calibration: mean={self._normal_score_mean:.4f}, "
              f"std={self._normal_score_std:.4f}")
        print(f"[SubspaceAD] Normalization bounds: [{self._score_lower_bound:.4f}, {self._score_upper_bound:.4f}]")
        print(f"[SubspaceAD] Mapping: lower→0.3, mean→0.5, upper→0.7, above→0.7-1.0")

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
            return min(1.0, max(0.0, raw_score / 100.0))

        mean = self._normal_score_mean
        lower = self._score_lower_bound
        upper = self._score_upper_bound

        if raw_score <= lower:
            if lower > 0:
                normalized = 0.3 * (raw_score / lower)
            else:
                normalized = 0.0
            print(f"[SubspaceAD] Normalize: range=below_lower, raw={raw_score:.4f}, lower={lower:.4f}, result={normalized:.4f}")
        elif raw_score <= mean:
            if mean > lower:
                normalized = 0.3 + 0.2 * (raw_score - lower) / (mean - lower)
            else:
                normalized = 0.4
            print(f"[SubspaceAD] Normalize: range=lower_to_mean, raw={raw_score:.4f}, mean={mean:.4f}, result={normalized:.4f}")
        elif raw_score <= upper:
            if upper > mean:
                normalized = 0.5 + 0.2 * (raw_score - mean) / (upper - mean)
            else:
                normalized = 0.6
            print(f"[SubspaceAD] Normalize: range=mean_to_upper, raw={raw_score:.4f}, upper={upper:.4f}, result={normalized:.4f}")
        else:
            excess = raw_score - upper
            scale = self._normal_score_std if self._normal_score_std > 0 else mean * 0.1
            ratio = excess / (3 * scale)
            normalized = 0.7 + 0.3 * min(1.0, ratio)
            print(f"[SubspaceAD] Normalize: range=above_upper, raw={raw_score:.4f}, excess={excess:.4f}, scale={scale:.4f}, ratio={ratio:.4f}, result={normalized:.4f}")

        return float(min(1.0, max(0.0, normalized)))

    def _calculate_anomaly_score(self, image_path: str) -> tuple:
        """计算单张图像的异常分数"""
        from src.subspacead.post_process.scoring import post_process_map

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

        # 计算原始分数 + 后处理热力图
        raw_score = self._compute_image_score(tokens, self._pca_params)

        # 生成热力图（后处理上采样）
        b, h, w_f, c = tokens.shape
        tokens_flat = tokens.reshape(b * h * w_f, c)
        from src.subspacead.post_process.scoring import calculate_anomaly_scores
        scores = calculate_anomaly_scores(tokens_flat, self._pca_params, self.score_method, self.drop_k)
        anomaly_map = scores.reshape(h, w_f)
        anomaly_map_resized = post_process_map(anomaly_map, self.img_size, blur=True)

        # 归一化分数到 [0, 1] 范围
        normalized_score = self._normalize_score(raw_score)

        # 调试输出
        print(f"[SubspaceAD] Image: raw_score={raw_score:.4f}, normalized={normalized_score:.4f}")
        if hasattr(self, '_normal_score_mean'):
            print(f"[SubspaceAD] Calibration: mean={self._normal_score_mean:.4f}")

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
