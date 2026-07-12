"""
ADer框架算法适配器 (v3)

支持模型:
- Benchmark 图像模型: cflow, pyramidflow, simplenet, destseg, realnet, rdpp,
  mambaad, invad, vitad, uniad (全部支持图片直接推理)
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from typing import Optional

# 添加 algorithms 和 ADer 目录到路径
# ADer 内部代码使用平铺导入 (from util import ...)，需要 ADer 根目录在 sys.path 中
_algorithms_dir = os.path.dirname(os.path.abspath(__file__))
_ader_dir = os.path.join(_algorithms_dir, 'ADer')
if _algorithms_dir not in sys.path:
    sys.path.insert(0, _algorithms_dir)
if _ader_dir not in sys.path:
    sys.path.insert(0, _ader_dir)

from backend.core import BaseDetector, DetectionResult, register_algorithm


def _import_ader():
    """延迟导入 ADer 模块，确保 CWD 正确

    ADer 内部的 configs/__base__/__init__.py 和 trainer/__init__.py
    使用 glob.glob('configs/__base__/...') 等基于 CWD 的路径，
    必须在 CWD 为 ADer 目录时才能正确导入。
    """
    _prev = os.getcwd()
    try:
        os.chdir(_ader_dir)
        from ADer import ADerTaskAssigner
        return ADerTaskAssigner
    finally:
        os.chdir(_prev)


class ADerBaseAdapter(BaseDetector):
    """ADer框架基础适配器 (v3)

    支持两种模式:
    1. 基准模型 (Benchmark): 直接加载模型进行图片推理
    2. 音频模型: 仅支持频谱图推理 (已弃用)

    对于基准模型，load_model() 会:
    1. 构建 ADer 配置
    2. 创建模型架构
    3. 加载预训练权重（如果存在）
    """

    def __init__(self, model_path: str, method: str, device: str = 'auto',
                 threshold: float = 0.5, config_path: Optional[str] = None, **kwargs):
        super().__init__(model_path, device, threshold, **kwargs)
        self.method = method
        self.config_path = config_path
        self._assigner = None
        self._net = None
        self._cfg = None
        self._transform = None
        self._image_size = 256

    def load_model(self) -> None:
        """加载ADer模型"""
        ADerTaskAssigner = _import_ader()

        start_time = time.time()
        print(f"[ADer:{self.method}] Loading model...")

        self._assigner = ADerTaskAssigner(method=self.method)

        # 对于 Benchmark 图像模型，尝试加载实际模型权重
        try:
            self._load_benchmark_model()
        except Exception as e:
            print(f"[ADer:{self.method}] Benchmark model load failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"[ADer:{self.method}] Falling back to assigner-only mode")

        self.is_loaded = True
        elapsed = time.time() - start_time
        model_status = "full model" if self._net is not None else "assigner only"
        print(f"[ADer:{self.method}] Load complete in {elapsed:.2f}s ({model_status})")

    def _load_benchmark_model(self):
        """尝试加载 Benchmark 图像模型的权重

        ADer 配置系统依赖当前工作目录定位 configs/__base__/，
        需要临时切换到 ADer 目录。
        """
        import argparse
        from ADer.configs import get_cfg
        from ADer.util.net import init_training
        from ADer.util.util import run_pre
        from model import get_model as ader_get_model

        # 设置 HF 离线模式，避免网络不可达时下载卡死
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _hf_cache = os.path.join(_project_root, "models", "pre_trained", "huggingface", "hub")
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', _hf_cache)
        os.environ.setdefault('TRANSFORMERS_CACHE', _hf_cache)
        os.environ.setdefault('HF_HOME', os.path.join(_project_root, "models", "pre_trained", "huggingface"))

        # 保存并切换工作目录
        _prev_cwd = os.getcwd()
        os.chdir(_ader_dir)

        try:
            # 解析配置
            parser = argparse.ArgumentParser()
            parser.add_argument('-c', '--cfg_path', default=self._assigner.cfg_path)
            parser.add_argument('-m', '--mode', default='test')
            parser.add_argument('--sleep', type=int, default=-1)
            parser.add_argument('--memory', type=int, default=-1)
            parser.add_argument('--dist_url', default='env://', type=str)
            parser.add_argument('--logger_rank', default=0, type=int)
            parser.add_argument('opts', nargs=argparse.REMAINDER)
            cfg_terminal = parser.parse_args([])

            cfg = get_cfg(cfg_terminal)
            run_pre(cfg)
            init_training(cfg)

        finally:
            os.chdir(_prev_cwd)

        self._cfg = cfg
        self._image_size = getattr(cfg, 'image_size', getattr(cfg, 'size', 256))

        # 创建模型
        print(f"[ADer:{self.method}] Creating model: {cfg.model.name}")
        self._net = ader_get_model(cfg.model)
        self._net.eval()

        # 加载权重（如果提供）
        if self.model_path and os.path.isfile(self.model_path):
            print(f"[ADer:{self.method}] Loading weights from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location='cpu', weights_only=True)
            if 'net' in state_dict:
                state_dict = state_dict['net']
            # 过滤不兼容的键
            model_dict = self._net.state_dict()
            compatible = {k: v for k, v in state_dict.items()
                         if k in model_dict and v.shape == model_dict[k].shape}
            self._net.load_state_dict(compatible, strict=False)
            print(f"[ADer:{self.method}] Loaded {len(compatible)}/{len(model_dict)} params")
        elif self.model_path:
            print(f"[ADer:{self.method}] Weights not found: {self.model_path}, using random init")
        else:
            print(f"[ADer:{self.method}] No weights path provided, using random init")

        self._net.to(self.device)

        # 准备预处理变换
        from torchvision import transforms
        self._transform = transforms.Compose([
            transforms.Resize((self._image_size, self._image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def _load_image(self, image_path: str) -> torch.Tensor:
        """加载并预处理图像"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self._transform(img).unsqueeze(0).to(self.device)
        return img_tensor

    def _compute_feature_anomaly_map(self, ft_list, fs_list, img_size,
                                     uni_am=False, use_cos=True,
                                     amap_mode='add', gaussian_sigma=4):
        """A组共享: 教师-学生特征比较 → 异常图

        Args:
            ft_list: 教师特征列表 (list of Tensors)
            fs_list: 学生特征列表 (list of Tensors)
            img_size: 输出尺寸 [H, W]
            uni_am: 是否统一特征图尺寸后再比较
            use_cos: 使用余弦相似度 (False 则使用 MSE)
            amap_mode: 聚合模式 ('add' 或 'mul')
            gaussian_sigma: 高斯平滑 sigma

        Returns:
            anomaly_map: numpy array [1, H, W] 或 [H, W]
        """
        from ADer.util.metric import Evaluator
        anomaly_map, _ = Evaluator.cal_anomaly_map(
            ft_list, fs_list,
            out_size=img_size,
            uni_am=uni_am,
            use_cos=use_cos,
            amap_mode=amap_mode,
            gaussian_sigma=gaussian_sigma,
        )
        return anomaly_map

    def predict(self, image_path: str) -> DetectionResult:
        """单张图像推理"""
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 如果成功加载了完整模型，使用直接推理
        if self._net is not None and self._transform is not None:
            return self._predict_direct(image_path, start_time)
        else:
            return self._predict_fallback(image_path, start_time)

    def _predict_direct(self, image_path: str, start_time: float) -> DetectionResult:
        """直接使用模型推理 — 子类可覆盖

        基类使用通用接口尝试推理。
        """
        img_tensor = self._load_image(image_path)

        try:
            with torch.no_grad():
                # 尝试多种推理接口
                if hasattr(self._net, 'predict'):
                    result = self._net.predict(img_tensor)
                    if isinstance(result, tuple) and len(result) == 2:
                        scores, preds = result
                        score = float(scores.mean().cpu())
                        anomaly_map = preds[0].squeeze().cpu().numpy() if preds is not None else None
                    elif isinstance(result, torch.Tensor):
                        score = float(result.max().cpu())
                        anomaly_map = result.squeeze().cpu().numpy()
                    else:
                        score = float(result) if not isinstance(result, (tuple, list)) else float(result[0])
                        anomaly_map = None
                elif hasattr(self._net, 'net_simplenet') and hasattr(self._net.net_simplenet, 'predict'):
                    scores, preds = self._net.net_simplenet.predict({'img': img_tensor})
                    score = float(scores.mean().cpu())
                    anomaly_map = preds[0].squeeze().cpu().numpy() if preds is not None else None
                else:
                    # 通用 forward 推理 — 尝试特征比较
                    output = self._net(img_tensor)
                    if isinstance(output, (tuple, list)) and len(output) >= 2:
                        ft_list, fs_list = output[0], output[1]
                        if isinstance(ft_list, (list, tuple)) and isinstance(fs_list, (list, tuple)):
                            anomaly_map = self._compute_feature_anomaly_map(
                                ft_list, fs_list,
                                img_size=[img_tensor.shape[2], img_tensor.shape[3]],
                            )
                            score = float(anomaly_map.max())
                            anomaly_map = anomaly_map[0] if anomaly_map.ndim == 3 else anomaly_map
                        else:
                            output_tensor = output[1]
                            score = float(output_tensor.max().cpu())
                            anomaly_map = output_tensor.squeeze().cpu().numpy()
                    elif isinstance(output, torch.Tensor):
                        score = float(output.max().cpu())
                        anomaly_map = output.squeeze().cpu().numpy()
                    elif hasattr(output, 'max'):
                        score = float(np.array(output).max())
                        anomaly_map = np.array(output).squeeze()
                    else:
                        score = 0.5
                        anomaly_map = None
        except Exception as e:
            print(f"[ADer:{self.method}] Direct inference failed: {e}, using fallback")
            return self._predict_fallback(image_path, start_time)

        inference_time = (time.time() - start_time) * 1000
        is_anomaly = score > self.threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=inference_time,
            metadata={'method': self.method, 'mode': 'direct'}
        )

    def _predict_fallback(self, image_path: str, start_time: float) -> DetectionResult:
        """回退推理: 通用前向传播"""
        import shutil
        import tempfile

        inference_dir = tempfile.mkdtemp(prefix='ader_inference_')

        _prev = os.getcwd()
        os.chdir(_ader_dir)

        try:
            dst_path = os.path.join(inference_dir, os.path.basename(image_path))
            shutil.copy(image_path, dst_path)

            self._assigner.inference(inference_dir=inference_dir)

            score = 0.5
            is_anomaly = score > self.threshold
        except Exception as e:
            print(f"[ADer:{self.method}] Fallback inference failed: {e}")
            score = 0.5
            is_anomaly = False
        finally:
            os.chdir(_prev)
            if os.path.exists(inference_dir):
                shutil.rmtree(inference_dir, ignore_errors=True)

        inference_time = (time.time() - start_time) * 1000

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=None,
            inference_time=inference_time,
            metadata={'method': self.method, 'mode': 'fallback'}
        )

    def release(self) -> None:
        """释放资源"""
        if self._net is not None:
            del self._net
            self._net = None
        self._assigner = None
        self._cfg = None
        self._transform = None
        super().release()


# ============================================================================
# A组: 教师-学生特征比较模型
# forward() → (feats_t, feats_s) → cal_anomaly_map()
# ============================================================================

class _FeatureComparisonMixin:
    """A组模型的共享 _predict_direct 实现

    适用于 forward() 返回 (teacher_features, student_features) 元组的模型。
    """

    def _predict_direct(self, image_path: str, start_time: float) -> DetectionResult:
        img_tensor = self._load_image(image_path)

        try:
            with torch.no_grad():
                output = self._net(img_tensor)
                if isinstance(output, (tuple, list)) and len(output) >= 2:
                    ft_list, fs_list = output[0], output[1]
                    anomaly_map = self._compute_feature_anomaly_map(
                        ft_list, fs_list,
                        img_size=[img_tensor.shape[2], img_tensor.shape[3]],
                    )
                    score = float(anomaly_map.max())
                    anomaly_map = anomaly_map[0] if anomaly_map.ndim == 3 else anomaly_map
                else:
                    raise RuntimeError(f"Unexpected output type: {type(output)}")
        except Exception as e:
            print(f"[ADer:{self.method}] Direct inference failed: {e}")
            return self._predict_fallback(image_path, start_time)

        inference_time = (time.time() - start_time) * 1000
        is_anomaly = score > self.threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=inference_time,
            metadata={'method': self.method, 'mode': 'feature_comparison'}
        )


# ============================================================================
# 注册 ADer 各算法
# ============================================================================

@register_algorithm("mambaad")
class MambaADAdapter(_FeatureComparisonMixin, ADerBaseAdapter):
    """MambaAD — 状态空间模型异常检测"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='MambaAD', **kwargs)


@register_algorithm("invad")
class InVadAdapter(ADerBaseAdapter):
    """InVad — 逆生成式异常检测

    InVad 使用 cfg.uni_am 和 cfg.use_cos 控制异常图计算，
    需要覆盖 _predict_direct 以使用配置参数。
    """
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='InVad', **kwargs)

    def _predict_direct(self, image_path: str, start_time: float) -> DetectionResult:
        img_tensor = self._load_image(image_path)

        try:
            with torch.no_grad():
                output = self._net(img_tensor)
                if isinstance(output, (tuple, list)) and len(output) >= 2:
                    ft_list, fs_list = output[0], output[1]
                    # InVad 使用 cfg 控制的参数
                    uni_am = getattr(self._cfg, 'uni_am', False) if self._cfg else False
                    use_cos = getattr(self._cfg, 'use_cos', True) if self._cfg else True
                    anomaly_map = self._compute_feature_anomaly_map(
                        ft_list, fs_list,
                        img_size=[img_tensor.shape[2], img_tensor.shape[3]],
                        uni_am=uni_am,
                        use_cos=use_cos,
                    )
                    score = float(anomaly_map.max())
                    anomaly_map = anomaly_map[0] if anomaly_map.ndim == 3 else anomaly_map
                else:
                    raise RuntimeError(f"Unexpected output type: {type(output)}")
        except Exception as e:
            print(f"[ADer:{self.method}] Direct inference failed: {e}")
            return self._predict_fallback(image_path, start_time)

        inference_time = (time.time() - start_time) * 1000
        is_anomaly = score > self.threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=inference_time,
            metadata={'method': self.method, 'mode': 'feature_comparison'}
        )


@register_algorithm("vitad")
class ViTADAdapter(_FeatureComparisonMixin, ADerBaseAdapter):
    """ViTAD — ViT Transformer异常检测"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='ViTAD', **kwargs)


@register_algorithm("unad")
class UniADAdapter(ADerBaseAdapter):
    """UniAD — 统一异常检测框架

    forward() 返回 (feats_t, feats_s, pred)，pred 直接作为异常图。
    """
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='UniAD', **kwargs)

    def _predict_direct(self, image_path: str, start_time: float) -> DetectionResult:
        img_tensor = self._load_image(image_path)

        try:
            with torch.no_grad():
                output = self._net(img_tensor)
                if isinstance(output, (tuple, list)) and len(output) >= 3:
                    # UniAD: (feats_t, feats_s, pred)
                    pred = output[2]
                    if isinstance(pred, torch.Tensor):
                        anomaly_map = pred.squeeze().cpu().numpy()
                        score = float(anomaly_map.max())
                    else:
                        raise RuntimeError(f"Unexpected pred type: {type(pred)}")
                else:
                    raise RuntimeError(f"Expected 3-tuple output, got {type(output)}")
        except Exception as e:
            print(f"[ADer:{self.method}] Direct inference failed: {e}")
            return self._predict_fallback(image_path, start_time)

        inference_time = (time.time() - start_time) * 1000
        is_anomaly = score > self.threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=inference_time,
            metadata={'method': self.method, 'mode': 'pred_direct'}
        )


@register_algorithm("cflow")
class CFlowAdapter(ADerBaseAdapter):
    """CFlow — 归一化流异常检测

    使用完整的流式推理管道: forward钩子 → Decoder → FIB → log-prob聚合 → 异常图。
    """
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='CFlow', **kwargs)

    def _predict_direct(self, image_path: str, start_time: float) -> DetectionResult:
        import torch.nn.functional as F
        img_tensor = self._load_image(image_path)

        try:
            with torch.no_grad():
                # Step 1: forward 触发钩子存储 pool_layers 特征
                self._net(img_tensor)

                # Step 2: 遍历池化层，进行流式推理
                test_dist = [[] for _ in self._net.pool_layers]
                height, width = [], []

                for l, layer in enumerate(self._net.pool_layers):
                    FIB, c_r, e_r, dec_idx, _, E, C, H, W = self._net.Decoder_forward(l, layer)
                    height.append(H)
                    width.append(W)

                    for f in range(FIB):
                        log_prob, _ = self._net.FIB_forward(
                            f, FIB, c_r, e_r, dec_idx,
                            self._net.N, E, C,
                            self._net.model_backbone.dec_arch
                        )
                        test_dist[l].extend(log_prob.detach().cpu().tolist())

                # Step 3: 后处理 — 概率 → 异常图
                test_map = [list() for _ in self._net.pool_layers]
                for l, _ in enumerate(self._net.pool_layers):
                    test_norm = torch.tensor(test_dist[l], dtype=torch.double)
                    test_norm -= torch.max(test_norm)
                    test_prob = torch.exp(test_norm)
                    test_mask = test_prob.reshape(-1, height[l], width[l])
                    test_map[l] = F.interpolate(
                        test_mask.unsqueeze(1),
                        size=self._image_size, mode='bilinear', align_corners=True
                    ).squeeze().numpy()

                # Step 4: 分数聚合 + 反转
                score_map = np.zeros_like(test_map[0])
                for l, _ in enumerate(self._net.pool_layers):
                    score_map += test_map[l]

                anomaly_map = score_map.max() - score_map
                score = float(anomaly_map.max())
        except Exception as e:
            print(f"[ADer:{self.method}] CFLow inference failed: {e}")
            import traceback
            traceback.print_exc()
            return self._predict_fallback(image_path, start_time)

        inference_time = (time.time() - start_time) * 1000
        is_anomaly = score > self.threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=inference_time,
            metadata={'method': self.method, 'mode': 'cflow'}
        )


@register_algorithm("pyramidflow")
class PyramidFlowAdapter(ADerBaseAdapter):
    """PyramidFlow — 金字塔归一化流异常检测

    使用 predict(imgs, template) 接口。如果无模板则使用零模板回退。
    """
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='PyramidFlow', **kwargs)
        self._template = None  # feat_mean from training

    def load_model(self) -> None:
        super().load_model()
        # 尝试从检查点加载模板
        self._load_template()

    def _load_template(self):
        """尝试从模型权重文件加载 PyramidFlow 模板"""
        if not self.model_path or not os.path.isfile(self.model_path):
            return

        try:
            ckpt = torch.load(self.model_path, map_location='cpu', weights_only=True)
            # 检查点可能包含 feat_mean
            if 'feat_mean' in ckpt:
                self._template = ckpt['feat_mean']
                print(f"[ADer:{self.method}] Loaded template from checkpoint")
            elif 'net' in ckpt and 'feat_mean' in ckpt['net']:
                self._template = [t.to(self.device) for t in ckpt['net']['feat_mean']]
                print(f"[ADer:{self.method}] Loaded template from checkpoint (net)")
        except Exception as e:
            print(f"[ADer:{self.method}] Could not load template: {e}")

    def _predict_direct(self, image_path: str, start_time: float) -> DetectionResult:
        img_tensor = self._load_image(image_path)

        try:
            with torch.no_grad():
                if hasattr(self._net, 'net_pyramidflow') and hasattr(self._net.net_pyramidflow, 'predict'):
                    if self._template is not None:
                        preds = self._net.net_pyramidflow.predict(img_tensor, self._template)
                    else:
                        # 无模板回退: 使用零模板
                        print(f"[ADer:{self.method}] No template available, using zero template")
                        pyramid2 = self._net.net_pyramidflow.pred_tempelate(img_tensor)
                        zero_template = [torch.zeros_like(p) for p in pyramid2]
                        preds = self._net.net_pyramidflow.predict(img_tensor, zero_template)

                    anomaly_map = preds.squeeze(dim=1).cpu().numpy()
                    score = float(anomaly_map.max())
                else:
                    # fallback: generic forward
                    output = self._net(img_tensor)
                    if isinstance(output, torch.Tensor):
                        score = float(output.max().cpu())
                        anomaly_map = output.squeeze().cpu().numpy()
                    else:
                        raise RuntimeError(f"Cannot infer from output type: {type(output)}")
        except Exception as e:
            print(f"[ADer:{self.method}] PyramidFlow inference failed: {e}")
            return self._predict_fallback(image_path, start_time)

        inference_time = (time.time() - start_time) * 1000
        is_anomaly = score > self.threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=inference_time,
            metadata={'method': self.method, 'mode': 'pyramidflow'}
        )


@register_algorithm("simplenet")
class SimpleNetAdapter(ADerBaseAdapter):
    """SimpleNet — 轻量级特征学习异常检测

    使用 net_simplenet.predict() 接口。
    """
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='SimpleNet', **kwargs)

    def _predict_direct(self, image_path: str, start_time: float) -> DetectionResult:
        img_tensor = self._load_image(image_path)

        try:
            with torch.no_grad():
                if hasattr(self._net, 'net_simplenet') and hasattr(self._net.net_simplenet, 'predict'):
                    scores, preds = self._net.net_simplenet.predict({'img': img_tensor})
                    score = float(np.array(scores).mean())
                    anomaly_map = preds[0].squeeze() if preds is not None else None
                elif hasattr(self._net, 'predict'):
                    result = self._net.predict(img_tensor)
                    if isinstance(result, tuple) and len(result) == 2:
                        scores, preds = result
                        score = float(scores.mean().cpu())
                        anomaly_map = preds[0].squeeze().cpu().numpy() if preds is not None else None
                    else:
                        score = float(result.max().cpu()) if isinstance(result, torch.Tensor) else float(result)
                        anomaly_map = result.squeeze().cpu().numpy() if isinstance(result, torch.Tensor) else None
                else:
                    raise RuntimeError("SimpleNet: no predict interface found")
        except Exception as e:
            print(f"[ADer:{self.method}] Direct inference failed: {e}")
            return self._predict_fallback(image_path, start_time)

        inference_time = (time.time() - start_time) * 1000
        is_anomaly = score > self.threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=inference_time,
            metadata={'method': self.method, 'mode': 'simplenet'}
        )


@register_algorithm("destseg")
class DeSTSegAdapter(ADerBaseAdapter):
    """DeSTSeg — 分割范式异常检测

    forward(ori_imgs) → (output_seg, output_de_st, output_de_st_list, new_mask)
    output_seg[:, 0] 为异常分割图。
    """
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='DeSTSeg', **kwargs)

    def _predict_direct(self, image_path: str, start_time: float) -> DetectionResult:
        img_tensor = self._load_image(image_path)

        try:
            with torch.no_grad():
                output = self._net(img_tensor)
                if isinstance(output, (tuple, list)) and len(output) >= 1:
                    output_seg = output[0]
                    if isinstance(output_seg, torch.Tensor) and output_seg.dim() >= 2:
                        # output_seg[:, 0, :, :] 是异常分割图
                        if output_seg.dim() == 4:
                            anomaly_map = output_seg[0, 0, :, :].cpu().numpy()
                        else:
                            anomaly_map = output_seg.squeeze().cpu().numpy()
                        score = float(anomaly_map.max())
                    else:
                        raise RuntimeError(f"Unexpected output_seg shape: {output_seg.shape if hasattr(output_seg, 'shape') else type(output_seg)}")
                else:
                    raise RuntimeError(f"Expected tuple output, got {type(output)}")
        except Exception as e:
            print(f"[ADer:{self.method}] DeSTSeg inference failed: {e}")
            return self._predict_fallback(image_path, start_time)

        inference_time = (time.time() - start_time) * 1000
        is_anomaly = score > self.threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=inference_time,
            metadata={'method': self.method, 'mode': 'destseg'}
        )


@register_algorithm("realnet")
class RealNetAdapter(ADerBaseAdapter):
    """RealNet — 真实场景异常检测

    forward(imgs, gt_imgs) → (logit_mask, pred, recon_f, gt_f)
    pred 直接作为异常图。推理时 gt_imgs 使用原图。
    """
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='RealNet', **kwargs)

    def _predict_direct(self, image_path: str, start_time: float) -> DetectionResult:
        img_tensor = self._load_image(image_path)

        try:
            with torch.no_grad():
                # RealNet 需要 (imgs, gt_imgs) 两个输入
                output = self._net(img_tensor, img_tensor)
                if isinstance(output, (tuple, list)) and len(output) >= 2:
                    pred = output[1]  # (logit_mask, pred, recon_f, gt_f)
                    if isinstance(pred, torch.Tensor):
                        anomaly_map = pred.squeeze().cpu().numpy()
                        score = float(anomaly_map.max())
                    else:
                        raise RuntimeError(f"Unexpected pred type: {type(pred)}")
                else:
                    raise RuntimeError(f"Expected tuple output, got {type(output)}")
        except Exception as e:
            print(f"[ADer:{self.method}] RealNet inference failed: {e}")
            return self._predict_fallback(image_path, start_time)

        inference_time = (time.time() - start_time) * 1000
        is_anomaly = score > self.threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=inference_time,
            metadata={'method': self.method, 'mode': 'realnet'}
        )


@register_algorithm("rdpp")
class RDPlusPlusAdapter(_FeatureComparisonMixin, ADerBaseAdapter):
    """RD++ — 增强反向蒸馏异常检测

    forward() → (feats_t, feats_s, L_proj)
    使用前两个特征元组进行 cal_anomaly_map()。
    """
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='RDpp', **kwargs)
