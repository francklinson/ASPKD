"""
ADer框架算法适配器 (v2)

支持模型:
- Benchmark 图像模型: cflow, pyramidflow, simplenet, destseg, realnet, rdpp
- 音频模型: mambaad, invad, vitad, unad (仅支持音频频谱图推理)
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


class ADerBaseAdapter(BaseDetector):
    """ADer框架基础适配器 (v2)

    支持两种模式:
    1. 基准模型 (Benchmark): 直接加载模型进行图片推理
    2. 音频模型: 仅支持频谱图推理

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
        from ADer import ADerTaskAssigner

        start_time = time.time()
        print(f"[ADer:{self.method}] Loading model...")

        self._assigner = ADerTaskAssigner(method=self.method)

        # 对于 Benchmark 图像模型，尝试加载实际模型权重
        try:
            self._load_benchmark_model()
        except Exception as e:
            print(f"[ADer:{self.method}] Benchmark model load failed: {e}")
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
        print(f"[ADer:{self.method}] Creating model: {cfg.model.type}")
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
        """直接使用模型推理"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self._transform(img).unsqueeze(0).to(self.device)

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
                    scores, preds = self._net.net_simplenet.predict(img_tensor)
                    score = float(scores.mean().cpu())
                    anomaly_map = preds[0].squeeze().cpu().numpy() if preds is not None else None
                else:
                    # 通用 forward 推理
                    output = self._net(img_tensor)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    if isinstance(output, torch.Tensor):
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
        """回退推理: 使用 ADerTaskAssigner 的 inference（音频管道）"""
        import shutil
        import tempfile

        inference_dir = tempfile.mkdtemp(prefix='ader_inference_')

        try:
            dst_path = os.path.join(inference_dir, os.path.basename(image_path))
            shutil.copy(image_path, dst_path)

            self._assigner.inference(inference_dir=inference_dir)

            # 尝试从 vis 目录读取结果
            score = 0.5
            is_anomaly = score > self.threshold
        except Exception as e:
            print(f"[ADer:{self.method}] Fallback inference failed: {e}")
            score = 0.5
            is_anomaly = False
        finally:
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
# 注册 ADer 各算法
# ============================================================================

@register_algorithm("mambaad")
class MambaADAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='MambaAD', **kwargs)


@register_algorithm("invad")
class InVadAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='InVad', **kwargs)


@register_algorithm("vitad")
class ViTADAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='ViTAD', **kwargs)


@register_algorithm("unad")
class UniADAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='UniAD', **kwargs)


@register_algorithm("cflow")
class CFlowAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='CFlow', **kwargs)


@register_algorithm("pyramidflow")
class PyramidFlowAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='PyramidFlow', **kwargs)


@register_algorithm("simplenet")
class SimpleNetAdapter(ADerBaseAdapter):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='SimpleNet', **kwargs)


@register_algorithm("destseg")
class DeSTSegAdapter(ADerBaseAdapter):
    """DeSTSeg — 分割范式异常检测 (ADer独家)"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='DeSTSeg', **kwargs)


@register_algorithm("realnet")
class RealNetAdapter(ADerBaseAdapter):
    """RealNet — 真实场景异常检测 (ADer独家)"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='RealNet', **kwargs)


@register_algorithm("rdpp")
class RDPlusPlusAdapter(ADerBaseAdapter):
    """RD++ — 增强反向蒸馏 (ADer独家)"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, method='RDpp', **kwargs)
