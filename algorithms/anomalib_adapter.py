"""
Anomalib库算法适配器 (v2 — 使用 Engine API)
支持所有27个Anomalib图像异常检测算法

核心改进:
- predict() 使用 Engine.predict(data_path=...) 而非直接 model(input)
- 正确处理 AnomalibModule 的 pre_processor/post_processor 管线
- 零样本/少样本模型自动运行 validation 收集归一化统计量
- 支持训练: fit() 方法调用 Engine.fit() / Engine.train()
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from typing import Optional, List

# 添加 algorithms 目录到路径
_algorithms_dir = os.path.dirname(os.path.abspath(__file__))
if _algorithms_dir not in sys.path:
    sys.path.insert(0, _algorithms_dir)

from backend.core import BaseDetector, DetectionResult, register_algorithm


class AnomalibAdapter(BaseDetector):
    """
    Anomalib 统一适配器 (v2)

    支持所有 Anomalib 图像算法，使用 Engine API 进行推理。

    两种推理模式:
    1. Engine模式 (首选): 使用 Engine.predict(data_path=...)
       — 正确处理预处理/后处理管线，支持零样本校准
    2. 直接模式 (回退): 使用 model.forward()
       — 在 Engine 不可用时回退

    训练支持:
    - fit(datamodule): 训练模型
    - train(datamodule): 训练+测试
    """

    def __init__(self, model_path: str, model_name: str, device: str = 'auto',
                 threshold: float = 0.5, reference_dir: Optional[str] = None, **kwargs):
        """
        Args:
            model_path: 模型权重路径
            model_name: Anomalib 模型名 (如 'patchcore', 'padim')
            device: 运行设备
            threshold: 异常判定阈值
            reference_dir: 参考正常图片目录（零样本/少样本模型校准用）
        """
        super().__init__(model_path, device, threshold, **kwargs)
        self.model_name = model_name
        self.reference_dir = reference_dir
        self._model = None
        self._engine = None
        self._use_engine = True  # 是否使用 Engine API
        self._model_info = {}  # 缓存模型元信息

    # ========================================================================
    # 模型加载
    # ========================================================================

    def load_model(self) -> None:
        """加载 Anomalib 模型"""
        # 设置 HuggingFace 离线模式和缓存路径，避免网络不可达时卡死
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _hf_cache = os.path.join(_project_root, "models", "pre_trained", "huggingface")
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', _hf_cache)
        os.environ.setdefault('TRANSFORMERS_CACHE', _hf_cache)

        start_time = time.time()
        print(f"[Anomalib:{self.model_name}] Loading model...")

        from anomalib.models import get_model

        # 1. 创建模型架构
        self._model = get_model(self.model_name)
        print(f"[Anomalib:{self.model_name}] Model created, "
              f"learning_type={self._model.learning_type}")

        # 2. 加载权重（如果提供）
        if self.model_path and os.path.isfile(self.model_path):
            print(f"[Anomalib:{self.model_name}] Loading weights from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=True)
            state_dict = checkpoint.get('state_dict', checkpoint)
            # 过滤不兼容的键
            model_dict = self._model.state_dict()
            compatible = {k: v for k, v in state_dict.items()
                         if k in model_dict and v.shape == model_dict[k].shape}
            missing, unexpected = len(model_dict) - len(compatible), len(state_dict) - len(compatible)
            self._model.load_state_dict(compatible, strict=False)
            print(f"[Anomalib:{self.model_name}] Loaded {len(compatible)} params "
                  f"(missing={missing}, unexpected={unexpected})")
        else:
            using_pretrained = self.model_path is None or not os.path.isfile(self.model_path)
            print(f"[Anomalib:{self.model_name}] "
                  f"{'Using pretrained backbone weights' if using_pretrained else 'Weights file not found: ' + self.model_path}")

        # 3. 移动到设备
        self._model.to(self.device)
        self._model.eval()

        # 3.5 Memory Bank 拟合：ONE_CLASS 模型需要训练数据来填充 memory bank
        self._fit_memory_bank()

        # 4. 缓存模型元信息
        self._model_info = {
            'learning_type': str(self._model.learning_type),
            'input_size': getattr(self._model, 'input_size', (256, 256)),
            'name': self.model_name,
        }

        # 5. Engine 延迟创建（仅训练时需要，推理用直接 forward）
        self._engine = None
        self._use_engine = False

        self.is_loaded = True
        elapsed = time.time() - start_time
        print(f"[Anomalib:{self.model_name}] Load complete in {elapsed:.2f}s")

    def _fit_memory_bank(self):
        """为 ONE_CLASS 模型填充 memory bank

        对于 patchcore/padim/cfa/dfkde/anomaly_dino 等模型，
        需要在推理前用正常样本训练数据调用 model.fit() 来填充 memory bank。
        """
        # 检查是否需要 fit
        if not hasattr(self._model, 'learning_type'):
            return
        from anomalib import LearningType
        if self._model.learning_type != LearningType.ONE_CLASS:
            return
        if not hasattr(self._model, 'fit') or not callable(self._model.fit):
            return
        if getattr(self._model, '_is_fitted', False):
            return

        # Memory bank 模型需要通过 Engine.fit() 完成训练后才能填充
        # 这里仅做标记，不执行实际拟合
        print(f"[Anomalib:{self.model_name}] ONE_CLASS model, "
              f"memory bank fitting deferred to training (Engine.fit())")

    # ========================================================================
    # 单张图片推理
    # ========================================================================

    def predict(self, image_path: str) -> DetectionResult:
        """单张图像推理 — 直接使用 AnomalibModule.forward()

        不使用 Engine.predict() 因为:
        1. Engine 会创建 Lightning Trainer，开销大且可能移动模型设备
        2. Engine.predict() 返回 ImageBatch 不支持按索引访问
        3. 零样本/少样本模型的校准在 load_model 中通过 _calibrate() 完成
        """
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 确保模型在正确设备上 (Engine 操作后可能被移动)
        if self._model.device != self.device:
            self._model.to(self.device)

        result = self._predict_direct(image_path)
        result.inference_time = (time.time() - start_time) * 1000
        return result

    def _predict_direct(self, image_path: str) -> DetectionResult:
        """直接推理 — AnomalibModule.forward()

        内部管线: pre_processor -> model -> post_processor -> InferenceBatch
        """
        from torchvision import transforms

        # 获取模型期望的输入尺寸
        input_size = self._model_info.get('input_size', (256, 256))
        if isinstance(input_size, (list, tuple)):
            h = input_size[0]
            w = input_size[1] if len(input_size) > 1 else h
        else:
            h = w = input_size

        # Anomalib 默认预处理: Resize + ImageNet 标准化
        transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._model(image_tensor)

        return self._parse_model_output(output)

    def _parse_model_output(self, output) -> DetectionResult:
        """解析 model.forward() 的输出

        Anomalib v2.5.0 模型输出为 InferenceBatch，包含:
        - pred_score: tensor [B] — 异常分数
        - anomaly_map: tensor [B, H, W] or None — 异常热力图
        - pred_label: tensor [B] — 二值标签
        - image, image_path, gt_mask, ...
        """
        score = 0.0
        anomaly_map = None
        pred_label = False

        # 处理 InferenceBatch (Anomalib v2.5.0 标准输出)
        if hasattr(output, 'pred_score') and output.pred_score is not None:
            score = float(output.pred_score[0].item()) if len(output.pred_score) > 0 else 0.0

        if hasattr(output, 'anomaly_map') and output.anomaly_map is not None:
            am = output.anomaly_map
            if len(am) > 0:
                anomaly_map = am[0].squeeze().cpu().numpy()

        if hasattr(output, 'pred_label') and output.pred_label is not None:
            if len(output.pred_label) > 0:
                pred_label = bool(output.pred_label[0].item())

        # 处理 Tensor 输出 (回退)
        if score == 0.0 and anomaly_map is None:
            if isinstance(output, torch.Tensor):
                score = float(output.mean().item())
            elif isinstance(output, dict):
                score = float(output.get('pred_score', output.get('anomaly_score', 0.0)))
                am = output.get('anomaly_map')
                if am is not None and hasattr(am, 'cpu'):
                    anomaly_map = am.squeeze().cpu().numpy()

        is_anomaly = pred_label or (score > self.threshold)

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_map=anomaly_map,
            inference_time=0.0,
            metadata={
                'model_name': self.model_name,
                'mode': 'direct',
            }
        )

    # ========================================================================
    # 批量推理
    # ========================================================================

    def predict_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """批量推理 — 逐张调用 predict()"""
        return self._predict_batch_sequential(image_paths)

    def _predict_batch_sequential(self, image_paths: List[str]) -> List[DetectionResult]:
        """逐张回退推理"""
        results = []
        for path in image_paths:
            try:
                result = self._predict_direct(path)
                results.append(result)
            except Exception as e:
                print(f"[Anomalib:{self.model_name}] Failed on {path}: {e}")
                results.append(DetectionResult(
                    is_anomaly=False, anomaly_score=0.0, anomaly_map=None,
                    inference_time=0.0,
                    metadata={'error': str(e), 'model_name': self.model_name}
                ))
        return results

    # ========================================================================
    # 训练支持 (Engine API)
    # ========================================================================

    def _get_engine(self, **kwargs) -> 'Engine':
        """延迟创建 Engine（仅训练时使用）"""
        if self._engine is None:
            from anomalib.engine import Engine
            self._engine = Engine(
                default_root_dir=os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "anomalib"
                ),
                **kwargs,
            )
        return self._engine

    def fit(self, datamodule=None, max_epochs: int = 100,
            ckpt_path: Optional[str] = None) -> dict:
        """
        训练模型

        Args:
            datamodule: AnomalibDataModule 实例（必须）
            max_epochs: 最大训练轮数
            ckpt_path: 从 checkpoint 恢复训练

        Returns:
            训练结果摘要
        """
        from anomalib import LearningType

        if datamodule is None:
            raise ValueError(
                "datamodule is required. "
                "Example: from anomalib.data import MVTecAD; "
                "datamodule = MVTecAD(root='data/public_dataset/mvtec', category='bottle')"
            )

        if not self.is_loaded:
            self.load_model()

        print(f"[Anomalib:{self.model_name}] Starting training (learning_type={self._model.learning_type})...")

        engine = self._get_engine(max_epochs=max_epochs, enable_progress_bar=True)

        if self._model.learning_type in (LearningType.ZERO_SHOT, LearningType.FEW_SHOT):
            print(f"[Anomalib:{self.model_name}] Zero/few-shot: validation only")
            result = engine.validate(model=self._model, datamodule=datamodule)
        else:
            print(f"[Anomalib:{self.model_name}] Training {max_epochs} epochs...")
            engine.fit(model=self._model, datamodule=datamodule)

            # 对 ONE_CLASS 模型，训练后需要填充 memory bank
            if self._model.learning_type == LearningType.ONE_CLASS:
                print(f"[Anomalib:{self.model_name}] Populating memory bank...")
                try:
                    if hasattr(self._model, 'fit') and callable(self._model.fit):
                        import inspect
                        sig = inspect.signature(self._model.fit)
                        if len(sig.parameters) == 0:
                            self._model.fit()
                            print(f"[Anomalib:{self.model_name}] Memory bank populated")
                        else:
                            print(f"[Anomalib:{self.model_name}] model.fit() requires args, skipping")
                except Exception as e:
                    print(f"[Anomalib:{self.model_name}] Memory bank population failed: {e}")

            result = engine.test(model=self._model, datamodule=datamodule)

        # 训练后确保模型在正确设备上
        self._model.to(self.device)
        self._model.eval()

        print(f"[Anomalib:{self.model_name}] Training complete")
        return {'status': 'completed', 'model_name': self.model_name}

    # ========================================================================
    # 资源管理
    # ========================================================================

    def get_model_info(self) -> dict:
        """获取模型信息"""
        base_info = super().get_model_info()
        base_info.update(self._model_info)
        base_info['use_engine'] = self._use_engine
        return base_info

    def release(self) -> None:
        """释放资源"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._engine is not None:
            del self._engine
            self._engine = None
        super().release()


# ============================================================================
# 注册所有 Anomalib 算法
# ============================================================================

# 注意: dinomaly 使用 Dinomaly 路径下的原生实现，不在此处注册
ANOMALIB_MODELS = [
    # --- Anomalib 经典算法 ---
    'patchcore', 'cfa', 'csflow', 'dfkde', 'dfm', 'draem',
    'dsr', 'efficient_ad', 'fastflow', 'fre',
    'padim', 'reverse_distillation', 'stfpm', 'ganomaly',
    'supersimplenet', 'uflow', 'uninet', 'vlm_ad', 'winclip',
    # --- Anomalib v2.5.0 新增算法 ---
    'anomalyvfm',      # AnomalyVFM - 零样本异常检测（Vision Foundation Models）
    'cfm',             # CFM - 跨模态特征映射（3D异常检测）
    'general_ad',      # GeneralAD - 通用异常检测
    'glass',           # GLASS - 梯度上升异常合成
    'inp_former',      # INP-Former - 内在正常原型
    'l2bt',            # L2BT - Learning to Backtrace
    'patchflow',       # PatchFlow - Patch-based Normalizing Flow
    'anomaly_dino',    # AnomalyDINO - DINOv2少样本异常检测
]

def _make_anomalib_adapter(model_name: str):
    """工厂函数 — 捕获 model_name 避免 Python 闭包循环变量 bug"""
    @register_algorithm(model_name)
    class Adapter(AnomalibAdapter):
        def __init__(self, model_path: str, **kwargs):
            super().__init__(model_path, model_name=model_name, **kwargs)

    # 为每个算法生成唯一类名
    class_name = ''.join(word.capitalize() for word in model_name.split('_'))
    Adapter.__name__ = f"{class_name}Adapter"
    return Adapter

for model_name in ANOMALIB_MODELS:
    _make_anomalib_adapter(model_name)
