"""
核心管理器类 - 算法管理和资源管理
"""
import gc
import os
import sys
import torch
import torch.cuda as cuda
from typing import List

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from algorithms import create_detector
from core import ConfigManager


class UnifiedAlgorithmManager:
    """统一算法管理器 - 使用新的统一接口管理所有算法"""

    def __init__(self, device: str = 'auto'):
        print(f"\n{'='*60}")
        print(f"[模型管理器] 🚀 初始化 UnifiedAlgorithmManager")
        print(f"[模型管理器] 初始设备设置: {device}")
        print(f"{'='*60}")

        self.detector = None
        self.algorithm_chosen = ""
        self.device = device
        self._model_loaded = False

        # 使用绝对路径加载配置
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "algorithms.yaml"
        )
        print(f"[模型管理器] 配置文件路径: {config_path}")
        print(f"[模型管理器] 配置文件存在: {os.path.exists(config_path)}")

        self.config = ConfigManager(config_path)
        print(f"[模型管理器] 配置加载完成，base_dir={self.config.base_dir}")

        # 获取可用算法列表
        self.available_algorithms = self._get_gui_algorithms()
        print(f"[模型管理器] 可用算法列表: {self.available_algorithms}")
        print(f"[模型管理器] ✓ 初始化完成")
        print(f"{'='*60}\n")

    def _get_gui_algorithms(self) -> List[str]:
        """获取GUI可用的算法列表"""
        return ["dinomaly_dinov2_small", "dinomaly_dinov3_small"]

    def update_algorithm(self, algorithm_choice: str, device: str = None):
        """更新算法"""
        print(f"\n{'='*60}")
        print(f"[模型管理器] 🔄 开始更新算法: {algorithm_choice}")
        print(f"{'='*60}")

        assert algorithm_choice in self.available_algorithms

        if self.detector is not None and algorithm_choice == self.algorithm_chosen:
            print(f"[模型管理器] ℹ 算法未改变，无需更新")
            print(f"{'='*60}\n")
            return

        if device is not None:
            self.device = device

        # 清理旧模型
        if self.detector is not None:
            self._cleanup_old_model()

        # 创建新检测器
        try:
            print(f"[模型管理器] 🏗️ 创建新检测器...")
            self.detector = create_detector(
                algorithm_name=algorithm_choice,
                config_manager=self.config,
                device=self.device
            )
            self.algorithm_chosen = algorithm_choice
            print(f"[模型管理器] ✓ 检测器创建成功: {self.algorithm_chosen}")
        except Exception as e:
            print(f"[模型管理器] ✗ 加载算法失败: {e}")
            raise
        finally:
            print(f"{'='*60}\n")

    def _cleanup_old_model(self):
        """清理旧模型资源"""
        print(f"[模型管理器] 🧹 清理旧模型: {self.algorithm_chosen}")
        try:
            self.detector.release()
        except Exception as e:
            print(f"[模型管理器] ⚠ 旧模型释放失败: {e}")
        self.detector = None
        self._model_loaded = False
        self._clear_cuda_cache()

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model_loaded and self.detector is not None

    def load_model(self):
        """加载模型并更新状态"""
        if self.detector is None:
            raise RuntimeError("检测器未初始化，请先调用 update_algorithm")
        self.detector.load_model()
        self._model_loaded = True
        print(f"[模型管理器] ✓ 模型加载完成")

    def unload_model(self):
        """卸载模型并更新状态"""
        if self.detector is not None:
            self.detector.release()
        self._model_loaded = False
        print(f"[模型管理器] ✓ 模型已卸载")

    @staticmethod
    def _clear_cuda_cache():
        """清除GPU缓存"""
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
            cuda.empty_cache()
            cuda.synchronize()
            gc.collect()
        except Exception as e:
            print(f"[模型管理器] ⚠ CUDA缓存清理失败: {e}")
