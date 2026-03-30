# -*- coding: utf-8 -*-
"""
Shazam - ASD 项目预处理器适配器

将 Shazam 音频指纹定位功能集成到 ASD 项目的预处理流程中，
作为 MFCC+DTW 定位算法的替代方案。
"""

import os
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .api import AudioFingerprinter, LocationResult


@dataclass
class AudioSegment:
    """音频片段信息"""
    start_time: float       # 起始时间（秒）
    end_time: float         # 结束时间（秒）
    confidence: float       # 置信度
    method: str             # 使用的方法


class ShazamLocator:
    """
    Shazam 音频定位器 - 用于 ASD 项目预处理

    功能：
        1. 在长音频中定位参考片段的位置
        2. 替代原有的 MFCCLocate 类
        3. 提供比 MFCC+DTW 更精准的定位

    用法:
        >>> locator = ShazamLocator(ref_audio="ref.wav")
        >>> segment = locator.locate("long_audio.wav")
        >>> print(f"片段位置: {segment.start_time}~{segment.end_time}s")
    """

    def __init__(self, ref_audio: str, sr: int = 16000,
                 threshold: int = 10, auto_add_ref: bool = True):
        """
        初始化 Shazam 定位器

        Args:
            ref_audio: 参考音频路径
            sr: 采样率（与 Shazam 配置保持一致）
            threshold: 匹配阈值，低于此值认为定位失败
            auto_add_ref: 自动将参考音频添加到指纹库
        """
        self.ref_audio = ref_audio
        self.sr = sr
        self.threshold = threshold
        self.auto_add_ref = auto_add_ref

        # 初始化指纹识别器
        self._fingerprinter: Optional[AudioFingerprinter] = None
        self._ref_added = False

        # 验证参考音频
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"参考音频不存在: {ref_audio}")

    def _get_fingerprinter(self) -> AudioFingerprinter:
        """获取指纹识别器（懒加载）"""
        if self._fingerprinter is None:
            self._fingerprinter = AudioFingerprinter()
            if self.auto_add_ref and not self._ref_added:
                self._fingerprinter.add_reference(self.ref_audio, name="shazam_ref")
                self._ref_added = True
        return self._fingerprinter

    def locate(self, audio_path: str, duration: Optional[float] = None) -> AudioSegment:
        """
        在音频中定位参考片段的位置

        Args:
            audio_path: 待定位的音频文件路径
            duration: 参考音频时长（秒），自动计算如果为None

        Returns:
            AudioSegment: 定位结果

        Raises:
            FileNotFoundError: 音频文件不存在
            RuntimeError: 定位失败
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # 获取参考音频时长
        if duration is None:
            import librosa
            duration = librosa.get_duration(path=self.ref_audio, sr=self.sr)

        # 使用 Shazam 定位
        fp = self._get_fingerprinter()
        location = fp.locate(
            long_audio_path=audio_path,
            reference_path=self.ref_audio,
            threshold=self.threshold
        )

        if not location.found:
            raise RuntimeError(f"定位失败: 在 {audio_path} 中未找到参考片段")

        return AudioSegment(
            start_time=location.start_time,
            end_time=location.start_time + duration,
            confidence=float(location.confidence),
            method="shazam"
        )

    def locate_batch(self, audio_paths: List[str],
                     duration: Optional[float] = None) -> List[Tuple[str, AudioSegment]]:
        """
        批量定位音频片段

        Args:
            audio_paths: 音频文件路径列表
            duration: 参考音频时长（秒）

        Returns:
            [(audio_path, AudioSegment), ...]: 定位结果列表
        """
        results = []
        for path in audio_paths:
            try:
                segment = self.locate(path, duration)
                results.append((path, segment))
            except Exception as e:
                print(f"定位失败 {path}: {e}")
                results.append((path, None))
        return results

    def close(self):
        """释放资源"""
        if self._fingerprinter:
            self._fingerprinter.close()
            self._fingerprinter = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


class ShazamPreprocessor:
    """
    Shazam 预处理器 - 完整的预处理流程

    将 Shazam 定位与时频图生成结合，直接替代原有的 Preprocessor 类

    用法:
        >>> preprocessor = ShazamPreprocessor(ref_file="ref.wav")
        >>> preprocessor.process_audio(
        ...     file_list=["audio1.wav", "audio2.wav"],
        ...     save_dir="output/"
        ... )
    """

    def __init__(self, ref_file: str, sr: int = 22050,
                 threshold: int = 10, img_size: Tuple[int, int] = (600, 600)):
        """
        初始化预处理器

        Args:
            ref_file: 参考音频文件
            sr: 目标采样率
            threshold: Shazam 匹配阈值
            img_size: 输出图片尺寸
        """
        self.ref_file = ref_file
        self.sr = sr
        self.threshold = threshold
        self.img_size = img_size

        # 初始化定位器
        self.locator = ShazamLocator(ref_audio=ref_file, threshold=threshold)

    def _generate_spectrogram(self, audio_path: str, output_path: str,
                              offset: float = 0.0, duration: Optional[float] = None):
        """
        生成时频图

        Args:
            audio_path: 音频文件路径
            output_path: 输出图片路径
            offset: 起始时间偏移
            duration: 时长
        """
        import librosa
        import librosa.display
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        # 加载音频
        y, sr = librosa.load(audio_path, offset=offset, duration=duration, sr=self.sr)

        # 幅值归一化
        y = librosa.util.normalize(y)

        # 计算STFT
        D = librosa.stft(y)

        # 转换为分贝标度
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # 绘制频谱图
        plt.figure(figsize=(self.img_size[0]/100, self.img_size[1]/100), dpi=100)
        librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def process_audio(self, file_list: List[str], save_dir: str) -> List[str]:
        """
        处理音频文件列表

        Args:
            file_list: 音频文件路径列表
            save_dir: 输出目录

        Returns:
            生成的图片路径列表
        """
        import librosa

        os.makedirs(save_dir, exist_ok=True)
        output_files = []

        # 获取参考音频时长
        ref_duration = librosa.get_duration(path=self.ref_file, sr=self.sr)

        for audio_path in file_list:
            if not os.path.exists(audio_path):
                print(f"文件不存在，跳过: {audio_path}")
                continue

            try:
                # 1. 定位片段
                print(f"正在处理: {audio_path}")
                segment = self.locator.locate(audio_path, duration=ref_duration)
                print(f"  定位成功: {segment.start_time:.2f}s ~ {segment.end_time:.2f}s")

                # 2. 生成时频图
                filename = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(save_dir, f"{filename}.png")

                self._generate_spectrogram(
                    audio_path=audio_path,
                    output_path=output_path,
                    offset=segment.start_time,
                    duration=ref_duration
                )

                output_files.append(output_path)
                print(f"  保存到: {output_path}")

            except Exception as e:
                print(f"  处理失败: {e}")

        return output_files

    def close(self):
        """释放资源"""
        if self.locator:
            self.locator.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def compare_with_mfcc_dtw(audio_path: str, ref_path: str):
    """
    对比 Shazam 和 MFCC+DTW 的定位效果

    Args:
        audio_path: 测试音频路径
        ref_path: 参考音频路径
    """
    print("=" * 60)
    print(f"对比测试: {audio_path}")
    print("=" * 60)

    # 方法1: Shazam
    print("\n[方法1] Shazam 指纹定位:")
    try:
        with ShazamLocator(ref_path) as locator:
            import librosa
            duration = librosa.get_duration(path=ref_path)
            segment = locator.locate(audio_path, duration=duration)
            print(f"  位置: {segment.start_time:.2f}s ~ {segment.end_time:.2f}s")
            print(f"  置信度: {segment.confidence}")
            print(f"  状态: ✓ 成功")
    except Exception as e:
        print(f"  状态: ✗ 失败 - {e}")

    # 方法2: MFCC+DTW（如果可用）
    print("\n[方法2] MFCC+DTW 定位:")
    try:
        # 尝试导入原有的预处理器
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from prepocessing import MFCCLocate

        locator = MFCCLocate(ref_file=ref_path, split_method='mfcc_dtw')
        offset = locator.locate(audio_path)
        import librosa
        duration = librosa.get_duration(path=ref_path)
        print(f"  位置: {offset:.2f}s ~ {offset + duration:.2f}s")
        print(f"  状态: ✓ 成功")
    except ImportError:
        print("  状态: - 未安装 MFCC+DTW 模块")
    except Exception as e:
        print(f"  状态: ✗ 失败 - {e}")


if __name__ == "__main__":
    # 测试用法
    print("Shazam ASD 适配器")
    print("=" * 60)

    # 示例1: 单独使用定位器
    print("\n示例1: 使用 ShazamLocator 定位音频")
    try:
        with ShazamLocator(ref_audio="ref/渡口片段10s.wav") as locator:
            segment = locator.locate("原始数据/test.wav")
            print(f"定位结果: {segment}")
    except Exception as e:
        print(f"需要测试文件: {e}")

    # 示例2: 完整预处理流程
    print("\n示例2: 使用 ShazamPreprocessor 预处理音频")
    try:
        with ShazamPreprocessor(ref_file="ref/渡口片段10s.wav") as preprocessor:
            output_files = preprocessor.process_audio(
                file_list=["原始数据/audio1.wav", "原始数据/audio2.wav"],
                save_dir="slice/"
            )
            print(f"生成文件: {output_files}")
    except Exception as e:
        print(f"需要测试文件: {e}")
