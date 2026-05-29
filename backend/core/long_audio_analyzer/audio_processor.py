# -*- coding: utf-8 -*-
"""
音频预处理模块

支持常见音频格式的解码与标准化，具备噪声检测和静音检测功能
"""

import os
import io
import librosa
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class AudioQuality(Enum):
    """音频质量等级"""
    HIGH = "high"           # 高质量
    MEDIUM = "medium"       # 中等质量
    LOW = "low"             # 低质量
    SILENCE = "silence"     # 静音/接近静音
    NOISY = "noisy"         # 噪声过大


@dataclass
class AudioSegmentInfo:
    """音频片段信息"""
    start_time: float       # 起始时间（秒）
    end_time: float         # 结束时间（秒）
    duration: float         # 时长
    sample_rate: int        # 采样率
    quality: AudioQuality   # 质量等级
    rms_energy: float       # RMS能量
    snr_db: float          # 信噪比（dB）
    is_silence: bool       # 是否为静音


class AudioPreprocessor:
    """
    音频预处理器
    
    功能：
        1. 支持多种音频格式（MP3、WAV、FLAC、AAC、OGG等）
        2. 音频标准化（采样率统一、声道转换）
        3. 静音检测和标记
        4. 音频质量评估
        5. 噪声水平估计
    """
    
    # 支持的音频格式
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    
    # 默认参数
    DEFAULT_SR = 16000      # 默认采样率
    DEFAULT_MONO = True     # 默认转换为单声道
    
    # 静音检测阈值
    SILENCE_THRESHOLD_DB = -40    # 静音阈值（dB）
    MIN_SILENCE_DURATION = 0.5    # 最小静音时长（秒）
    
    # 质量评估参数
    SNR_THRESHOLD_HIGH = 20       # 高信噪比阈值
    SNR_THRESHOLD_MEDIUM = 10     # 中等信噪比阈值
    
    def __init__(self, target_sr: int = DEFAULT_SR, mono: bool = DEFAULT_MONO):
        """
        初始化预处理器
        
        Args:
            target_sr: 目标采样率
            mono: 是否转换为单声道
        """
        self.target_sr = target_sr
        self.mono = mono
        
    def load_audio(self, audio_path: str, offset: float = 0.0, 
                   duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            offset: 起始偏移（秒）
            duration: 加载时长（秒），None表示加载全部
            
        Returns:
            (audio_data, sample_rate)
            
        Raises:
            ValueError: 不支持的音频格式
            FileNotFoundError: 文件不存在
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 检查文件格式
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的音频格式: {ext}，支持的格式: {self.SUPPORTED_FORMATS}")
        
        try:
            # 加载音频
            audio, sr = librosa.load(
                audio_path,
                sr=self.target_sr,
                mono=self.mono,
                offset=offset,
                duration=duration
            )
            
            return audio, sr
            
        except Exception as e:
            raise RuntimeError(f"音频加载失败: {e}")
    
    def analyze_segment(self, audio: np.ndarray, sr: int, 
                        start_time: float = 0.0) -> AudioSegmentInfo:
        """
        分析音频片段质量
        
        Args:
            audio: 音频数据
            sr: 采样率
            start_time: 起始时间
            
        Returns:
            AudioSegmentInfo: 片段信息
        """
        duration = len(audio) / sr
        end_time = start_time + duration
        
        # 计算RMS能量
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # 静音检测
        is_silence = rms_db < self.SILENCE_THRESHOLD_DB
        
        # 估计SNR
        snr_db = self._estimate_snr(audio)
        
        # 确定质量等级
        if is_silence:
            quality = AudioQuality.SILENCE
        elif snr_db < 5:
            quality = AudioQuality.NOISY
        elif snr_db >= self.SNR_THRESHOLD_HIGH:
            quality = AudioQuality.HIGH
        elif snr_db >= self.SNR_THRESHOLD_MEDIUM:
            quality = AudioQuality.MEDIUM
        else:
            quality = AudioQuality.LOW
        
        return AudioSegmentInfo(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            sample_rate=sr,
            quality=quality,
            rms_energy=rms_db,
            snr_db=snr_db,
            is_silence=is_silence
        )
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """
        估计音频信噪比
        
        使用基于能量的简单估计方法
        
        Args:
            audio: 音频数据
            
        Returns:
            估计的SNR（dB）
        """
        # 分帧
        frame_length = int(0.025 * self.target_sr)  # 25ms
        hop_length = int(0.010 * self.target_sr)    # 10ms
        
        # 计算每帧能量
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.sum(frames ** 2, axis=0)
        
        # 使用最低能量的帧作为噪声估计
        sorted_energy = np.sort(frame_energy)
        noise_frames = sorted_energy[:len(sorted_energy)//10]  # 取10%最低能量帧
        signal_frames = sorted_energy[-len(sorted_energy)//10:]  # 取10%最高能量帧
        
        noise_power = np.mean(noise_frames) + 1e-10
        signal_power = np.mean(signal_frames) + 1e-10
        
        snr = 10 * np.log10(signal_power / noise_power)
        return max(snr, 0)  # 确保非负
    
    def detect_silence_regions(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """
        检测静音区域
        
        Args:
            audio: 音频数据
            sr: 采样率
            
        Returns:
            静音区域列表 [(start, end), ...]
        """
        # 计算短时能量
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # 转换为dB
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # 检测静音帧
        silence_frames = rms_db < self.SILENCE_THRESHOLD_DB
        
        # 合并连续的静音帧
        silence_regions = []
        start_frame = None
        
        for i, is_silence in enumerate(silence_frames):
            if is_silence and start_frame is None:
                start_frame = i
            elif not is_silence and start_frame is not None:
                duration = (i - start_frame) * hop_length / sr
                if duration >= self.MIN_SILENCE_DURATION:
                    start_time = start_frame * hop_length / sr
                    end_time = i * hop_length / sr
                    silence_regions.append((start_time, end_time))
                start_frame = None
        
        # 处理结尾
        if start_frame is not None:
            duration = (len(silence_frames) - start_frame) * hop_length / sr
            if duration >= self.MIN_SILENCE_DURATION:
                start_time = start_frame * hop_length / sr
                end_time = len(audio) / sr
                silence_regions.append((start_time, end_time))
        
        return silence_regions
    
    def normalize_audio(self, audio: np.ndarray, 
                        target_db: float = -20.0) -> np.ndarray:
        """
        音频响度标准化
        
        Args:
            audio: 音频数据
            target_db: 目标响度（dB）
            
        Returns:
            标准化后的音频
        """
        current_rms = np.sqrt(np.mean(audio ** 2))
        current_db = 20 * np.log10(current_rms + 1e-10)
        
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20)
        
        # 限制增益范围，避免削波
        gain_linear = np.clip(gain_linear, 0.01, 10.0)
        
        return audio * gain_linear
    
    def preprocess_for_fingerprint(self, audio_path: str, 
                                    offset: float = 0.0,
                                    duration: Optional[float] = None) -> Dict:
        """
        完整的预处理流程
        
        Args:
            audio_path: 音频路径
            offset: 起始偏移
            duration: 处理时长
            
        Returns:
            包含音频数据和分析信息的字典
        """
        # 加载音频
        audio, sr = self.load_audio(audio_path, offset, duration)
        
        # 分析质量
        segment_info = self.analyze_segment(audio, sr, offset)
        
        # 标准化
        audio_normalized = self.normalize_audio(audio)
        
        # 检测静音区域
        silence_regions = self.detect_silence_regions(audio, sr)
        
        return {
            'audio': audio_normalized,
            'sample_rate': sr,
            'segment_info': segment_info,
            'silence_regions': silence_regions,
            'is_valid': not segment_info.is_silence and segment_info.snr_db >= 5
        }
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        获取音频时长
        
        Args:
            audio_path: 音频路径
            
        Returns:
            时长（秒）
        """
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            raise RuntimeError(f"无法获取音频时长: {e}")


# ==================== 便捷函数 ====================

def quick_preprocess(audio_path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    """
    快速预处理音频
    
    Args:
        audio_path: 音频路径
        target_sr: 目标采样率
        
    Returns:
        预处理后的音频数据，失败返回None
    """
    try:
        processor = AudioPreprocessor(target_sr=target_sr)
        result = processor.preprocess_for_fingerprint(audio_path)
        return result['audio'] if result['is_valid'] else None
    except Exception:
        return None


def batch_preprocess(audio_paths: List[str], 
                     target_sr: int = 16000) -> Dict[str, np.ndarray]:
    """
    批量预处理音频
    
    Args:
        audio_paths: 音频路径列表
        target_sr: 目标采样率
        
    Returns:
        路径到音频数据的映射
    """
    results = {}
    processor = AudioPreprocessor(target_sr=target_sr)
    
    for path in audio_paths:
        try:
            result = processor.preprocess_for_fingerprint(path)
            if result['is_valid']:
                results[path] = result['audio']
        except Exception:
            continue
    
    return results
