# -*- coding: utf-8 -*-
"""
滑动窗口指纹提取模块

针对长音频的滑动窗口指纹提取，支持重叠窗口和自适应窗口大小
"""

import numpy as np
import librosa
import hashlib
from typing import List, Tuple, Dict, Optional, Generator
from dataclasses import dataclass
from scipy.ndimage import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure
import sys
import os

# 添加父目录到路径以导入shazam模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class WindowFingerprint:
    """窗口指纹数据"""
    window_id: int              # 窗口ID
    start_time: float           # 窗口起始时间（秒）
    end_time: float             # 窗口结束时间（秒）
    hashes: List[Tuple[str, int]]  # [(hash, offset), ...]
    hash_count: int             # 哈希数量
    
    
@dataclass
class FingerprintConfig:
    """指纹提取配置"""
    # STFT参数
    sr: int = 16000
    n_fft: int = 4096
    hop_length: int = 1024
    win_length: int = 4096
    
    # 峰值检测参数
    amp_min: int = 5
    neighborhood: int = 15
    
    # 哈希生成参数
    near_num: int = 20
    min_time_delta: int = 0
    max_time_delta: int = 200
    
    # 滑动窗口参数
    window_size: float = 10.0      # 窗口大小（秒）
    step_size: float = 5.0         # 步长（秒）
    min_window_size: float = 3.0   # 最小窗口大小
    
    # 自适应参数
    adaptive_window: bool = True   # 是否启用自适应窗口
    min_hashes_per_window: int = 50  # 每窗口最小哈希数


class FingerprintGenerator:
    """
    指纹生成器（基于Shazam算法）
    
    从音频数据中提取频谱峰值并生成哈希指纹
    """
    
    def __init__(self, config: FingerprintConfig):
        self.config = config
        
    def generate(self, audio: np.ndarray) -> List[Tuple[str, int]]:
        """
        生成音频指纹
        
        Args:
            audio: 音频数据
            
        Returns:
            [(hash, offset), ...]
        """
        # 1. STFT变换
        spectrogram = self._compute_spectrogram(audio)
        
        # 2. 频谱处理
        spectrogram = self._process_spectrogram(spectrogram)
        
        # 3. 峰值检测
        peaks = self._detect_peaks(spectrogram)
        
        # 4. 生成哈希
        hashes = list(self._generate_hashes(peaks))
        
        return hashes
    
    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """计算频谱图"""
        arr2D = librosa.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        return np.abs(arr2D)
    
    def _process_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """处理频谱图"""
        # 用最小值替换0
        min_val = np.min(spectrogram[np.nonzero(spectrogram)])
        spectrogram[spectrogram == 0] = min_val
        
        # 对数变换
        spectrogram = 10 * np.log10(spectrogram)
        spectrogram[spectrogram == -np.inf] = 0
        
        return spectrogram
    
    def _detect_peaks(self, spectrogram: np.ndarray) -> List[Tuple[int, int]]:
        """
        检测频谱峰值（星座图）
        
        Returns:
            [(time_idx, freq_idx), ...]
        """
        # 生成结构元素
        struct = generate_binary_structure(2, 1)
        neighborhood = iterate_structure(struct, self.config.neighborhood)
        
        # 局部最大值检测
        local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram
        
        # 获取峰值坐标
        amps = spectrogram[local_max]
        freqs, times = np.where(local_max)
        
        # 过滤低能量峰值
        peaks = []
        for t, f, amp in zip(times, freqs, amps):
            if amp > self.config.amp_min:
                peaks.append((int(t), int(f)))
        
        return peaks
    
    def _generate_hashes(self, peaks: List[Tuple[int, int]]) -> Generator[Tuple[str, int], None, None]:
        """
        生成组合哈希
        
        Args:
            peaks: [(time, freq), ...]
            
        Yields:
            (hash, offset)
        """
        # 按时间排序
        peaks = sorted(peaks, key=lambda x: x[0])
        
        # 锚点-近邻组合
        for i in range(len(peaks)):
            for j in range(1, self.config.near_num):
                if i + j < len(peaks):
                    t1, f1 = peaks[i]
                    t2, f2 = peaks[i + j]
                    
                    t_delta = t2 - t1
                    
                    if self.config.min_time_delta <= t_delta <= self.config.max_time_delta:
                        # 生成哈希
                        hash_str = f"{f1}|{f2}|{t_delta}"
                        hash_val = hashlib.sha1(hash_str.encode('utf-8')).hexdigest()
                        yield hash_val, t1


class SlidingWindowFingerprintExtractor:
    """
    滑动窗口指纹提取器
    
    针对长音频的滑动窗口指纹提取，支持：
    1. 固定大小滑动窗口
    2. 自适应窗口大小（根据音频内容调整）
    3. 重叠窗口处理
    4. 静音区域跳过
    """
    
    def __init__(self, config: Optional[FingerprintConfig] = None):
        self.config = config or FingerprintConfig()
        self.generator = FingerprintGenerator(self.config)
        
    def extract(self, audio: np.ndarray, sr: int, 
                start_offset: float = 0.0) -> List[WindowFingerprint]:
        """
        提取滑动窗口指纹
        
        Args:
            audio: 音频数据
            sr: 采样率
            start_offset: 起始时间偏移
            
        Returns:
            WindowFingerprint列表
        """
        window_size_samples = int(self.config.window_size * sr)
        step_size_samples = int(self.config.step_size * sr)
        min_window_samples = int(self.config.min_window_size * sr)
        
        fingerprints = []
        window_id = 0
        
        # 滑动窗口处理
        for start_sample in range(0, len(audio), step_size_samples):
            end_sample = min(start_sample + window_size_samples, len(audio))
            
            # 跳过过短的窗口
            if end_sample - start_sample < min_window_samples:
                continue
            
            # 提取窗口音频
            window_audio = audio[start_sample:end_sample]
            
            # 计算时间
            window_start_time = start_offset + start_sample / sr
            window_end_time = start_offset + end_sample / sr
            
            # 生成指纹
            hashes = self.generator.generate(window_audio)
            
            # 自适应窗口处理：如果哈希数不足，尝试扩展窗口
            if (self.config.adaptive_window and 
                len(hashes) < self.config.min_hashes_per_window and
                end_sample < len(audio)):
                
                # 扩展窗口直到哈希数足够或达到最大限制
                extended_end = end_sample
                max_extend = int(self.config.window_size * sr * 2)  # 最大扩展到2倍
                
                while (len(hashes) < self.config.min_hashes_per_window and 
                       extended_end - start_sample < max_extend and
                       extended_end < len(audio)):
                    
                    extended_end = min(extended_end + step_size_samples, len(audio))
                    window_audio = audio[start_sample:extended_end]
                    hashes = self.generator.generate(window_audio)
                
                window_end_time = start_offset + extended_end / sr
            
            # 创建窗口指纹
            if len(hashes) > 0:
                fp = WindowFingerprint(
                    window_id=window_id,
                    start_time=window_start_time,
                    end_time=window_end_time,
                    hashes=hashes,
                    hash_count=len(hashes)
                )
                fingerprints.append(fp)
                window_id += 1
        
        return fingerprints
    
    def extract_with_silence_skip(self, audio: np.ndarray, sr: int,
                                   silence_regions: List[Tuple[float, float]],
                                   start_offset: float = 0.0) -> List[WindowFingerprint]:
        """
        提取指纹（跳过静音区域）
        
        Args:
            audio: 音频数据
            sr: 采样率
            silence_regions: 静音区域列表 [(start, end), ...]
            start_offset: 起始时间偏移
            
        Returns:
            WindowFingerprint列表
        """
        # 合并重叠的静音区域
        silence_regions = self._merge_regions(silence_regions)
        
        # 提取有效区域
        valid_regions = self._get_valid_regions(
            len(audio) / sr, silence_regions, start_offset
        )
        
        all_fingerprints = []
        
        for region_start, region_end in valid_regions:
            start_sample = int((region_start - start_offset) * sr)
            end_sample = int((region_end - start_offset) * sr)
            
            region_audio = audio[start_sample:end_sample]
            
            fps = self.extract(region_audio, sr, region_start)
            all_fingerprints.extend(fps)
        
        # 重新编号
        for i, fp in enumerate(all_fingerprints):
            fp.window_id = i
        
        return all_fingerprints
    
    def _merge_regions(self, regions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """合并重叠的区域"""
        if not regions:
            return []
        
        # 按起始时间排序
        regions = sorted(regions, key=lambda x: x[0])
        
        merged = [regions[0]]
        for current in regions[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # 有重叠
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def _get_valid_regions(self, total_duration: float, 
                           silence_regions: List[Tuple[float, float]],
                           offset: float) -> List[Tuple[float, float]]:
        """获取有效（非静音）区域"""
        if not silence_regions:
            return [(offset, offset + total_duration)]
        
        valid_regions = []
        current_pos = offset
        
        for silence_start, silence_end in silence_regions:
            if current_pos < silence_start:
                valid_regions.append((current_pos, silence_start))
            current_pos = max(current_pos, silence_end)
        
        if current_pos < offset + total_duration:
            valid_regions.append((current_pos, offset + total_duration))
        
        return valid_regions
    
    def extract_batch(self, audio_segments: List[Tuple[np.ndarray, int, float]]) -> List[List[WindowFingerprint]]:
        """
        批量提取指纹
        
        Args:
            audio_segments: [(audio, sr, offset), ...]
            
        Returns:
            WindowFingerprint列表的列表
        """
        results = []
        for audio, sr, offset in audio_segments:
            fps = self.extract(audio, sr, offset)
            results.append(fps)
        return results


# ==================== 便捷函数 ====================

def quick_extract(audio_path: str, window_size: float = 10.0, 
                  step_size: float = 5.0) -> List[WindowFingerprint]:
    """
    快速提取音频指纹
    
    Args:
        audio_path: 音频路径
        window_size: 窗口大小（秒）
        step_size: 步长（秒）
        
    Returns:
        WindowFingerprint列表
    """
    from .audio_processor import AudioPreprocessor
    
    # 预处理
    processor = AudioPreprocessor()
    result = processor.preprocess_for_fingerprint(audio_path)
    
    if not result['is_valid']:
        return []
    
    # 提取指纹
    config = FingerprintConfig(window_size=window_size, step_size=step_size)
    extractor = SlidingWindowFingerprintExtractor(config)
    
    return extractor.extract_with_silence_skip(
        result['audio'],
        result['sample_rate'],
        result['silence_regions']
    )
