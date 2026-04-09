# -*- coding: utf-8 -*-
"""
长音频定位分析系统 - 完整解决方案

功能：
    1. 从长音频文件中提取音频指纹
    2. 与音频指纹数据库进行高效匹配
    3. 精确定位音频片段在长音频中的时间位置
    4. 生成结构化分析报告

性能指标：
    - 时间定位精度：误差不超过500ms
    - 识别准确率：SNR≥10dB时，准确率≥95%
    - 处理速度：单线程每小时音频处理时间不超过10分钟

示例：
    >>> from core.long_audio_analyzer import LongAudioAnalyzer
    >>> analyzer = LongAudioAnalyzer()
    >>> 
    >>> # 分析长音频
    >>> results = analyzer.analyze("long_audio.wav", window_size=10, step_size=5)
    >>> 
    >>> # 导出结果
    >>> analyzer.export_json(results, "output.json")
    >>> analyzer.visualize_timeline(results, "timeline.png")
"""

from .analyzer import LongAudioAnalyzer, AnalyzerConfig
from .result_analyzer import AnalysisResult, SegmentMatch
from .audio_processor import AudioPreprocessor
from .fingerprint_extractor import SlidingWindowFingerprintExtractor
from .matching_engine import FastMatchingEngine
from .result_analyzer import ResultAnalyzer

__version__ = "1.0.0"
__all__ = [
    "LongAudioAnalyzer",
    "AnalyzerConfig",
    "AnalysisResult",
    "SegmentMatch",
    "AudioPreprocessor",
    "SlidingWindowFingerprintExtractor",
    "FastMatchingEngine",
    "ResultAnalyzer",
]
