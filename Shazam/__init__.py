# -*- coding: utf-8 -*-
"""
Shazam 音频指纹识别模块 - 外部调用接口

功能：
    1. 音频指纹创建与存储
    2. 音频识别与匹配
    3. 音频片段定位

示例：
    >>> from Shazam import AudioFingerprinter
    >>> fingerprinter = AudioFingerprinter()
    >>>
    >>> # 添加参考音频到指纹库
    >>> fingerprinter.add_reference("reference.wav", name="参考音频")
    >>>
    >>> # 识别音频
    >>> result = fingerprinter.recognize("query.wav")
    >>> print(result.name, result.offset)
    >>>
    >>> # 定位音频片段
    >>> position = fingerprinter.locate("long_audio.wav", "reference.wav")
"""

from .api import AudioFingerprinter, RecognitionResult, LocationResult
from .api import create_fingerprint_db, batch_recognize, batch_locate

__version__ = "1.0.0"
__all__ = [
    "AudioFingerprinter",
    "RecognitionResult",
    "LocationResult",
    "create_fingerprint_db",
    "batch_recognize",
    "batch_locate",
]
