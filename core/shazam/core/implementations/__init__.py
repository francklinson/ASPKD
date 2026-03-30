# -*- coding: utf-8 -*-
"""
Implementation classes for music fingerprinting
"""

from .stft.stft_create import StftMusicProcessorCreate
from .stft.stft_predict import StftMusicProcessorPredict

__all__ = [
    'StftMusicProcessorCreate',
    'StftMusicProcessorPredict',
]
