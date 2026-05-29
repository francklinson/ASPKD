# -*- coding: utf-8 -*-
"""
STFT-based music fingerprinting implementation
"""

from .stft_create import StftMusicProcessorCreate
from .stft_predict import StftMusicProcessorPredict

__all__ = [
    'StftMusicProcessorCreate',
    'StftMusicProcessorPredict',
]
