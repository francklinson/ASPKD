# -*- coding: utf-8 -*-
"""
Core module for audio fingerprinting algorithms
"""

from .interfaces.base_processor import MusicProcessor
from .interfaces.create_processor import MusicProcessorCreate
from .interfaces.predict_processor import MusicProcessorPredict
from .implementations.stft.stft_create import StftMusicProcessorCreate
from .implementations.stft.stft_predict import StftMusicProcessorPredict

__all__ = [
    'MusicProcessor',
    'MusicProcessorCreate',
    'MusicProcessorPredict',
    'StftMusicProcessorCreate',
    'StftMusicProcessorPredict',
]
