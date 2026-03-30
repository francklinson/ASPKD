# -*- coding: utf-8 -*-
"""
Abstract base classes for music processors
"""

from .base_processor import MusicProcessor
from .create_processor import MusicProcessorCreate
from .predict_processor import MusicProcessorPredict

__all__ = [
    'MusicProcessor',
    'MusicProcessorCreate',
    'MusicProcessorPredict',
]
