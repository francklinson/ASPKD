# -*- coding: utf-8 -*-
"""
PreciseSegmentLocator 适配器

用于无缝替换 LongAudioAnalyzer，兼容现有接口
"""

from typing import List, Optional, Callable
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.precise_segment_locator.locator import (
    PreciseSegmentLocator, 
    SegmentLocatorConfig,
    SegmentInfo,
    LocatorResult
)
from core.long_audio_analyzer import AnalyzerConfig
from dataclasses import dataclass, field


@dataclass
class SegmentMatchAdapter:
    """适配 SegmentMatch 接口，兼容 LongAudioAnalyzer 的输出格式"""
    music_id: int
    music_name: str
    start_time: float
    end_time: float
    confidence: int
    match_ratio: float = 0.0
    window_count: int = 1
    is_reliable: bool = True
    
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class AnalysisResultAdapter:
    """适配 AnalysisResult 接口"""
    audio_path: str
    total_duration: float
    segment_matches: List[SegmentMatchAdapter] = field(default_factory=list)
    processing_time: float = 0.0
    stats: dict = field(default_factory=dict)
    
    def get_reliable_matches(self):
        return [m for m in self.segment_matches if m.is_reliable]


class PreciseSegmentLocatorAdapter:
    """
    PreciseSegmentLocator 适配器
    
    提供与 LongAudioAnalyzer 兼容的接口，便于无缝替换
    
    使用方式（替换 monitor_service.py 中的 LongAudioAnalyzer）:
        
        # 旧代码
        from core.long_audio_analyzer import LongAudioAnalyzer, AnalyzerConfig
        config = AnalyzerConfig(...)
        analyzer = LongAudioAnalyzer(config, db_connector)
        
        # 新代码
        from core.precise_segment_locator.adapter import PreciseSegmentLocatorAdapter
        config = AnalyzerConfig(...)  # 配置兼容
        analyzer = PreciseSegmentLocatorAdapter(config, db_connector)
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None, db_connector=None):
        """
        初始化适配器
        
        Args:
            config: AnalyzerConfig 配置对象（兼容 LongAudioAnalyzer 的配置）
            db_connector: 数据库连接器
        """
        # 转换配置
        locator_config = SegmentLocatorConfig()
        if config:
            locator_config.threshold = config.match_threshold
            locator_config.time_tolerance = config.time_tolerance
            locator_config.min_segment_duration = config.min_segment_duration
            locator_config.segment_duration = config.window_size
        
        self.locator = PreciseSegmentLocator(config=locator_config, db_connector=db_connector)
        self.db_connector = db_connector
        
    def add_reference(self, audio_path: str, music_id: int, music_name: str = "") -> bool:
        """
        添加参考音频（兼容接口）
        
        Args:
            audio_path: 音频路径（未使用，兼容参数）
            music_id: 音乐ID
            music_name: 音乐名称
            
        Returns:
            是否成功
        """
        return self.locator.add_reference(music_id, music_name)
    
    def analyze(self, audio_path: str, 
                progress_callback: Optional[Callable[[str, float], None]] = None
                ) -> AnalysisResultAdapter:
        """
        分析长音频（兼容 LongAudioAnalyzer.analyze 接口）
        
        Args:
            audio_path: 音频文件路径
            progress_callback: 进度回调函数
            
        Returns:
            AnalysisResultAdapter（兼容 AnalysisResult 格式）
        """
        if progress_callback:
            progress_callback("提取音频指纹...", 0.25)
        
        # 执行定位
        result = self.locator.locate_segments(audio_path)
        
        if progress_callback:
            progress_callback("分析完成", 1.0)
        
        # 转换为兼容格式
        segment_matches = []
        for seg in result.segments:
            segment_matches.append(SegmentMatchAdapter(
                music_id=seg.music_id,
                music_name=seg.music_name,
                start_time=seg.start_time,
                end_time=seg.end_time,
                confidence=seg.confidence,
                match_ratio=seg.match_ratio,
                is_reliable=seg.is_reliable
            ))
        
        # 构建统计信息
        stats = {
            'window_count': 1,  # 全局提取，只有一个"窗口"
            'match_count': len(result.segments),
            'coverage': sum(s.duration() for s in result.segments) / max(result.total_duration, 1),
            'audio_quality': 'good',
            'processing_method': 'PreciseSegmentLocator'
        }
        
        return AnalysisResultAdapter(
            audio_path=audio_path,
            total_duration=result.total_duration,
            segment_matches=segment_matches,
            processing_time=result.processing_time,
            stats=stats
        )
    
    def close(self):
        """释放资源"""
        self.locator.close()
    
    # 以下属性用于兼容 LongAudioAnalyzer 的内部访问
    @property
    def matching_engine(self):
        """兼容属性：返回 locator 的引用索引"""
        return self.locator
    
    @property
    def index(self):
        """兼容属性"""
        return self.locator.reference_index
