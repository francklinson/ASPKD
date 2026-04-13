# -*- coding: utf-8 -*-
"""
PreciseSegmentLocator - 精确片段定位器

针对长音频中多参考音频检测的优化方案。

核心特点：
1. 全局指纹提取：对整个音频一次性提取指纹，避免滑动窗口的冗余计算
2. 批量数据库查询：优化数据库访问，减少查询次数
3. 多点时间对齐：通过聚类分析找出所有参考音频的所有出现位置
4. 精确定位：直接计算精确起始时间（帧级别）

适用场景：
- 长音频中包含多个不同时间点的参考音频片段
- 需要精确定位每个片段的起始时间
- 需要高效处理，减少计算冗余

示例：
    from core.precise_segment_locator import PreciseSegmentLocator, SegmentLocatorConfig
    
    config = SegmentLocatorConfig()
    locator = PreciseSegmentLocator(config=config)
    
    # 添加参考音频
    locator.add_reference_from_db(music_id=1, music_name="渡口")
    locator.add_reference_from_db(music_id=2, music_name="青藏高原")
    
    # 分析长音频
    result = locator.locate_segments("long_audio.wav")
    
    for segment in result.segments:
        print(f"{segment.music_name}: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
"""

from .locator import PreciseSegmentLocator, SegmentLocatorConfig, SegmentInfo, LocatorResult

__all__ = [
    'PreciseSegmentLocator',
    'SegmentLocatorConfig',
    'SegmentInfo',
    'LocatorResult',
]
