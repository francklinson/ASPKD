# -*- coding: utf-8 -*-
"""
PreciseSegmentLocator 使用示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.precise_segment_locator import PreciseSegmentLocator, SegmentLocatorConfig
from core.shazam.database.connector import MySQLConnector


def example_basic():
    """基础使用示例"""
    print("=" * 60)
    print("PreciseSegmentLocator 基础示例")
    print("=" * 60)
    
    # 创建配置
    config = SegmentLocatorConfig(
        threshold=10,              # 最小匹配hash数
        min_match_ratio=0.05,      # 最小匹配比例
        time_tolerance=2.0,        # 时间容差（秒）
        segment_duration=10.0      # 切分片段时长
    )
    
    # 创建定位器
    locator = PreciseSegmentLocator(config=config)
    
    # 从数据库添加参考音频
    db_connector = MySQLConnector()
    
    # 查找参考音频ID（假设数据库中已有这些音频）
    music_ids = {
        '渡口': db_connector.find_music_by_music_name('渡口'),
        '青藏高原': db_connector.find_music_by_music_name('青藏高原'),
    }
    
    # 添加参考音频到定位器
    for name, music_id in music_ids.items():
        if music_id:
            locator.add_reference(music_id, name)
        else:
            print(f"警告: 未找到参考音频 '{name}'")
    
    # 分析长音频
    audio_path = "ref/asd_src_audio.wav"  # 替换为实际路径
    
    if os.path.exists(audio_path):
        result = locator.locate_segments(audio_path)
        
        print("\n检测到的片段:")
        for segment in result.segments:
            print(f"  {segment.music_name}: {segment.start_time:.2f}s - {segment.end_time:.2f}s "
                  f"(置信度: {segment.confidence}, 可靠: {segment.is_reliable})")
    else:
        print(f"音频文件不存在: {audio_path}")
    
    locator.close()
    print("=" * 60)


def example_comparison():
    """与LongAudioAnalyzer对比示例"""
    print("=" * 60)
    print("性能对比: PreciseSegmentLocator vs LongAudioAnalyzer")
    print("=" * 60)
    
    import time
    
    audio_path = "ref/asd_src_audio.wav"
    if not os.path.exists(audio_path):
        print(f"音频文件不存在: {audio_path}")
        return
    
    # 1. 测试 PreciseSegmentLocator
    print("\n1. PreciseSegmentLocator:")
    from core.precise_segment_locator import PreciseSegmentLocator, SegmentLocatorConfig
    
    config = SegmentLocatorConfig()
    locator = PreciseSegmentLocator(config=config)
    
    # 添加参考音频
    db_connector = MySQLConnector()
    for name in ['渡口', '青藏高原']:
        music_id = db_connector.find_music_by_music_name(name)
        if music_id:
            locator.add_reference(music_id, name)
    
    start = time.time()
    result1 = locator.locate_segments(audio_path)
    time1 = time.time() - start
    
    print(f"   耗时: {time1:.2f}s")
    print(f"   片段数: {len(result1.segments)}")
    for seg in result1.segments:
        print(f"     - {seg.music_name}: {seg.start_time:.2f}s")
    
    locator.close()
    
    # 2. 测试 LongAudioAnalyzer
    print("\n2. LongAudioAnalyzer:")
    from core.long_audio_analyzer import LongAudioAnalyzer, AnalyzerConfig
    
    config2 = AnalyzerConfig(window_size=10.0, step_size=5.0)
    analyzer = LongAudioAnalyzer(config=config2, db_connector=db_connector)
    
    start = time.time()
    result2 = analyzer.analyze(audio_path)
    time2 = time.time() - start
    
    print(f"   耗时: {time2:.2f}s")
    print(f"   片段数: {len(result2.segment_matches)}")
    for seg in result2.segment_matches:
        print(f"     - {seg.music_name}: {seg.start_time:.2f}s")
    
    print(f"\n对比:")
    print(f"  速度提升: {time2/time1:.1f}x")
    print(f"  PreciseSegmentLocator {'✓ 更快' if time1 < time2 else '✗ 更慢'}")
    
    print("=" * 60)


def example_batch_processing():
    """批量处理示例"""
    print("=" * 60)
    print("批量处理多个音频文件")
    print("=" * 60)
    
    # 创建定位器
    config = SegmentLocatorConfig()
    locator = PreciseSegmentLocator(config=config)
    
    # 添加参考音频
    db_connector = MySQLConnector()
    ref_names = ['渡口', '青藏高原']
    for name in ref_names:
        music_id = db_connector.find_music_by_music_name(name)
        if music_id:
            locator.add_reference(music_id, name)
    
    # 批量处理
    audio_files = [
        "ref/asd_src_audio.wav",
        # 添加更多文件...
    ]
    
    results = []
    for audio_path in audio_files:
        if os.path.exists(audio_path):
            print(f"\n处理: {audio_path}")
            result = locator.locate_segments(audio_path)
            results.append(result)
            
            # 打印结果
            for seg in result.segments:
                print(f"  {seg.music_name}: {seg.start_time:.2f}s - {seg.end_time:.2f}s")
        else:
            print(f"跳过不存在文件: {audio_path}")
    
    locator.close()
    print("=" * 60)


if __name__ == "__main__":
    # 运行示例
    try:
        example_basic()
    except Exception as e:
        print(f"基础示例出错: {e}")
    
    print("\n")
    
    try:
        example_comparison()
    except Exception as e:
        print(f"对比示例出错: {e}")
