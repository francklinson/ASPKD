# -*- coding: utf-8 -*-
"""
长音频定位分析系统 - 使用示例

演示如何使用LongAudioAnalyzer进行长音频分析
"""

import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from long_audio_analyzer import (
    LongAudioAnalyzer, 
    AnalyzerConfig,
    AnalysisResult,
    SegmentMatch
)


def example_1_basic_usage():
    """示例1: 基础用法 - 分析单个长音频文件"""
    print("="*60)
    print("示例1: 基础用法")
    print("="*60)
    
    # 创建分析器（使用默认配置）
    analyzer = LongAudioAnalyzer()
    
    # 添加参考音频到索引
    analyzer.add_reference("ref/song1.wav", music_id=1, music_name="歌曲1")
    analyzer.add_reference("ref/song2.wav", music_id=2, music_name="歌曲2")
    
    # 分析长音频
    audio_path = "long_audio.wav"
    
    # 定义进度回调
    def progress_callback(message: str, progress: float):
        print(f"[{progress*100:5.1f}%] {message}")
    
    # 执行分析
    result = analyzer.analyze(audio_path, progress_callback)
    
    # 打印报告
    analyzer.print_report(result)
    
    return result


def example_2_custom_config():
    """示例2: 使用自定义配置"""
    print("\n" + "="*60)
    print("示例2: 自定义配置")
    print("="*60)
    
    # 创建自定义配置
    config = AnalyzerConfig(
        window_size=15.0,           # 15秒窗口
        step_size=7.5,              # 7.5秒步长
        match_threshold=15,         # 更高的匹配阈值
        min_segment_duration=5.0,   # 最小5秒片段
        use_parallel=True,          # 启用并行处理
        max_workers=8,              # 8个线程
        skip_silence=True           # 跳过静音区域
    )
    
    # 使用自定义配置创建分析器
    analyzer = LongAudioAnalyzer(config=config)
    
    # 添加参考音频...
    # analyzer.add_reference(...)
    
    # 分析
    # result = analyzer.analyze("audio.wav")
    
    print("自定义配置创建成功:")
    print(f"  窗口大小: {config.window_size}s")
    print(f"  步长: {config.step_size}s")
    print(f"  匹配阈值: {config.match_threshold}")
    print(f"  线程数: {config.max_workers}")


def example_3_batch_processing():
    """示例3: 批量处理多个文件"""
    print("\n" + "="*60)
    print("示例3: 批量处理")
    print("="*60)
    
    analyzer = LongAudioAnalyzer()
    
    # 添加参考音频到索引
    # analyzer.add_reference(...)
    
    # 批量分析
    audio_files = [
        "audio1.wav",
        "audio2.wav",
        "audio3.wav"
    ]
    
    def batch_progress(message: str, progress: float):
        print(f"[总进度 {progress*100:5.1f}%] {message}")
    
    results = analyzer.analyze_batch(audio_files, batch_progress)
    
    # 处理结果
    for path, result in zip(audio_files, results):
        print(f"\n文件: {path}")
        print(f"  发现 {len(result.segment_matches)} 个匹配片段")
        print(f"  处理时间: {result.processing_time:.2f}秒")


def example_4_with_database():
    """示例4: 使用数据库连接"""
    print("\n" + "="*60)
    print("示例4: 数据库连接")
    print("="*60)
    
    try:
        from shazam.database.connector import MySQLConnector
        
        # 创建数据库连接
        connector = MySQLConnector()
        
        # 创建分析器（自动从数据库构建索引）
        analyzer = LongAudioAnalyzer(db_connector=connector)
        
        print("数据库索引构建成功")
        print(f"索引统计: {analyzer.matching_engine.get_index_stats()}")
        
        # 分析音频
        # result = analyzer.analyze("audio.wav")
        
    except ImportError:
        print("数据库模块未安装，跳过此示例")
    except Exception as e:
        print(f"数据库连接失败: {e}")


def example_5_export_results():
    """示例5: 导出分析结果"""
    print("\n" + "="*60)
    print("示例5: 结果导出")
    print("="*60)
    
    analyzer = LongAudioAnalyzer()
    
    # 假设已完成分析
    # result = analyzer.analyze("audio.wav")
    
    # 创建示例结果
    result = AnalysisResult(
        audio_path="example.wav",
        total_duration=120.0,
        segment_matches=[
            SegmentMatch(
                music_id=1,
                music_name="示例歌曲1",
                start_time=10.0,
                end_time=25.0,
                confidence=150,
                match_ratio=0.75,
                window_count=3,
                is_reliable=True
            ),
            SegmentMatch(
                music_id=2,
                music_name="示例歌曲2",
                start_time=60.0,
                end_time=80.0,
                confidence=200,
                match_ratio=0.85,
                window_count=4,
                is_reliable=True
            )
        ]
    )
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 导出JSON
    json_path = os.path.join(output_dir, "report.json")
    analyzer.export_json(result, json_path)
    print(f"JSON导出: {json_path}")
    
    # 导出CSV
    csv_path = os.path.join(output_dir, "report.csv")
    analyzer.export_csv(result, csv_path)
    print(f"CSV导出: {csv_path}")
    
    # 生成可视化
    viz_path = os.path.join(output_dir, "timeline.png")
    analyzer.visualize_timeline(result, viz_path)
    print(f"可视化: {viz_path}")
    
    # 打印JSON内容预览
    print("\nJSON内容预览:")
    print(result.to_json(indent=2)[:500] + "...")


def example_6_analyze_with_report():
    """示例6: 一键生成完整报告"""
    print("\n" + "="*60)
    print("示例6: 一键生成报告")
    print("="*60)
    
    from long_audio_analyzer.analyzer import analyze_with_report
    
    # 分析并生成完整报告
    # analyze_with_report(
    #     audio_path="long_audio.wav",
    #     output_dir="reports",
    #     db_connector=None  # 可选
    # )
    
    print("使用说明:")
    print("  from long_audio_analyzer import analyze_with_report")
    print("  analyze_with_report('audio.wav', 'output_dir')")
    print("\n将生成以下文件:")
    print("  - report.json: 详细JSON报告")
    print("  - report.csv: CSV格式数据")
    print("  - timeline.png: 可视化时间线")


def example_7_process_results():
    """示例7: 处理分析结果"""
    print("\n" + "="*60)
    print("示例7: 结果处理")
    print("="*60)
    
    # 创建示例结果
    result = AnalysisResult(
        audio_path="example.wav",
        total_duration=300.0,
        segment_matches=[
            SegmentMatch(1, "歌曲A", 10.0, 30.0, 100, 0.5, 4, True),
            SegmentMatch(1, "歌曲A", 100.0, 120.0, 120, 0.6, 4, True),
            SegmentMatch(2, "歌曲B", 50.0, 70.0, 80, 0.4, 4, False),
            SegmentMatch(3, "歌曲C", 150.0, 180.0, 150, 0.7, 6, True),
        ]
    )
    
    print(f"总片段数: {len(result.segment_matches)}")
    
    # 获取可靠匹配
    reliable = result.get_reliable_matches()
    print(f"可靠匹配数: {len(reliable)}")
    
    # 获取特定歌曲的匹配
    song_a_matches = result.get_matches_by_music(1)
    print(f"歌曲A出现次数: {len(song_a_matches)}")
    
    # 转换为字典
    data = result.to_dict()
    print(f"\n覆盖率: {data['stats'].get('coverage', {})}")


def example_8_performance_optimization():
    """示例8: 性能优化配置"""
    print("\n" + "="*60)
    print("示例8: 性能优化")
    print("="*60)
    
    # 高性能配置（适用于服务器环境）
    high_perf_config = AnalyzerConfig(
        window_size=10.0,
        step_size=5.0,
        match_threshold=10,
        use_parallel=True,
        max_workers=16,           # 更多线程
        batch_size=20,            # 更大批次
        skip_silence=True         # 跳过静音
    )
    
    # 高精度配置（适用于高质量要求场景）
    high_accuracy_config = AnalyzerConfig(
        window_size=5.0,          # 更小窗口
        step_size=2.5,            # 更小步长
        match_threshold=20,       # 更高阈值
        time_tolerance=1.0,       # 更严格的时间容差
        min_segment_duration=2.0, # 更短片段
        use_parallel=True,
        max_workers=4
    )
    
    print("高性能配置:")
    print(f"  线程数: {high_perf_config.max_workers}")
    print(f"  批次大小: {high_perf_config.batch_size}")
    
    print("\n高精度配置:")
    print(f"  窗口大小: {high_accuracy_config.window_size}s")
    print(f"  匹配阈值: {high_accuracy_config.match_threshold}")


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("长音频定位分析系统 - 使用示例")
    print("="*60)
    
    examples = [
        ("基础用法", example_1_basic_usage),
        ("自定义配置", example_2_custom_config),
        ("批量处理", example_3_batch_processing),
        ("数据库连接", example_4_with_database),
        ("结果导出", example_5_export_results),
        ("一键报告", example_6_analyze_with_report),
        ("结果处理", example_7_process_results),
        ("性能优化", example_8_performance_optimization),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n示例 '{name}' 执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("所有示例执行完成")
    print("="*60)


if __name__ == "__main__":
    main()
