# -*- coding: utf-8 -*-
"""
长音频定位分析器 - 主模块

整合所有组件，提供完整的长音频分析功能
"""

import os
import time
import json
from typing import List, Optional, Dict, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np

from .audio_processor import AudioPreprocessor
from .fingerprint_extractor import (
    SlidingWindowFingerprintExtractor, 
    FingerprintConfig,
    WindowFingerprint
)
from .matching_engine import FastMatchingEngine, MatchResult
from .result_analyzer import ResultAnalyzer, AnalysisResult, SegmentMatch

# 尝试导入shazam的数据库连接器
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shazam.database.connector import MySQLConnector
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False


@dataclass
class AnalyzerConfig:
    """分析器配置"""
    # 窗口参数
    window_size: float = 10.0       # 窗口大小（秒）
    step_size: float = 5.0          # 步长（秒）
    
    # 匹配参数
    match_threshold: int = 10       # 匹配阈值
    min_match_ratio: float = 0.05   # 最小匹配比例
    
    # 分析参数
    time_tolerance: float = 2.0     # 时间容差（秒）
    min_segment_duration: float = 3.0  # 最小片段时长
    
    # 性能参数
    use_parallel: bool = True       # 是否使用并行处理
    max_workers: int = 4            # 最大线程数
    batch_size: int = 10            # 批处理大小
    
    # 质量参数
    skip_silence: bool = True       # 是否跳过静音区域
    min_audio_quality: str = "low"  # 最低音频质量


class LongAudioAnalyzer:
    """
    长音频定位分析器
    
    主要功能：
    1. 长音频的滑动窗口指纹提取
    2. 与指纹数据库的高效匹配
    3. 时间定位和片段识别
    4. 结果分析和报告生成
    
    性能指标：
    - 时间定位精度：误差不超过500ms
    - 识别准确率：SNR≥10dB时，准确率≥95%
    - 处理速度：单线程每小时音频处理时间不超过10分钟
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None, 
                 db_connector=None):
        """
        初始化分析器
        
        Args:
            config: 分析器配置
            db_connector: 数据库连接对象（可选）
        """
        self.config = config or AnalyzerConfig()
        self.db_connector = db_connector
        
        # 初始化组件
        self.audio_processor = AudioPreprocessor()
        
        fp_config = FingerprintConfig(
            window_size=self.config.window_size,
            step_size=self.config.step_size
        )
        self.fingerprint_extractor = SlidingWindowFingerprintExtractor(fp_config)
        
        self.matching_engine = FastMatchingEngine(
            threshold=self.config.match_threshold,
            min_match_ratio=self.config.min_match_ratio
        )
        
        self.result_analyzer = ResultAnalyzer(
            time_tolerance=self.config.time_tolerance,
            min_segment_duration=self.config.min_segment_duration
        )
        
        # 如果有数据库连接，构建索引
        if db_connector is not None:
            self._build_index_from_db()
        
        # 线程本地存储
        self._thread_local = threading.local()
    
    def _build_index_from_db(self):
        """从数据库构建索引"""
        if self.db_connector is not None:
            self.matching_engine.build_index_from_database(self.db_connector)
    
    def add_reference(self, audio_path: str, music_id: int, 
                      music_name: str = "") -> bool:
        """
        添加参考音频到索引
        
        Args:
            audio_path: 音频文件路径
            music_id: 歌曲ID
            music_name: 歌曲名称
            
        Returns:
            是否成功
        """
        try:
            # 预处理
            result = self.audio_processor.preprocess_for_fingerprint(audio_path)
            if not result['is_valid']:
                return False
            
            # 提取指纹
            window_fps = self.fingerprint_extractor.extract(
                result['audio'], 
                result['sample_rate']
            )
            
            # 合并所有窗口的指纹
            all_hashes = []
            for window_fp in window_fps:
                all_hashes.extend(window_fp.hashes)
            
            # 添加到索引
            self.matching_engine.add_reference(music_id, all_hashes, music_name)
            
            return True
            
        except Exception as e:
            print(f"添加参考音频失败: {e}")
            return False
    
    def analyze(self, audio_path: str, 
                progress_callback: Optional[Callable[[str, float], None]] = None
                ) -> AnalysisResult:
        """
        分析长音频文件
        
        Args:
            audio_path: 音频文件路径
            progress_callback: 进度回调函数 (message, progress)
            
        Returns:
            AnalysisResult: 分析结果
        """
        start_time = time.time()
        
        def update_progress(message: str, progress: float):
            if progress_callback:
                progress_callback(message, progress)
        
        # 1. 获取音频信息
        update_progress("获取音频信息...", 0.05)
        total_duration = self.audio_processor.get_audio_duration(audio_path)
        
        # 2. 预处理音频
        update_progress("预处理音频...", 0.10)
        preprocess_result = self.audio_processor.preprocess_for_fingerprint(audio_path)
        
        if not preprocess_result['is_valid']:
            return AnalysisResult(
                audio_path=audio_path,
                total_duration=total_duration,
                stats={'error': '音频质量不符合要求'}
            )
        
        # 3. 提取指纹
        update_progress("提取音频指纹...", 0.25)
        if self.config.skip_silence:
            window_fps = self.fingerprint_extractor.extract_with_silence_skip(
                preprocess_result['audio'],
                preprocess_result['sample_rate'],
                preprocess_result['silence_regions']
            )
        else:
            window_fps = self.fingerprint_extractor.extract(
                preprocess_result['audio'],
                preprocess_result['sample_rate']
            )
        
        if not window_fps:
            return AnalysisResult(
                audio_path=audio_path,
                total_duration=total_duration,
                stats={'error': '未能提取到有效指纹'}
            )
        
        # 4. 匹配指纹
        update_progress("匹配指纹...", 0.50)
        if self.config.use_parallel and len(window_fps) > 1:
            window_matches = self._parallel_match(window_fps, update_progress)
        else:
            window_matches = self._sequential_match(window_fps, update_progress)
        
        # 5. 分析结果
        update_progress("分析匹配结果...", 0.90)
        segment_matches = self.result_analyzer.analyze(window_matches, total_duration)
        
        # 6. 计算统计信息
        coverage = self.result_analyzer.calculate_coverage(segment_matches, total_duration)
        
        processing_time = time.time() - start_time
        
        update_progress("分析完成", 1.0)
        
        # 构建结果
        stats = {
            'window_count': len(window_fps),
            'match_count': len([w for w in window_matches if w.best_match]),
            'coverage': coverage,
            'audio_quality': preprocess_result['segment_info'].quality.value,
            'snr_db': preprocess_result['segment_info'].snr_db
        }
        
        return AnalysisResult(
            audio_path=audio_path,
            total_duration=total_duration,
            segment_matches=segment_matches,
            processing_time=processing_time,
            stats=stats
        )
    
    def _sequential_match(self, window_fps: List[WindowFingerprint],
                         progress_callback: Optional[Callable] = None) -> List:
        """串行匹配"""
        window_matches = []
        total = len(window_fps)
        
        for i, window_fp in enumerate(window_fps):
            match = self.matching_engine.match_window(window_fp)
            window_matches.append(match)
            
            if progress_callback and i % 10 == 0:
                progress = 0.50 + (i / total) * 0.35
                progress_callback(f"匹配窗口 {i+1}/{total}...", progress)
        
        return window_matches
    
    def _parallel_match(self, window_fps: List[WindowFingerprint],
                       progress_callback: Optional[Callable] = None) -> List:
        """并行匹配"""
        window_matches = [None] * len(window_fps)
        
        def match_single(args):
            idx, window_fp = args
            return idx, self.matching_engine.match_window(window_fp)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(match_single, (i, fp)): i 
                for i, fp in enumerate(window_fps)
            }
            
            completed = 0
            total = len(window_fps)
            
            for future in as_completed(futures):
                idx, match = future.result()
                window_matches[idx] = match
                completed += 1
                
                if progress_callback and completed % 10 == 0:
                    progress = 0.50 + (completed / total) * 0.35
                    progress_callback(f"匹配窗口 {completed}/{total}...", progress)
        
        return window_matches
    
    def analyze_batch(self, audio_paths: List[str],
                     progress_callback: Optional[Callable[[str, float], None]] = None
                     ) -> List[AnalysisResult]:
        """
        批量分析多个音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            progress_callback: 进度回调函数
            
        Returns:
            AnalysisResult列表
        """
        results = []
        total = len(audio_paths)
        
        for i, path in enumerate(audio_paths):
            def file_progress(msg: str, prog: float):
                if progress_callback:
                    overall_progress = (i + prog) / total
                    progress_callback(f"[{i+1}/{total}] {msg}", overall_progress)
            
            result = self.analyze(path, file_progress)
            results.append(result)
        
        return results
    
    def export_json(self, result: AnalysisResult, output_path: str):
        """
        导出结果为JSON文件
        
        Args:
            result: 分析结果
            output_path: 输出文件路径
        """
        import numpy as np
        
        def convert_to_serializable(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(result.to_dict()), f, indent=2, ensure_ascii=False)
    
    def export_csv(self, result: AnalysisResult, output_path: str):
        """
        导出结果为CSV文件
        
        Args:
            result: 分析结果
            output_path: 输出文件路径
        """
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Music ID', 'Music Name', 'Start Time (s)', 'End Time (s)',
                'Duration (s)', 'Confidence', 'Match Ratio', 'Reliable'
            ])
            
            for segment in result.segment_matches:
                writer.writerow([
                    segment.music_id,
                    segment.music_name,
                    segment.start_time,
                    segment.end_time,
                    segment.duration(),
                    segment.confidence,
                    segment.match_ratio,
                    segment.is_reliable
                ])
    
    def visualize_timeline(self, result: AnalysisResult, output_path: str,
                          time_resolution: float = 1.0):
        """
        可视化时间线
        
        Args:
            result: 分析结果
            output_path: 输出图片路径
            time_resolution: 时间分辨率
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # 获取唯一歌曲列表
            music_names = list(set(s.music_name for s in result.segment_matches))
            colors = plt.cm.tab20(np.linspace(0, 1, len(music_names)))
            color_map = dict(zip(music_names, colors))
            
            # 绘制每个片段
            for i, segment in enumerate(result.segment_matches):
                color = color_map[segment.music_name]
                alpha = 0.8 if segment.is_reliable else 0.4
                
                ax.barh(
                    i,
                    segment.duration(),
                    left=segment.start_time,
                    height=0.6,
                    color=color,
                    alpha=alpha,
                    edgecolor='black',
                    linewidth=0.5
                )
                
                # 添加标签
                if segment.duration() > 5:
                    ax.text(
                        segment.start_time + segment.duration() / 2,
                        i,
                        segment.music_name[:20],
                        ha='center',
                        va='center',
                        fontsize=8
                    )
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Match Segments')
            ax.set_title(f'Audio Timeline - {os.path.basename(result.audio_path)}')
            ax.set_xlim(0, result.total_duration)
            ax.grid(True, alpha=0.3)
            
            # 添加图例
            patches = [mpatches.Patch(color=color_map[name], label=name[:30]) 
                      for name in music_names[:10]]  # 只显示前10个
            ax.legend(handles=patches, loc='upper right', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("matplotlib未安装，无法生成可视化")
        except Exception as e:
            print(f"可视化生成失败: {e}")
    
    def print_report(self, result: AnalysisResult):
        """
        打印分析报告
        
        Args:
            result: 分析结果
        """
        print("\n" + "="*60)
        print("长音频定位分析报告")
        print("="*60)
        print(f"音频文件: {result.audio_path}")
        print(f"总时长: {result.total_duration:.2f}秒")
        print(f"处理时间: {result.processing_time:.2f}秒")
        print(f"\n发现 {len(result.segment_matches)} 个匹配片段:")
        print("-"*60)
        
        for i, segment in enumerate(result.segment_matches, 1):
            status = "✓" if segment.is_reliable else "?"
            print(f"{status} [{i}] {segment.music_name}")
            print(f"    时间: {segment.start_time:.2f}s - {segment.end_time:.2f}s "
                  f"({segment.duration():.2f}s)")
            print(f"    置信度: {segment.confidence}, 匹配比例: {segment.match_ratio:.2%}")
        
        print("="*60)


# ==================== 便捷函数 ====================

def quick_analyze(audio_path: str, db_connector=None) -> AnalysisResult:
    """
    快速分析音频
    
    Args:
        audio_path: 音频路径
        db_connector: 数据库连接（可选）
        
    Returns:
        分析结果
    """
    analyzer = LongAudioAnalyzer(db_connector=db_connector)
    return analyzer.analyze(audio_path)


def analyze_with_report(audio_path: str, output_dir: str, db_connector=None):
    """
    分析音频并生成完整报告
    
    Args:
        audio_path: 音频路径
        output_dir: 输出目录
        db_connector: 数据库连接（可选）
    """
    # 分析
    analyzer = LongAudioAnalyzer(db_connector=db_connector)
    result = analyzer.analyze(audio_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 导出JSON
    json_path = os.path.join(output_dir, f"{base_name}_report.json")
    analyzer.export_json(result, json_path)
    
    # 导出CSV
    csv_path = os.path.join(output_dir, f"{base_name}_report.csv")
    analyzer.export_csv(result, csv_path)
    
    # 生成可视化
    viz_path = os.path.join(output_dir, f"{base_name}_timeline.png")
    analyzer.visualize_timeline(result, viz_path)
    
    # 打印报告
    analyzer.print_report(result)
    
    print(f"\n报告已保存到: {output_dir}")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")
    print(f"  - 可视化: {viz_path}")
