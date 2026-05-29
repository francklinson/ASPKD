# -*- coding: utf-8 -*-
"""
结果分析与去重模块

对匹配结果进行去重、排序、时间校准和合并
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class SegmentMatch:
    """音频片段匹配结果"""
    music_id: int
    music_name: str
    start_time: float           # 在长音频中的起始时间
    end_time: float             # 在长音频中的结束时间
    confidence: int             # 匹配置信度
    match_ratio: float          # 匹配比例
    window_count: int           # 匹配的窗口数量
    is_reliable: bool           # 是否可靠
    
    def duration(self) -> float:
        """获取片段时长"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'music_id': self.music_id,
            'music_name': self.music_name,
            'start_time': round(self.start_time, 3),
            'end_time': round(self.end_time, 3),
            'duration': round(self.duration(), 3),
            'confidence': self.confidence,
            'match_ratio': round(self.match_ratio, 4),
            'window_count': self.window_count,
            'is_reliable': self.is_reliable
        }


@dataclass
class AnalysisResult:
    """完整分析结果"""
    audio_path: str
    total_duration: float
    segment_matches: List[SegmentMatch] = field(default_factory=list)
    processing_time: float = 0.0
    stats: Dict = field(default_factory=dict)
    
    def get_reliable_matches(self) -> List[SegmentMatch]:
        """获取可靠的匹配结果"""
        return [m for m in self.segment_matches if m.is_reliable]
    
    def get_matches_by_music(self, music_id: int) -> List[SegmentMatch]:
        """获取特定歌曲的所有匹配"""
        return [m for m in self.segment_matches if m.music_id == music_id]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'audio_path': self.audio_path,
            'total_duration': round(self.total_duration, 3),
            'processing_time': round(self.processing_time, 3),
            'segment_count': len(self.segment_matches),
            'reliable_count': len(self.get_reliable_matches()),
            'stats': self.stats,
            'segments': [s.to_dict() for s in self.segment_matches]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ResultAnalyzer:
    """
    结果分析器
    
    功能：
        1. 窗口匹配结果聚合
        2. 时间连续性分析
        3. 重复片段去重
        4. 结果排序和过滤
    """
    
    def __init__(self, 
                 time_tolerance: float = 2.0,    # 时间容差（秒）
                 min_segment_duration: float = 3.0,  # 最小片段时长
                 overlap_threshold: float = 0.5):  # 重叠阈值
        self.time_tolerance = time_tolerance
        self.min_segment_duration = min_segment_duration
        self.overlap_threshold = overlap_threshold
    
    def analyze(self, window_matches: List, total_duration: float) -> List[SegmentMatch]:
        """
        分析窗口匹配结果
        
        Args:
            window_matches: WindowMatch列表
            total_duration: 音频总时长
            
        Returns:
            SegmentMatch列表
        """
        # 1. 按歌曲分组
        music_groups = self._group_by_music(window_matches)
        
        # 2. 为每首歌分析时间连续性
        all_segments = []
        for music_id, matches in music_groups.items():
            segments = self._analyze_continuity(matches, music_id)
            all_segments.extend(segments)
        
        # 3. 合并重叠的片段
        merged_segments = self._merge_overlapping_segments(all_segments)
        
        # 4. 过滤短片段
        filtered_segments = [
            s for s in merged_segments 
            if s.duration() >= self.min_segment_duration
        ]
        
        # 5. 按起始时间排序
        filtered_segments.sort(key=lambda x: x.start_time)
        
        return filtered_segments
    
    def _group_by_music(self, window_matches: List) -> Dict[int, List]:
        """按歌曲ID分组窗口匹配结果"""
        groups = defaultdict(list)
        
        for wm in window_matches:
            if wm.best_match is not None:
                match = wm.best_match
                groups[match.music_id].append({
                    'window_start': wm.window_start,
                    'window_end': wm.window_end,
                    'match': match
                })
        
        return groups
    
    def _analyze_continuity(self, matches: List[Dict], 
                           music_id: int) -> List[SegmentMatch]:
        """
        分析时间连续性，将连续的窗口合并为片段
        
        Args:
            matches: 该歌曲的窗口匹配列表
            music_id: 歌曲ID
            
        Returns:
            SegmentMatch列表
        """
        if not matches:
            return []
        
        # 按窗口起始时间排序
        matches = sorted(matches, key=lambda x: x['window_start'])
        
        segments = []
        current_segment = {
            'start': matches[0]['window_start'],
            'end': matches[0]['window_end'],
            'matches': [matches[0]],
            'total_confidence': matches[0]['match'].confidence,
            'total_ratio': matches[0]['match'].match_ratio
        }
        
        for i in range(1, len(matches)):
            current_match = matches[i]
            prev_end = current_segment['end']
            curr_start = current_match['window_start']
            
            # 检查是否连续（在容差范围内）
            if curr_start <= prev_end + self.time_tolerance:
                # 连续的，扩展当前片段
                current_segment['end'] = max(current_segment['end'], 
                                             current_match['window_end'])
                current_segment['matches'].append(current_match)
                current_segment['total_confidence'] += current_match['match'].confidence
                current_segment['total_ratio'] += current_match['match'].match_ratio
            else:
                # 不连续，保存当前片段并开始新片段
                segments.append(self._create_segment_match(
                    current_segment, music_id, 
                    matches[0]['match'].music_name
                ))
                
                current_segment = {
                    'start': current_match['window_start'],
                    'end': current_match['window_end'],
                    'matches': [current_match],
                    'total_confidence': current_match['match'].confidence,
                    'total_ratio': current_match['match'].match_ratio
                }
        
        # 保存最后一个片段
        segments.append(self._create_segment_match(
            current_segment, music_id,
            matches[0]['match'].music_name
        ))
        
        return segments
    
    def _create_segment_match(self, segment_data: Dict, 
                             music_id: int, music_name: str) -> SegmentMatch:
        """创建SegmentMatch对象"""
        matches = segment_data['matches']
        window_count = len(matches)
        
        # 计算平均置信度和匹配比例
        avg_confidence = segment_data['total_confidence'] / window_count
        avg_ratio = segment_data['total_ratio'] / window_count
        
        # 判断是否可靠
        is_reliable = all(m['match'].is_reliable for m in matches)
        
        return SegmentMatch(
            music_id=music_id,
            music_name=music_name,
            start_time=segment_data['start'],
            end_time=segment_data['end'],
            confidence=int(avg_confidence),
            match_ratio=avg_ratio,
            window_count=window_count,
            is_reliable=is_reliable
        )
    
    def _merge_overlapping_segments(self, segments: List[SegmentMatch]) -> List[SegmentMatch]:
        """
        合并重叠的片段
        
        如果两个片段重叠超过阈值，保留置信度更高的
        """
        if not segments:
            return []
        
        # 按起始时间排序
        segments = sorted(segments, key=lambda x: x.start_time)
        
        merged = [segments[0]]
        
        for current in segments[1:]:
            last = merged[-1]
            
            # 计算重叠
            overlap_start = max(last.start_time, current.start_time)
            overlap_end = min(last.end_time, current.end_time)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # 计算重叠比例
            min_duration = min(last.duration(), current.duration())
            overlap_ratio = overlap_duration / min_duration if min_duration > 0 else 0
            
            if overlap_ratio >= self.overlap_threshold:
                # 有显著重叠，合并或选择更好的
                if current.music_id == last.music_id:
                    # 同一首歌，合并
                    merged[-1] = SegmentMatch(
                        music_id=last.music_id,
                        music_name=last.music_name,
                        start_time=min(last.start_time, current.start_time),
                        end_time=max(last.end_time, current.end_time),
                        confidence=max(last.confidence, current.confidence),
                        match_ratio=max(last.match_ratio, current.match_ratio),
                        window_count=last.window_count + current.window_count,
                        is_reliable=last.is_reliable and current.is_reliable
                    )
                else:
                    # 不同歌曲，选择置信度更高的
                    if current.confidence > last.confidence:
                        merged[-1] = current
            else:
                # 没有显著重叠，添加为新片段
                merged.append(current)
        
        return merged
    
    def filter_by_confidence(self, segments: List[SegmentMatch], 
                            min_confidence: int = 10) -> List[SegmentMatch]:
        """按置信度过滤"""
        return [s for s in segments if s.confidence >= min_confidence]
    
    def filter_by_duration(self, segments: List[SegmentMatch],
                          min_duration: float = 3.0) -> List[SegmentMatch]:
        """按时长过滤"""
        return [s for s in segments if s.duration() >= min_duration]
    
    def get_timeline(self, segments: List[SegmentMatch], 
                    total_duration: float,
                    time_resolution: float = 1.0) -> Dict[float, List[str]]:
        """
        生成时间线
        
        Args:
            segments: 片段列表
            total_duration: 总时长
            time_resolution: 时间分辨率（秒）
            
        Returns:
            {时间点: [歌曲名称列表]}
        """
        timeline = defaultdict(list)
        
        for t in np.arange(0, total_duration, time_resolution):
            for segment in segments:
                if segment.start_time <= t < segment.end_time:
                    timeline[round(t, 1)].append(segment.music_name)
        
        return dict(timeline)
    
    def calculate_coverage(self, segments: List[SegmentMatch],
                          total_duration: float) -> Dict:
        """
        计算音频覆盖率
        
        Args:
            segments: 片段列表
            total_duration: 总时长
            
        Returns:
            覆盖率统计
        """
        # 合并所有片段的时间范围
        covered_time = 0.0
        if segments:
            # 按起始时间排序
            sorted_segments = sorted(segments, key=lambda x: x.start_time)
            
            current_start = sorted_segments[0].start_time
            current_end = sorted_segments[0].end_time
            
            for segment in sorted_segments[1:]:
                if segment.start_time <= current_end:
                    # 重叠，扩展
                    current_end = max(current_end, segment.end_time)
                else:
                    # 不重叠，累加并重新开始
                    covered_time += current_end - current_start
                    current_start = segment.start_time
                    current_end = segment.end_time
            
            # 加上最后一个区间
            covered_time += current_end - current_start
        
        coverage_ratio = covered_time / total_duration if total_duration > 0 else 0
        
        return {
            'total_duration': total_duration,
            'covered_duration': covered_time,
            'coverage_ratio': round(coverage_ratio, 4),
            'uncovered_duration': total_duration - covered_time
        }


class DuplicateRemover:
    """
    重复片段去除器
    
    处理同一歌曲在同一位置的重复匹配
    """
    
    def __init__(self, time_tolerance: float = 1.0):
        self.time_tolerance = time_tolerance
    
    def remove_duplicates(self, segments: List[SegmentMatch]) -> List[SegmentMatch]:
        """
        去除重复片段
        
        对于同一歌曲在相近时间位置的多个匹配，保留置信度最高的
        """
        if not segments:
            return []
        
        # 按歌曲ID分组
        by_music = defaultdict(list)
        for segment in segments:
            by_music[segment.music_id].append(segment)
        
        # 为每首歌去除重复
        unique_segments = []
        for music_id, music_segments in by_music.items():
            unique = self._remove_duplicates_for_music(music_segments)
            unique_segments.extend(unique)
        
        # 按起始时间排序
        unique_segments.sort(key=lambda x: x.start_time)
        
        return unique_segments
    
    def _remove_duplicates_for_music(self, segments: List[SegmentMatch]) -> List[SegmentMatch]:
        """为单个歌曲去除重复"""
        if len(segments) <= 1:
            return segments
        
        # 按起始时间排序
        segments = sorted(segments, key=lambda x: x.start_time)
        
        unique = [segments[0]]
        
        for current in segments[1:]:
            last = unique[-1]
            
            # 检查是否是重复（时间和位置相近）
            time_diff = abs(current.start_time - last.start_time)
            
            if time_diff < self.time_tolerance:
                # 是重复，保留置信度更高的
                if current.confidence > last.confidence:
                    unique[-1] = current
            else:
                # 不是重复，添加
                unique.append(current)
        
        return unique


# ==================== 便捷函数 ====================

def quick_analyze(window_matches: List, total_duration: float) -> List[SegmentMatch]:
    """
    快速分析窗口匹配结果
    
    Args:
        window_matches: 窗口匹配结果
        total_duration: 音频总时长
        
    Returns:
        片段匹配列表
    """
    analyzer = ResultAnalyzer()
    return analyzer.analyze(window_matches, total_duration)


def merge_analysis_results(results: List[AnalysisResult]) -> AnalysisResult:
    """
    合并多个分析结果
    
    Args:
        results: 分析结果列表
        
    Returns:
        合并后的结果
    """
    if not results:
        return AnalysisResult(audio_path="", total_duration=0)
    
    all_segments = []
    total_duration = 0
    total_time = 0
    
    for result in results:
        all_segments.extend(result.segment_matches)
        total_duration = max(total_duration, result.total_duration)
        total_time += result.processing_time
    
    # 去重
    remover = DuplicateRemover()
    unique_segments = remover.remove_duplicates(all_segments)
    
    # 排序
    unique_segments.sort(key=lambda x: x.start_time)
    
    return AnalysisResult(
        audio_path=results[0].audio_path if results else "",
        total_duration=total_duration,
        segment_matches=unique_segments,
        processing_time=total_time
    )
