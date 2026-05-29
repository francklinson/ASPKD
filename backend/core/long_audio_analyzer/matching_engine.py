# -*- coding: utf-8 -*-
"""
高效指纹比对引擎

采用倒排索引和批量查询优化，支持快速匹配
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class MatchResult:
    """匹配结果"""
    music_id: int               # 匹配的歌曲ID
    music_name: str             # 歌曲名称
    confidence: int             # 置信度（匹配哈希数）
    offset: float               # 时间偏移（秒）
    offset_frames: int          # 时间偏移（帧）
    match_ratio: float          # 匹配比例
    is_reliable: bool           # 是否可靠


@dataclass
class WindowMatch:
    """窗口匹配结果"""
    window_id: int              # 窗口ID
    window_start: float         # 窗口起始时间
    window_end: float           # 窗口结束时间
    matches: List[MatchResult]  # 匹配结果列表
    best_match: Optional[MatchResult] = None  # 最佳匹配


class InvertedIndex:
    """
    倒排索引
    
    用于快速指纹查找，将哈希映射到 (music_id, offset) 列表
    """
    
    def __init__(self):
        # hash -> [(music_id, offset), ...]
        self.index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.music_info: Dict[int, str] = {}  # music_id -> music_name
        self.total_hashes: Dict[int, int] = {}  # music_id -> hash_count
        
    def add_fingerprint(self, music_id: int, hashes: List[Tuple[str, int]], 
                        music_name: str = ""):
        """
        添加指纹到索引
        
        Args:
            music_id: 歌曲ID
            hashes: [(hash, offset), ...]
            music_name: 歌曲名称
        """
        self.music_info[music_id] = music_name
        self.total_hashes[music_id] = len(hashes)
        
        for hash_val, offset in hashes:
            self.index[hash_val].append((music_id, offset))
    
    def batch_query(self, hashes: List[Tuple[str, int]]) -> List[Tuple[int, int, int]]:
        """
        批量查询哈希
        
        Args:
            hashes: [(hash, query_offset), ...]
            
        Returns:
            [(music_id, db_offset, query_offset), ...]
        """
        results = []
        hash_set = set(h[0] for h in hashes)
        
        # 构建查询偏移映射
        query_offsets = {h: offset for h, offset in hashes}
        
        # 批量查询
        for hash_val in hash_set:
            if hash_val in self.index:
                for music_id, db_offset in self.index[hash_val]:
                    results.append((music_id, db_offset, query_offsets[hash_val]))
        
        return results
    
    def get_music_name(self, music_id: int) -> str:
        """获取歌曲名称"""
        return self.music_info.get(music_id, f"Unknown_{music_id}")
    
    def get_hash_count(self, music_id: int) -> int:
        """获取歌曲哈希数量"""
        return self.total_hashes.get(music_id, 0)


class FastMatchingEngine:
    """
    快速指纹比对引擎
    
    功能：
        1. 基于倒排索引的快速匹配
        2. 时间对齐投票算法
        3. 置信度计算和可靠性评估
        4. 支持批量匹配
    """
    
    def __init__(self, threshold: int = 10, min_match_ratio: float = 0.05):
        """
        初始化匹配引擎
        
        Args:
            threshold: 匹配阈值（最小匹配哈希数）
            min_match_ratio: 最小匹配比例
        """
        self.index = InvertedIndex()
        self.threshold = threshold
        self.min_match_ratio = min_match_ratio
        
        # 性能统计
        self.stats = {
            'total_queries': 0,
            'successful_matches': 0,
            'avg_query_time': 0.0
        }
        
    def build_index_from_database(self, connector):
        """
        从数据库构建倒排索引
        
        Args:
            connector: 数据库连接对象
        """
        # 获取所有歌曲
        sql = "SELECT music_id, music_name FROM music"
        connector.cursor.execute(sql)
        musics = connector.cursor.fetchall()
        
        for music_id, music_name in musics:
            # 获取该歌曲的所有指纹
            sql = "SELECT hash, offset FROM finger_prints WHERE music_id_fk = %s"
            connector.cursor.execute(sql, (music_id,))
            fingerprints = connector.cursor.fetchall()
            
            hashes = [(h, int(o)) for h, o in fingerprints]
            self.index.add_fingerprint(music_id, hashes, music_name)
        
        print(f"索引构建完成，共 {len(musics)} 首歌曲，{len(self.index.index)} 个唯一哈希")
    
    def add_reference(self, music_id: int, hashes: List[Tuple[str, int]], 
                      music_name: str = ""):
        """
        添加参考指纹
        
        Args:
            music_id: 歌曲ID
            hashes: 指纹列表
            music_name: 歌曲名称
        """
        self.index.add_fingerprint(music_id, hashes, music_name)
    
    def match(self, hashes: List[Tuple[str, int]], 
              query_hash_count: int = 0) -> Optional[MatchResult]:
        """
        执行指纹匹配
        
        Args:
            hashes: 查询指纹 [(hash, offset), ...]
            query_hash_count: 查询指纹总数（用于计算匹配比例）
            
        Returns:
            MatchResult或None
        """
        if not hashes:
            return None
        
        # 批量查询
        matches = self.index.batch_query(hashes)
        
        if not matches:
            return None
        
        # 时间对齐投票
        result = self._align_voting(matches)
        
        if result is None:
            return None
        
        music_id, offset_frames, confidence = result
        
        # 计算匹配比例
        total_db_hashes = self.index.get_hash_count(music_id)
        match_ratio = confidence / max(total_db_hashes, query_hash_count, 1)
        
        # 判断是否可靠
        is_reliable = (confidence >= self.threshold and 
                       match_ratio >= self.min_match_ratio)
        
        # 转换为秒 - 使用与Shazam配置一致的参数
        try:
            from shazam.utils.hparam import hp
            hop_length = hp.fingerprint.core.stft.hop_length
            sr = hp.fingerprint.core.stft.sr
        except ImportError:
            hop_length = 1024
            sr = 16000
        offset_seconds = offset_frames * hop_length / sr
        
        return MatchResult(
            music_id=music_id,
            music_name=self.index.get_music_name(music_id),
            confidence=confidence,
            offset=offset_seconds,
            offset_frames=offset_frames,
            match_ratio=match_ratio,
            is_reliable=is_reliable
        )
    
    def _align_voting(self, matches: List[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
        """
        时间对齐投票算法
        
        Args:
            matches: [(music_id, db_offset, query_offset), ...]
            
        Returns:
            (music_id, offset, confidence) 或 None
        """
        # 计算偏移差值
        # offset_diff = db_offset - query_offset
        vote_map = defaultdict(lambda: defaultdict(int))
        
        for music_id, db_offset, query_offset in matches:
            offset_diff = db_offset - query_offset
            vote_map[music_id][offset_diff] += 1
        
        # 找出投票最多的 (music_id, offset)
        best_music_id = -1
        best_offset = 0
        best_confidence = 0
        
        for music_id, offset_votes in vote_map.items():
            for offset_diff, count in offset_votes.items():
                if count > best_confidence:
                    best_confidence = count
                    best_offset = offset_diff
                    best_music_id = music_id
        
        if best_confidence < self.threshold:
            return None
        
        return best_music_id, best_offset, best_confidence
    
    def match_window(self, window_fingerprint) -> WindowMatch:
        """
        匹配单个窗口
        
        Args:
            window_fingerprint: WindowFingerprint对象
            
        Returns:
            WindowMatch对象
        """
        from .fingerprint_extractor import WindowFingerprint
        
        # 获取所有可能的匹配
        matches = self.index.batch_query(window_fingerprint.hashes)
        
        # 按歌曲分组投票
        vote_map = defaultdict(lambda: defaultdict(int))
        for music_id, db_offset, query_offset in matches:
            offset_diff = db_offset - query_offset
            vote_map[music_id][offset_diff] += 1
        
        # 为每首歌找出最佳匹配
        match_results = []
        for music_id, offset_votes in vote_map.items():
            best_offset = max(offset_votes.items(), key=lambda x: x[1])
            offset_frames, confidence = best_offset
            
            if confidence >= self.threshold:
                total_db_hashes = self.index.get_hash_count(music_id)
                match_ratio = confidence / max(total_db_hashes, window_fingerprint.hash_count, 1)
                
                # 使用与Shazam配置一致的参数
                try:
                    from shazam.utils.hparam import hp
                    hop_length = hp.fingerprint.core.stft.hop_length
                    sr = hp.fingerprint.core.stft.sr
                except ImportError:
                    hop_length = 1024
                    sr = 16000
                offset_seconds = offset_frames * hop_length / sr
                
                match_results.append(MatchResult(
                    music_id=music_id,
                    music_name=self.index.get_music_name(music_id),
                    confidence=confidence,
                    offset=offset_seconds,
                    offset_frames=offset_frames,
                    match_ratio=match_ratio,
                    is_reliable=match_ratio >= self.min_match_ratio
                ))
        
        # 按置信度排序
        match_results.sort(key=lambda x: x.confidence, reverse=True)
        
        best_match = match_results[0] if match_results else None
        
        return WindowMatch(
            window_id=window_fingerprint.window_id,
            window_start=window_fingerprint.start_time,
            window_end=window_fingerprint.end_time,
            matches=match_results,
            best_match=best_match
        )
    
    def match_batch(self, window_fingerprints: List) -> List[WindowMatch]:
        """
        批量匹配窗口
        
        Args:
            window_fingerprints: WindowFingerprint列表
            
        Returns:
            WindowMatch列表
        """
        results = []
        for window_fp in window_fingerprints:
            match = self.match_window(window_fp)
            results.append(match)
        return results
    
    def get_index_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            'total_songs': len(self.index.music_info),
            'total_unique_hashes': len(self.index.index),
            'avg_hashes_per_song': np.mean(list(self.index.total_hashes.values())) if self.index.total_hashes else 0
        }


class AdaptiveThresholdMatcher(FastMatchingEngine):
    """
    自适应阈值匹配器
    
    根据音频质量和内容动态调整匹配阈值
    """
    
    def __init__(self, base_threshold: int = 10, min_match_ratio: float = 0.05):
        super().__init__(base_threshold, min_match_ratio)
        self.base_threshold = base_threshold
        
    def calculate_adaptive_threshold(self, hash_count: int, 
                                     audio_quality: str = "medium") -> int:
        """
        计算自适应阈值
        
        Args:
            hash_count: 查询指纹数量
            audio_quality: 音频质量 (high/medium/low/noisy)
            
        Returns:
            自适应阈值
        """
        # 基础阈值
        threshold = self.base_threshold
        
        # 根据指纹数量调整
        if hash_count < 50:
            threshold = max(5, int(threshold * 0.5))
        elif hash_count > 200:
            threshold = int(threshold * 1.5)
        
        # 根据音频质量调整
        quality_factors = {
            "high": 1.0,
            "medium": 0.8,
            "low": 0.6,
            "noisy": 0.5
        }
        factor = quality_factors.get(audio_quality, 0.8)
        threshold = int(threshold * factor)
        
        return max(threshold, 3)  # 最小阈值为3
    
    def match_with_quality(self, hashes: List[Tuple[str, int]], 
                          audio_quality: str = "medium") -> Optional[MatchResult]:
        """
        根据音频质量执行匹配
        
        Args:
            hashes: 查询指纹
            audio_quality: 音频质量
            
        Returns:
            MatchResult或None
        """
        # 计算自适应阈值
        adaptive_threshold = self.calculate_adaptive_threshold(
            len(hashes), audio_quality
        )
        
        # 临时修改阈值
        original_threshold = self.threshold
        self.threshold = adaptive_threshold
        
        try:
            result = self.match(hashes, len(hashes))
        finally:
            self.threshold = original_threshold
        
        return result


# ==================== 便捷函数 ====================

def create_matching_engine_from_db(connector, threshold: int = 10) -> FastMatchingEngine:
    """
    从数据库创建匹配引擎
    
    Args:
        connector: 数据库连接
        threshold: 匹配阈值
        
    Returns:
        FastMatchingEngine实例
    """
    engine = FastMatchingEngine(threshold=threshold)
    engine.build_index_from_database(connector)
    return engine
