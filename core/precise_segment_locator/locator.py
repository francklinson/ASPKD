# -*- coding: utf-8 -*-
"""
PreciseSegmentLocator - 精确片段定位器

核心实现：基于全局指纹提取和多点时间对齐的长音频检测
"""

import numpy as np
import librosa
import hashlib
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.ndimage import maximum_filter, iterate_structure
from scipy.ndimage.morphology import generate_binary_structure
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.shazam.database.connector import MySQLConnector
from core.shazam.utils.hparam import hp


@dataclass
class SegmentInfo:
    """检测到的片段信息"""
    music_id: int
    music_name: str
    start_time: float           # 精确起始时间（秒）
    end_time: float             # 结束时间（秒）
    confidence: int             # 置信度（匹配hash数）
    match_ratio: float          # 匹配比例
    is_reliable: bool = False   # 是否可靠
    
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        return {
            'music_id': self.music_id,
            'music_name': self.music_name,
            'start_time': round(self.start_time, 3),
            'end_time': round(self.end_time, 3),
            'duration': round(self.duration(), 3),
            'confidence': self.confidence,
            'match_ratio': round(self.match_ratio, 4),
            'is_reliable': self.is_reliable
        }


@dataclass
class LocatorResult:
    """定位结果"""
    audio_path: str
    total_duration: float
    segments: List[SegmentInfo] = field(default_factory=list)
    processing_time: float = 0.0
    
    def get_by_music(self, music_id: int) -> List[SegmentInfo]:
        """获取指定音乐的所有片段"""
        return [s for s in self.segments if s.music_id == music_id]
    
    def get_reliable_segments(self) -> List[SegmentInfo]:
        """获取可靠的片段"""
        return [s for s in self.segments if s.is_reliable]


@dataclass
class SegmentLocatorConfig:
    """定位器配置"""
    # 指纹参数（与Shazam配置一致）
    sr: int = 16000
    n_fft: int = 4096
    hop_length: int = 1024
    win_length: int = 4096
    
    # 峰值检测参数
    amp_min: int = 5
    neighborhood: int = 15
    
    # hash生成参数
    near_num: int = 20
    min_time_delta: int = 0
    max_time_delta: int = 200
    
    # 匹配参数
    threshold: int = 10           # 最小匹配hash数
    min_match_ratio: float = 0.01  # 最小匹配比例（降低以捕获更多匹配）
    
    # 时间聚类参数
    time_tolerance: float = 2.0   # 时间容差（秒），用于合并相近的匹配点
    min_segment_duration: float = 3.0  # 最小片段时长
    
    # 切分参数
    segment_duration: float = 10.0  # 切分片段时长（秒）
    
    # 数据库批处理参数
    db_batch_size: int = 1000     # 数据库查询批次大小


class PreciseSegmentLocator:
    """
    精确片段定位器
    
    通过全局指纹提取和多点时间对齐，实现长音频中多参考音频的精确检测和定位。
    
    工作流程：
    1. 提取长音频完整指纹（全局，不分割窗口）
    2. 批量查询数据库（优化查询负载）
    3. 时间对齐聚类（找出所有匹配点）
    4. 精确定位（直接计算起始时间）
    5. 切分音频（固定时长）
    """
    
    def __init__(self, config: Optional[SegmentLocatorConfig] = None, 
                 db_connector: Optional[MySQLConnector] = None):
        """
        初始化定位器
        
        Args:
            config: 配置对象
            db_connector: 数据库连接器（可选）
        """
        self.config = config or SegmentLocatorConfig()
        self.db_connector = db_connector
        
        # 参考音频索引
        self.reference_index: Dict[int, Dict] = {}  # music_id -> {hashes, music_name, hash_count}
        
        if db_connector is None:
            self.db_connector = MySQLConnector()
    
    def add_reference(self, music_id: int, music_name: str = "") -> bool:
        """
        从数据库添加参考音频
        
        Args:
            music_id: 音乐ID
            music_name: 音乐名称
            
        Returns:
            是否成功
        """
        try:
            # 获取指纹
            sql = "SELECT hash, offset FROM finger_prints WHERE music_id_fk = %s"
            self.db_connector.cursor.execute(sql, (music_id,))
            fingerprints = self.db_connector.cursor.fetchall()
            
            if not fingerprints:
                print(f"[Locator] 警告: 音乐ID {music_id} 没有指纹")
                return False
            
            # 构建hash索引
            hash_index = defaultdict(list)  # hash -> [offset1, offset2, ...]
            for hash_val, offset in fingerprints:
                hash_index[hash_val].append(int(offset))
            
            self.reference_index[music_id] = {
                'music_name': music_name or self.db_connector.find_music_name_by_music_id(music_id) or f"Music_{music_id}",
                'hash_index': hash_index,
                'total_hashes': len(fingerprints),
                'unique_hashes': len(hash_index)
            }
            
            print(f"[Locator] 添加参考音频: {music_name} (ID: {music_id}, 指纹: {len(fingerprints)})")
            return True
            
        except Exception as e:
            print(f"[Locator] 添加参考音频失败 {music_id}: {e}")
            return False
    
    def add_reference_from_db(self, music_id: int, music_name: str = "") -> bool:
        """从数据库加载参考音频（别名）"""
        return self.add_reference(music_id, music_name)
    
    def clear_references(self):
        """清除所有参考音频"""
        self.reference_index.clear()
        print("[Locator] 已清除所有参考音频")
    
    def _extract_fingerprint(self, audio_path: str) -> List[Tuple[str, int]]:
        """
        提取音频指纹（全局，不分割窗口）
        
        Args:
            audio_path: 音频路径
            
        Returns:
            [(hash, offset), ...]
        """
        # 加载音频
        y, sr = librosa.load(audio_path, sr=self.config.sr)
        
        # STFT
        arr2D = librosa.stft(
            y,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        spectrogram = np.abs(arr2D)
        
        # 峰值检测
        peaks = self._get_peaks(spectrogram)
        
        # 生成hash
        hashes = list(self._generate_hashes(peaks))
        
        print(f"[Locator] 提取指纹: {len(hashes)} 个hash ({len(peaks)} 个峰值)")
        return hashes
    
    def _get_peaks(self, spectrogram: np.ndarray) -> List[Tuple[int, int]]:
        """
        频谱峰值检测（星座图）
        
        Args:
            spectrogram: 频谱图
            
        Returns:
            [(time_idx, freq_idx), ...]
        """
        # 使用最大滤波器找局部最大值（与Shazam base_processor.py:61-64 一致）
        struct = generate_binary_structure(2, 1)  # 生成基础十字架结构
        neighborhood = iterate_structure(struct, self.config.neighborhood)
        
        local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram
        
        # 应用阈值
        amps = spectrogram[local_max]
        j, i = np.where(local_max)  # j=频率索引, i=时间索引（与Shazam一致）
        
        # 过滤低能量峰值
        peaks_mask = amps > self.config.amp_min
        # 返回 (时间, 频率) 与Shazam base_processor.py:95 一致
        peaks = list(zip(i[peaks_mask], j[peaks_mask]))
        
        return peaks
    
    def _generate_hashes(self, peaks: List[Tuple[int, int]]) -> Tuple[str, int]:
        """
        从峰值生成hash
        
        Args:
            peaks: [(time_idx, freq_idx), ...]
            
        Yields:
            (hash, time_offset)
        """
        # 按时间排序
        peaks = sorted(peaks)
        
        for i in range(len(peaks)):
            for j in range(1, self.config.near_num):
                if i + j < len(peaks):
                    t1, f1 = peaks[i]
                    t2, f2 = peaks[i + j]
                    
                    t_delta = t2 - t1
                    if self.config.min_time_delta <= t_delta <= self.config.max_time_delta:
                        hash_str = f"{f1}|{f2}|{t_delta}"
                        hash_val = hashlib.sha1(hash_str.encode("utf-8")).hexdigest()
                        yield hash_val, t1
    
    def _batch_query_matches(self, hashes: List[Tuple[str, int]]) -> List[Tuple[int, int, int]]:
        """
        批量查询数据库匹配（优化数据库负载）
        
        策略：
        1. 先在内存索引中查找（如果有）
        2. 批量查询数据库（分批处理，避免一次性查询过多）
        
        Args:
            hashes: [(hash, offset), ...]
            
        Returns:
            [(music_id, db_offset, query_offset), ...]
        """
        matches = []
        
        # 方法1：内存索引查询（如果有参考音频索引）
        if self.reference_index:
            for hash_val, query_offset in hashes:
                for music_id, ref_data in self.reference_index.items():
                    if hash_val in ref_data['hash_index']:
                        for db_offset in ref_data['hash_index'][hash_val]:
                            matches.append((music_id, db_offset, query_offset))
            
            if matches:
                print(f"[Locator] 内存索引匹配: {len(matches)} 个")
                return matches
        
        # 方法2：数据库批量查询（分批处理）
        hash_values = [h for h, _ in hashes]
        hash_to_offset = {h: o for h, o in hashes}
        
        # 分批查询，避免单次查询过大
        for i in range(0, len(hash_values), self.config.db_batch_size):
            batch = hash_values[i:i + self.config.db_batch_size]
            
            # 构建IN查询
            placeholders = ', '.join(['%s'] * len(batch))
            sql = f"""
                SELECT music_id_fk, hash, offset 
                FROM finger_prints 
                WHERE hash IN ({placeholders})
            """
            
            self.db_connector.cursor.execute(sql, tuple(batch))
            results = self.db_connector.cursor.fetchall()
            
            for music_id, hash_val, db_offset in results:
                if hash_val in hash_to_offset:
                    query_offset = hash_to_offset[hash_val]
                    matches.append((music_id, int(db_offset), query_offset))
        
        print(f"[Locator] 数据库匹配: {len(matches)} 个")
        return matches
    
    def _time_alignment_clustering(self, matches: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], int]:
        """
        时间对齐聚类
        
        核心算法：
        offset_diff = db_offset - query_offset
        相同的 (music_id, offset_diff) 表示同一个匹配位置
        
        Args:
            matches: [(music_id, db_offset, query_offset), ...]
            
        Returns:
            {(music_id, offset_diff): confidence, ...}
        """
        vote_map = defaultdict(int)
        
        for music_id, db_offset, query_offset in matches:
            offset_diff = db_offset - query_offset
            vote_map[(music_id, offset_diff)] += 1
        
        return dict(vote_map)
    
    def _find_all_match_points(self, vote_map: Dict[Tuple[int, int], int]) -> List[Dict]:
        """
        找出所有有效的匹配点
        
        不只找最佳匹配，找所有超过阈值的匹配
        
        Args:
            vote_map: {(music_id, offset_diff): confidence}
            
        Returns:
            [{'music_id': int, 'offset_diff': int, 'confidence': int}, ...]
        """
        candidates = []
        
        # 调试：打印投票映射统计
        print(f"[Locator] 投票映射统计: 共 {len(vote_map)} 个 (music_id, offset_diff) 组合")
        
        # 按 music_id 分组统计
        music_stats = defaultdict(list)
        for (music_id, offset_diff), confidence in vote_map.items():
            music_stats[music_id].append((offset_diff, confidence))
        
        for music_id, offsets in music_stats.items():
            music_name = "Unknown"
            if music_id in self.reference_index:
                music_name = self.reference_index[music_id]['music_name']
            else:
                try:
                    music_name = self.db_connector.find_music_name_by_music_id(music_id) or f"Music_{music_id}"
                except:
                    pass
            # 找出该音乐的最高置信度
            best_conf = max(c for _, c in offsets)
            print(f"[Locator]   音乐 {music_id} ({music_name}): {len(offsets)} 个偏移点, 最高置信度: {best_conf}")
        
        for (music_id, offset_diff), confidence in vote_map.items():
            # 获取参考音频信息
            if music_id in self.reference_index:
                # 使用内存索引中的数据
                ref_data = self.reference_index[music_id]
                music_name = ref_data['music_name']
                total_hashes = ref_data['total_hashes']
            else:
                # 从数据库获取信息
                try:
                    music_name = self.db_connector.find_music_name_by_music_id(music_id) or f"Music_{music_id}"
                    # 查询该音乐的总指纹数
                    self.db_connector.cursor.execute(
                        "SELECT COUNT(*) FROM finger_prints WHERE music_id_fk = %s", (music_id,)
                    )
                    total_hashes = self.db_connector.cursor.fetchone()[0]
                except Exception as e:
                    print(f"[Locator] 获取音乐 {music_id} 信息失败: {e}")
                    continue
            
            match_ratio = confidence / max(total_hashes, 1)
            
            # 检查阈值
            if confidence >= self.config.threshold and match_ratio >= self.config.min_match_ratio:
                candidates.append({
                    'music_id': music_id,
                    'music_name': music_name,
                    'offset_diff': offset_diff,
                    'confidence': confidence,
                    'match_ratio': match_ratio
                })
        
        # 按置信度排序
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"[Locator] 找到 {len(candidates)} 个候选匹配点 (阈值: confidence>={self.config.threshold}, ratio>={self.config.min_match_ratio})")
        for c in candidates[:5]:  # 打印前5个
            print(f"[Locator]   候选: {c['music_name']} offset={c['offset_diff']} conf={c['confidence']} ratio={c['match_ratio']:.4f}")
        
        return candidates
    
    def _merge_nearby_matches(self, candidates: List[Dict]) -> List[SegmentInfo]:
        """
        合并相近的匹配点
        
        同一个片段可能有多个相近的匹配点，需要合并
        
        Args:
            candidates: 候选匹配点列表
            
        Returns:
            合并后的片段列表
        """
        if not candidates:
            return []
        
        # 按music_id分组
        by_music = defaultdict(list)
        for c in candidates:
            by_music[c['music_id']].append(c)
        
        segments = []
        
        for music_id, music_candidates in by_music.items():
            # 转换为时间
            for c in music_candidates:
                c['start_time'] = c['offset_diff'] * self.config.hop_length / self.config.sr
            
            # 按时间排序
            music_candidates.sort(key=lambda x: x['start_time'])
            
            # 合并相近的匹配点
            current_group = [music_candidates[0]]
            
            for i in range(1, len(music_candidates)):
                curr = music_candidates[i]
                prev_start = current_group[-1]['start_time']
                
                # 如果时间上足够接近，合并到同一组
                if abs(curr['start_time'] - prev_start) <= self.config.time_tolerance:
                    current_group.append(curr)
                else:
                    # 创建片段
                    segment = self._create_segment_from_group(current_group)
                    if segment.duration() >= self.config.min_segment_duration:
                        segments.append(segment)
                    
                    current_group = [curr]
            
            # 处理最后一组
            if current_group:
                segment = self._create_segment_from_group(current_group)
                if segment.duration() >= self.config.min_segment_duration:
                    segments.append(segment)
        
        # 按起始时间排序
        segments.sort(key=lambda x: x.start_time)
        
        return segments
    
    def _create_segment_from_group(self, group: List[Dict]) -> SegmentInfo:
        """从匹配组创建片段"""
        # 选择置信度最高的作为代表
        best = max(group, key=lambda x: x['confidence'])
        
        # 计算时间范围
        start_time = best['start_time']
        end_time = start_time + self.config.segment_duration
        
        # 判断是否可靠
        is_reliable = best['confidence'] >= self.config.threshold * 2
        
        return SegmentInfo(
            music_id=best['music_id'],
            music_name=best['music_name'],
            start_time=start_time,
            end_time=end_time,
            confidence=best['confidence'],
            match_ratio=best['match_ratio'],
            is_reliable=is_reliable
        )
    
    def locate_segments(self, audio_path: str) -> LocatorResult:
        """
        定位长音频中的所有片段
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            LocatorResult
        """
        import time
        start_time = time.time()
        
        print(f"[Locator] 开始分析: {audio_path}")
        
        # 1. 获取音频时长
        total_duration = librosa.get_duration(path=audio_path, sr=self.config.sr)
        
        # 2. 提取完整指纹（全局，不分割窗口）
        print("[Locator] 步骤1: 提取全局指纹...")
        hashes = self._extract_fingerprint(audio_path)
        
        if not hashes:
            print("[Locator] 警告: 未提取到指纹")
            return LocatorResult(audio_path=audio_path, total_duration=total_duration)
        
        # 3. 批量查询数据库（优化负载）
        print("[Locator] 步骤2: 批量查询匹配...")
        matches = self._batch_query_matches(hashes)
        
        if not matches:
            print("[Locator] 警告: 未找到匹配")
            return LocatorResult(audio_path=audio_path, total_duration=total_duration)
        
        # 4. 时间对齐聚类
        print("[Locator] 步骤3: 时间对齐聚类...")
        vote_map = self._time_alignment_clustering(matches)
        
        # 5. 找出所有匹配点
        print("[Locator] 步骤4: 找出所有匹配点...")
        candidates = self._find_all_match_points(vote_map)
        
        # 6. 合并相近匹配点
        print("[Locator] 步骤5: 合并相近匹配点...")
        segments = self._merge_nearby_matches(candidates)
        
        processing_time = time.time() - start_time
        
        print(f"[Locator] 分析完成: 找到 {len(segments)} 个片段, 耗时 {processing_time:.2f}s")
        for seg in segments:
            print(f"  - {seg.music_name}: {seg.start_time:.2f}s - {seg.end_time:.2f}s (置信度: {seg.confidence})")
        
        return LocatorResult(
            audio_path=audio_path,
            total_duration=total_duration,
            segments=segments,
            processing_time=processing_time
        )
    
    def close(self):
        """释放资源"""
        if self.db_connector:
            self.db_connector.close()
            print("[Locator] 数据库连接已关闭")
