# -*- coding: utf-8 -*-
"""
Shazam 音频指纹识别 - 统一外部调用接口

提供简单易用的API用于：
- 音频指纹创建和管理
- 音频识别和匹配
- 音频片段定位
"""

import os
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

from .core.implementations.stft.stft_create import StftMusicProcessorCreate
from .core.implementations.stft.stft_predict import StftMusicProcessorPredict
from .database.connector import MySQLConnector, DatabaseChecker
from .utils.hparam import Hparam, hp


@dataclass
class RecognitionResult:
    """音频识别结果"""
    music_id: int           # 匹配到的音乐ID
    name: str               # 音乐名称
    offset: float           # 时间偏移量（秒）
    confidence: int         # 置信度（匹配哈希数量）
    matched: bool           # 是否匹配成功

    def __repr__(self):
        status = "匹配成功" if self.matched else "未匹配"
        return f"RecognitionResult({status}: {self.name}, offset={self.offset:.2f}s, confidence={self.confidence})"


@dataclass
class LocationResult:
    """音频定位结果"""
    found: bool             # 是否找到
    start_time: float       # 起始时间（秒）
    end_time: float         # 结束时间（秒）
    confidence: int         # 置信度
    music_id: int = -1      # 匹配到的音乐ID
    music_name: str = ""    # 音乐名称

    def __repr__(self):
        if self.found:
            return f"LocationResult(found={self.found}, position={self.start_time:.2f}s~{self.end_time:.2f}s, confidence={self.confidence})"
        return "LocationResult(found=False)"


class AudioFingerprinter:
    """
    音频指纹识别器 - 统一调用接口

    用法:
        >>> fingerprinter = AudioFingerprinter()
        >>>
        >>> # 添加参考音频
        >>> fingerprinter.add_reference("song1.wav", name="歌曲1")
        >>>
        >>> # 识别查询音频
        >>> result = fingerprinter.recognize("query.wav")
        >>> print(result.name, result.offset)
        >>>
        >>> # 定位片段位置
        >>> loc = fingerprinter.locate("long_audio.wav", reference_name="歌曲1")
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化指纹识别器

        Args:
            config_path: 配置文件路径，默认使用 config/config.yaml
        """
        # 加载配置
        if config_path:
            self.hp = Hparam(config_path)
        else:
            from .utils.hparam import hp
            self.hp = hp

        # 初始化数据库连接
        self._connector: Optional[MySQLConnector] = None
        self._creator: Optional[StftMusicProcessorCreate] = None
        self._predictor: Optional[StftMusicProcessorPredict] = None

        # 采样率配置
        self.sr = self.hp.fingerprint.core.stft.sr
        self.hop_length = self.hp.fingerprint.core.stft.hop_length

    def _get_connector(self) -> MySQLConnector:
        """获取数据库连接（懒加载）"""
        if self._connector is None:
            self._connector = MySQLConnector()
        return self._connector

    def _get_creator(self) -> StftMusicProcessorCreate:
        """获取指纹创建器（懒加载）"""
        if self._creator is None:
            self._creator = StftMusicProcessorCreate()
        return self._creator

    def _get_predictor(self) -> StftMusicProcessorPredict:
        """获取指纹预测器（懒加载）"""
        if self._predictor is None:
            self._predictor = StftMusicProcessorPredict()
        return self._predictor

    def close(self):
        """关闭数据库连接，释放资源"""
        if self._connector:
            try:
                self._connector.cursor.close()
                self._connector.conn.close()
            except Exception:
                pass
            self._connector = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def init_database(self):
        """初始化数据库（检查并创建表）"""
        dc = DatabaseChecker()
        dc.check_database()
        dc.check_tables()

    def add_reference(self, audio_path: str, name: Optional[str] = None) -> int:
        """
        添加参考音频到指纹库

        Args:
            audio_path: 音频文件路径
            name: 音频名称（默认使用文件名）

        Returns:
            music_id: 音乐ID

        Raises:
            FileNotFoundError: 音频文件不存在
            Exception: 数据库操作失败
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        connector = self._get_connector()
        creator = self._get_creator()

        # 使用指定名称或文件名
        if name is None:
            name = Path(audio_path).stem

        # 检查是否已存在
        existing_id = connector.find_music_by_music_path(audio_path)
        if existing_id is not None:
            return existing_id

        # 创建指纹并保存
        creator.create_finger_prints_and_save_database(
            music_path=audio_path,
            connector=connector
        )

        # 返回音乐ID
        return connector.find_music_by_music_path(audio_path)

    def add_references(self, audio_paths: List[str], names: Optional[List[str]] = None) -> List[int]:
        """
        批量添加参考音频

        Args:
            audio_paths: 音频文件路径列表
            names: 音频名称列表（可选）

        Returns:
            music_ids: 音乐ID列表
        """
        names = names or [None] * len(audio_paths)
        return [self.add_reference(path, name) for path, name in zip(audio_paths, names)]

    def recognize(self, query_path: str, threshold: int = 10) -> RecognitionResult:
        """
        识别查询音频

        Args:
            query_path: 查询音频路径
            threshold: 匹配阈值，低于此值认为匹配失败

        Returns:
            RecognitionResult: 识别结果
        """
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"查询音频不存在: {query_path}")

        connector = self._get_connector()
        predictor = self._get_predictor()

        # 执行识别
        result = predictor.predict_music(
            music_path=query_path,
            connector=connector
        )

        music_id = result['music_id']
        offset = result['music_offset']
        confidence = result['max_hash_count']

        # 判断是否匹配成功
        # offset == -1 是初始值表示未匹配，其他值（包括负数）都是有效偏移
        matched = confidence >= threshold and music_id != -1 and offset != -1

        # 获取音乐名称
        name = ""
        if matched:
            name = connector.find_music_name_by_music_id(music_id) or ""

        # 计算时间偏移（秒），offset 是帧索引（可能为负值）
        if matched:
            offset_seconds = offset * self.hop_length / self.sr
        else:
            offset_seconds = -1.0  # 未匹配返回 -1

        return RecognitionResult(
            music_id=music_id,
            name=name,
            offset=offset_seconds,
            confidence=confidence,
            matched=matched
        )

    def locate(self, long_audio_path: str, reference_path: Optional[str] = None,
               reference_name: Optional[str] = None, threshold: int = 10,
               auto_match: bool = False) -> LocationResult:
        """
        在长音频中定位参考片段的位置

        Args:
            long_audio_path: 长音频文件路径
            reference_path: 参考音频路径（与 reference_name 二选一，若都未提供则需设置 auto_match=True）
            reference_name: 参考音频名称（已添加到指纹库中）
            threshold: 匹配阈值
            auto_match: 是否自动从数据库匹配参考音频（无需指定 reference_path/name）

        Returns:
            LocationResult: 定位结果

        Raises:
            ValueError: 未指定参考音频且 auto_match=False
            FileNotFoundError: 音频文件不存在
        """
        if not os.path.exists(long_audio_path):
            raise FileNotFoundError(f"音频文件不存在: {long_audio_path}")

        if reference_path is None and reference_name is None and not auto_match:
            raise ValueError("必须指定 reference_path/reference_name 之一，或设置 auto_match=True")

        # 识别（自动从数据库找最匹配的参考音频）
        result = self.recognize(long_audio_path, threshold=threshold)

        if not result.matched:
            return LocationResult(found=False, start_time=0, end_time=0, confidence=0)

        # 验证匹配到的参考音频是否是指定的（如果指定了）
        if reference_name and result.name != reference_name:
            return LocationResult(found=False, start_time=0, end_time=0, confidence=0,
                                  music_name=f"匹配到 {result.name}，但指定的是 {reference_name}")

        # 计算参考音频时长
        if reference_path and os.path.exists(reference_path):
            try:
                import librosa
                ref_duration = librosa.get_duration(path=reference_path, sr=self.sr)
            except Exception:
                # 如果读取失败，使用默认时长
                ref_duration = 10.0
        else:
            # 默认使用10秒
            ref_duration = 10.0

        return LocationResult(
            found=True,
            start_time=result.offset,
            end_time=result.offset + ref_duration,
            confidence=result.confidence,
            music_id=result.music_id,
            music_name=result.name
        )

    def get_all_references(self) -> List[Dict]:
        """
        获取所有参考音频列表

        Returns:
            参考音频信息列表 [{music_id, name, path, hash_count}, ...]
        """
        connector = self._get_connector()
        hp = self.hp

        sql = f"SELECT {hp.fingerprint.database.tables.music.column.music_id}, " \
              f"{hp.fingerprint.database.tables.music.column.music_name}, " \
              f"{hp.fingerprint.database.tables.music.column.music_path} " \
              f"FROM {hp.fingerprint.database.tables.music.name}"

        connector.cursor.execute(sql)
        results = connector.cursor.fetchall()

        references = []
        for music_id, name, path in results:
            hash_count = connector.calculation_hash_num_by_music_id(music_id)
            references.append({
                'music_id': music_id,
                'name': name,
                'path': path,
                'hash_count': hash_count
            })

        return references

    def delete_reference(self, music_id: int) -> bool:
        """
        删除参考音频及其指纹

        Args:
            music_id: 音乐ID

        Returns:
            bool: 是否删除成功
        """
        connector = self._get_connector()
        hp = self.hp

        try:
            # 删除指纹
            sql_fp = f"DELETE FROM {hp.fingerprint.database.tables.finger_prints.name} " \
                     f"WHERE {hp.fingerprint.database.tables.finger_prints.column.music_id_fk} = %s"
            connector.cursor.execute(sql_fp, (music_id,))

            # 删除音乐记录
            sql_music = f"DELETE FROM {hp.fingerprint.database.tables.music.name} " \
                        f"WHERE {hp.fingerprint.database.tables.music.column.music_id} = %s"
            connector.cursor.execute(sql_music, (music_id,))

            connector.conn.commit()
            return True
        except Exception as e:
            print(f"删除失败: {e}")
            return False

    def clear_database(self) -> bool:
        """
        清空指纹数据库（慎用）

        Returns:
            bool: 是否清空成功
        """
        connector = self._get_connector()
        hp = self.hp

        try:
            sql_fp = f"TRUNCATE TABLE {hp.fingerprint.database.tables.finger_prints.name}"
            connector.cursor.execute(sql_fp)

            sql_music = f"TRUNCATE TABLE {hp.fingerprint.database.tables.music.name}"
            connector.cursor.execute(sql_music)

            connector.conn.commit()
            return True
        except Exception as e:
            print(f"清空失败: {e}")
            return False


# ==================== 便捷函数 ====================

def create_fingerprint_db(audio_dir: str, pattern: str = "*.wav",
                          config_path: Optional[str] = None) -> List[int]:
    """
    从目录批量创建指纹数据库

    Args:
        audio_dir: 音频文件目录
        pattern: 文件匹配模式，默认 "*.wav"
        config_path: 配置文件路径

    Returns:
        music_ids: 添加的音乐ID列表

    Example:
        >>> ids = create_fingerprint_db("/path/to/audio", "*.mp3")
    """
    import glob

    fingerprinter = AudioFingerprinter(config_path)
    fingerprinter.init_database()

    audio_files = glob.glob(os.path.join(audio_dir, pattern))
    ids = fingerprinter.add_references(audio_files)

    fingerprinter.close()
    return ids


def batch_recognize(query_paths: List[str], threshold: int = 10,
                    config_path: Optional[str] = None) -> List[RecognitionResult]:
    """
    批量识别音频文件

    Args:
        query_paths: 查询音频路径列表
        threshold: 匹配阈值
        config_path: 配置文件路径

    Returns:
        识别结果列表

    Example:
        >>> results = batch_recognize(["q1.wav", "q2.wav"])
        >>> for r in results:
        ...     print(r.name, r.matched)
    """
    fingerprinter = AudioFingerprinter(config_path)
    results = [fingerprinter.recognize(path, threshold) for path in query_paths]
    fingerprinter.close()
    return results


def batch_locate(long_audio_paths: List[str], reference_path: str,
                 threshold: int = 10, config_path: Optional[str] = None) -> List[LocationResult]:
    """
    批量定位音频片段位置

    Args:
        long_audio_paths: 长音频路径列表
        reference_path: 参考音频路径
        threshold: 匹配阈值
        config_path: 配置文件路径

    Returns:
        定位结果列表

    Example:
        >>> positions = batch_locate(["long1.wav", "long2.wav"], "ref.wav")
        >>> for pos in positions:
        ...     print(pos.start_time, pos.found)
    """
    fingerprinter = AudioFingerprinter(config_path)

    # 先添加参考音频
    fingerprinter.add_reference(reference_path)

    # 批量定位
    results = [fingerprinter.locate(path, reference_path=reference_path, threshold=threshold)
               for path in long_audio_paths]

    fingerprinter.close()
    return results
