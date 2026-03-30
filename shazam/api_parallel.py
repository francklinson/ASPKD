# -*- coding: utf-8 -*-
"""
Shazam 并行批量查询接口

使用多线程并发加速多文件的音频指纹查询
"""

import os
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time

from .api import AudioFingerprinter, RecognitionResult, LocationResult
from .database.MySQLConnector import MySQLConnector
from .core.STFT.STFTMusicProcessorPredict import STFTMusicProcessorPredict
from .utils.hparam import Hparam


@dataclass
class ParallelResult:
    """并行查询结果"""
    file_path: str
    success: bool
    result: Optional[RecognitionResult]
    error: Optional[str]
    process_time: float


class ThreadLocalFingerprinter:
    """
    线程本地指纹识别器
    
    每个线程拥有独立的 MySQL 连接，避免线程安全问题
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._local = threading.local()
    
    def _get_fingerprinter(self) -> AudioFingerprinter:
        """获取线程本地的指纹识别器"""
        if not hasattr(self._local, 'fingerprinter'):
            self._local.fingerprinter = AudioFingerprinter(self.config_path)
        return self._local.fingerprinter
    
    def recognize(self, query_path: str, threshold: int = 10) -> RecognitionResult:
        """识别单个音频"""
        fp = self._get_fingerprinter()
        return fp.recognize(query_path, threshold)
    
    def locate(self, long_audio_path: str, reference_path: Optional[str] = None,
               reference_name: Optional[str] = None, threshold: int = 10,
               auto_match: bool = False) -> LocationResult:
        """定位单个音频"""
        fp = self._get_fingerprinter()
        return fp.locate(long_audio_path, reference_path, reference_name, threshold, auto_match)
    
    def close(self):
        """关闭线程本地资源"""
        if hasattr(self._local, 'fingerprinter'):
            self._local.fingerprinter.close()
            del self._local.fingerprinter


class ParallelAudioFingerprinter:
    """
    并行音频指纹识别器
    
    使用线程池并发处理多个音频文件的识别/定位
    
    用法:
        >>> fingerprinter = ParallelAudioFingerprinter(max_workers=4)
        >>> 
        >>> # 批量识别
        >>> results = fingerprinter.batch_recognize(
        ...     query_paths=["a.wav", "b.wav", "c.wav"],
        ...     threshold=10
        ... )
        >>> 
        >>> # 批量定位
        >>> locations = fingerprinter.batch_locate(
        ...     long_audio_paths=["long1.wav", "long2.wav"],
        ...     reference_path="ref.wav",
        ...     threshold=10
        ... )
    """
    
    def __init__(self, max_workers: int = 4, config_path: Optional[str] = None):
        """
        初始化并行指纹识别器

        Args:
            max_workers: 最大线程数，默认为4
            config_path: 配置文件路径
        """
        self.max_workers = max_workers
        self.config_path = config_path
        self._thread_local = ThreadLocalFingerprinter(config_path)
        
        # 统计信息
        self._stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_time': 0.0
        }
    
    def _recognize_single(self, query_path: str, threshold: int) -> ParallelResult:
        """单个音频的识别任务"""
        start_time = time.time()
        
        try:
            if not os.path.exists(query_path):
                return ParallelResult(
                    file_path=query_path,
                    success=False,
                    result=None,
                    error=f"文件不存在: {query_path}",
                    process_time=time.time() - start_time
                )
            
            result = self._thread_local.recognize(query_path, threshold)
            
            return ParallelResult(
                file_path=query_path,
                success=True,
                result=result,
                error=None,
                process_time=time.time() - start_time
            )
            
        except Exception as e:
            return ParallelResult(
                file_path=query_path,
                success=False,
                result=None,
                error=str(e),
                process_time=time.time() - start_time
            )
    
    def _locate_single(self, long_audio_path: str, reference_path: Optional[str],
                       reference_name: Optional[str], threshold: int,
                       auto_match: bool) -> ParallelResult:
        """单个音频的定位任务"""
        start_time = time.time()
        
        try:
            if not os.path.exists(long_audio_path):
                return ParallelResult(
                    file_path=long_audio_path,
                    success=False,
                    result=None,
                    error=f"文件不存在: {long_audio_path}",
                    process_time=time.time() - start_time
                )
            
            location = self._thread_local.locate(
                long_audio_path=long_audio_path,
                reference_path=reference_path,
                reference_name=reference_name,
                threshold=threshold,
                auto_match=auto_match
            )
            
            # 将 LocationResult 包装成 RecognitionResult 格式
            result = RecognitionResult(
                music_id=location.music_id,
                name=location.music_name,
                offset=location.start_time,
                confidence=location.confidence,
                matched=location.found
            )
            
            return ParallelResult(
                file_path=long_audio_path,
                success=True,
                result=result,
                error=None,
                process_time=time.time() - start_time
            )
            
        except Exception as e:
            return ParallelResult(
                file_path=long_audio_path,
                success=False,
                result=None,
                error=str(e),
                process_time=time.time() - start_time
            )
    
    def batch_recognize(self, query_paths: List[str], threshold: int = 10,
                       progress_callback: Optional[Callable[[int, int], None]] = None,
                       use_parallel: bool = True) -> List[ParallelResult]:
        """
        批量识别音频文件

        Args:
            query_paths: 查询音频路径列表
            threshold: 匹配阈值
            progress_callback: 进度回调函数，参数为(已完成数, 总数)
            use_parallel: 是否使用并行处理

        Returns:
            并行结果列表
        """
        if not query_paths:
            return []
        
        self._stats['total'] = len(query_paths)
        self._stats['success'] = 0
        self._stats['failed'] = 0
        self._stats['total_time'] = 0.0
        
        results = []
        
        if use_parallel and len(query_paths) > 1:
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_path = {
                    executor.submit(self._recognize_single, path, threshold): path 
                    for path in query_paths
                }
                
                # 收集结果
                completed = 0
                for future in as_completed(future_to_path):
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        self._stats['success'] += 1
                    else:
                        self._stats['failed'] += 1
                    
                    self._stats['total_time'] += result.process_time
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(query_paths))
        else:
            # 串行处理
            for i, path in enumerate(query_paths):
                result = self._recognize_single(path, threshold)
                results.append(result)
                
                if result.success:
                    self._stats['success'] += 1
                else:
                    self._stats['failed'] += 1
                
                self._stats['total_time'] += result.process_time
                
                if progress_callback:
                    progress_callback(i + 1, len(query_paths))
        
        return results
    
    def batch_locate(self, long_audio_paths: List[str], 
                    reference_path: Optional[str] = None,
                    reference_name: Optional[str] = None,
                    threshold: int = 10,
                    auto_match: bool = False,
                    progress_callback: Optional[Callable[[int, int], None]] = None,
                    use_parallel: bool = True) -> List[ParallelResult]:
        """
        批量定位音频片段位置

        Args:
            long_audio_paths: 长音频路径列表
            reference_path: 参考音频路径
            reference_name: 参考音频名称
            threshold: 匹配阈值
            auto_match: 是否自动匹配
            progress_callback: 进度回调函数
            use_parallel: 是否使用并行处理

        Returns:
            并行结果列表
        """
        if not long_audio_paths:
            return []
        
        self._stats['total'] = len(long_audio_paths)
        self._stats['success'] = 0
        self._stats['failed'] = 0
        self._stats['total_time'] = 0.0
        
        results = []
        
        if use_parallel and len(long_audio_paths) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(
                        self._locate_single, path, reference_path, 
                        reference_name, threshold, auto_match
                    ): path 
                    for path in long_audio_paths
                }
                
                completed = 0
                for future in as_completed(future_to_path):
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        self._stats['success'] += 1
                    else:
                        self._stats['failed'] += 1
                    
                    self._stats['total_time'] += result.process_time
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(long_audio_paths))
        else:
            for i, path in enumerate(long_audio_paths):
                result = self._locate_single(
                    path, reference_path, reference_name, threshold, auto_match
                )
                results.append(result)
                
                if result.success:
                    self._stats['success'] += 1
                else:
                    self._stats['failed'] += 1
                
                self._stats['total_time'] += result.process_time
                
                if progress_callback:
                    progress_callback(i + 1, len(long_audio_paths))
        
        return results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self._stats.copy()
    
    def close(self):
        """释放资源"""
        self._thread_local.close()


# ==================== 便捷函数 ====================

def batch_recognize_parallel(query_paths: List[str], 
                             threshold: int = 10,
                             max_workers: int = 4,
                             config_path: Optional[str] = None) -> List[ParallelResult]:
    """
    并行批量识别音频文件

    Args:
        query_paths: 查询音频路径列表
        threshold: 匹配阈值
        max_workers: 最大线程数
        config_path: 配置文件路径

    Returns:
        并行结果列表

    Example:
        >>> results = batch_recognize_parallel(["a.wav", "b.wav", "c.wav"], max_workers=4)
        >>> for r in results:
        ...     print(r.file_path, r.result.matched if r.success else r.error)
    """
    fingerprinter = ParallelAudioFingerprinter(max_workers, config_path)
    results = fingerprinter.batch_recognize(query_paths, threshold)
    fingerprinter.close()
    return results


def batch_locate_parallel(long_audio_paths: List[str],
                         reference_path: Optional[str] = None,
                         reference_name: Optional[str] = None,
                         threshold: int = 10,
                         auto_match: bool = False,
                         max_workers: int = 4,
                         config_path: Optional[str] = None) -> List[ParallelResult]:
    """
    并行批量定位音频片段位置

    Args:
        long_audio_paths: 长音频路径列表
        reference_path: 参考音频路径
        reference_name: 参考音频名称
        threshold: 匹配阈值
        auto_match: 是否自动匹配
        max_workers: 最大线程数
        config_path: 配置文件路径

    Returns:
        并行结果列表

    Example:
        >>> results = batch_locate_parallel(["long1.wav", "long2.wav"], 
        ...                                 reference_path="ref.wav",
        ...                                 max_workers=4)
        >>> for r in results:
        ...     print(r.file_path, r.result.offset if r.success else r.error)
    """
    fingerprinter = ParallelAudioFingerprinter(max_workers, config_path)
    results = fingerprinter.batch_locate(
        long_audio_paths, reference_path, reference_name, threshold, auto_match
    )
    fingerprinter.close()
    return results
