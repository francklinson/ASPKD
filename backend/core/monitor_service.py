"""
目录监控服务
用于实时监控目录下的新文件
"""
import os
import sys
import time
import asyncio
from typing import List, Set, Dict, Optional
from datetime import datetime
from collections import deque

# 确保导入路径正确
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
venv_site_packages = os.path.join(project_root, ".venv", "lib", "python3.12", "site-packages")
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from backend.core.task_manager import task_manager
from backend.core.websocket import websocket_manager


class AudioFileHandler(FileSystemEventHandler):
    """音频文件事件处理器"""
    
    def __init__(self, monitor_service, extensions: List[str], loop: asyncio.AbstractEventLoop):
        self.monitor_service = monitor_service
        self.extensions = set(ext.lower() for ext in extensions)
        self.loop = loop  # 主事件循环
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = event.src_path
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in self.extensions:
            filename = os.path.basename(file_path)
            print(f"[Monitor] 检测到新文件: {file_path}")
            # 发送日志广播
            asyncio.run_coroutine_threadsafe(
                websocket_manager.broadcast({
                    "type": "monitor_log",
                    "data": {
                        "level": "info",
                        "message": f"📁 检测到新文件: {filename}"
                    }
                }),
                self.loop
            )
            # 使用线程安全的方式提交到主事件循环
            asyncio.run_coroutine_threadsafe(
                self._delayed_process(file_path), 
                self.loop
            )
    
    async def _delayed_process(self, file_path: str, delay: float = 1.0):
        """延迟处理，等待文件写入完成"""
        await asyncio.sleep(delay)
        
        if os.path.exists(file_path):
            await self.monitor_service.process_file(file_path)


class MonitorService:
    """监控服务"""
    
    def __init__(self):
        self.is_running: bool = False
        self.directory: Optional[str] = None
        self.interval: int = 30
        self.algorithm: str = "dinomaly_dinov3_small"
        self.device: str = "auto"
        self.file_extensions: List[str] = [".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"]
        
        # 参考音频选择
        self.reference_audios: List[str] = []  # 用户选择的参考音频列表
        
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[AudioFileHandler] = None
        
        self.processed_files: Set[str] = set()
        self.detection_results: deque = deque(maxlen=1000)
        self.total_processed: int = 0
        self.anomaly_count: int = 0
        self.start_time: Optional[datetime] = None
        
        self._scan_task: Optional[asyncio.Task] = None
        
        # 批量处理队列
        self._pending_files: List[str] = []
        self._batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
        self._batch_delay: float = 3.0  # 等待3秒收集文件（给文件写入留足时间）
        
        # 长音频分析器
        self._analyzer = None
        
        # 检测器（常驻内存，避免重复加载）
        self._detector = None
        self._current_algorithm: Optional[str] = None
    
    async def start(
        self,
        directory: str,
        interval: int = 30,
        algorithm: str = "dinomaly_dinov3_small",
        device: str = "auto",
        detect_existing: bool = False,
        file_extensions: List[str] = None,
        reference_audios: List[str] = None
    ) -> bool:
        """启动监控"""
        if self.is_running:
            return False
        
        if not os.path.exists(directory):
            raise ValueError(f"目录不存在: {directory}")
        
        self.directory = directory
        self.interval = interval
        self.algorithm = algorithm
        self.device = device
        self.file_extensions = file_extensions or self.file_extensions
        self.reference_audios = reference_audios or []
        self.start_time = datetime.now()
        
        # 初始化长音频分析器
        await self._init_analyzer()
        
        # 初始化检测器（常驻内存）
        await self._init_detector()
        
        # 扫描已有文件
        existing_files = self._get_audio_files()
        
        if detect_existing:
            # 使用长音频分析器处理已有文件
            if existing_files:
                await websocket_manager.broadcast({
                    "type": "monitor_log",
                    "data": {
                        "level": "info",
                        "message": f"🔄 开始检测 {len(existing_files)} 个已有文件..."
                    }
                })
                
                # 逐个处理文件
                for file_path in existing_files:
                    await self._process_long_audio(file_path)
        else:
            # 记录已有文件，不处理
            self.processed_files.update(existing_files)
        
        # 启动文件系统监控
        self.event_handler = AudioFileHandler(self, self.file_extensions, asyncio.get_event_loop())
        self.observer = Observer()
        self.observer.schedule(self.event_handler, directory, recursive=False)
        self.observer.start()
        
        # 启动定期扫描（作为备用）
        self._scan_task = asyncio.create_task(self._periodic_scan())
        
        self.is_running = True
        print(f"[Monitor] 开始监控目录: {directory}")
        
        return True
    
    async def stop(self) -> bool:
        """停止监控"""
        if not self.is_running:
            return False
        
        self.is_running = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
            self._scan_task = None
        
        # 处理队列中剩余的文件
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        
        # 处理剩余待处理文件
        async with self._batch_lock:
            if self._pending_files:
                remaining_files = self._pending_files.copy()
                self._pending_files.clear()
                await websocket_manager.broadcast({
                    "type": "monitor_log",
                    "data": {
                        "level": "info",
                        "message": f"🔄 停止前处理剩余 {len(remaining_files)} 个文件"
                    }
                })
                await self._process_files_batch(remaining_files)
        
        # 释放检测器资源
        if self._detector:
            try:
                import time
                print(f"[Monitor] [模型卸载] 开始卸载检测器: {self._current_algorithm}")
                unload_start = time.time()
                self._detector.release()
                unload_time = time.time() - unload_start
                print(f"[Monitor] [模型卸载] 检测器卸载完成，耗时: {unload_time:.3f}s")
                
                print(f"[Monitor] [模型卸载] 清理GPU缓存...")
                cache_start = time.time()
                import torch
                torch.cuda.empty_cache()
                cache_time = time.time() - cache_start
                print(f"[Monitor] [模型卸载] GPU缓存清理完成，耗时: {cache_time:.3f}s")
                
                self._detector = None
                self._current_algorithm = None
                print(f"[Monitor] [模型卸载] 检测器资源已完全释放")
            except Exception as e:
                print(f"[Monitor] [模型卸载] 释放检测器资源时出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[Monitor] [模型卸载] 检测器未加载，无需卸载")
        
        # 释放长音频分析器资源
        if self._analyzer:
            try:
                print(f"[Monitor] [分析器卸载] 开始卸载长音频分析器...")
                self._analyzer = None
                print(f"[Monitor] [分析器卸载] 长音频分析器资源已释放")
            except Exception as e:
                print(f"[Monitor] [分析器卸载] 释放分析器资源时出错: {e}")
        
        print("[Monitor] 监控已停止")
        return True
    
    async def _periodic_scan(self):
        """定期扫描（备用机制）"""
        while self.is_running:
            try:
                await asyncio.sleep(self.interval)
                
                if not self.is_running:
                    break
                
                # 扫描新文件
                current_files = set(self._get_audio_files())
                new_files = current_files - self.processed_files
                
                if new_files:
                    print(f"[Monitor] 扫描发现 {len(new_files)} 个新文件")
                    await websocket_manager.broadcast({
                        "type": "monitor_log",
                        "data": {
                            "level": "info",
                            "message": f"🔍 定期扫描发现 {len(new_files)} 个新文件"
                        }
                    })
                    for file_path in new_files:
                        await self.process_file(file_path)
                
            except Exception as e:
                print(f"[Monitor] 扫描出错: {e}")
    
    async def process_file(self, file_path: str):
        """处理单个文件（加入批量队列）"""
        if file_path in self.processed_files:
            return
        
        async with self._batch_lock:
            self._pending_files.append(file_path)
            queue_length = len(self._pending_files)
            
            # 启动批量处理任务（如果未启动）
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._batch_process())
                print(f"[Monitor] 启动批量处理任务，当前队列: {queue_length} 个文件")
        
        # 发送队列日志
        filename = os.path.basename(file_path)
        await websocket_manager.broadcast({
            "type": "monitor_log",
            "data": {
                "level": "info",
                "message": f"📥 文件加入队列: {filename} (队列长度: {queue_length})"
            }
        })
    
    async def _batch_process(self):
        """批量处理队列中的文件"""
        # 等待一段时间收集文件
        await asyncio.sleep(self._batch_delay)
        
        async with self._batch_lock:
            if not self._pending_files:
                return
            
            # 获取当前队列中的所有文件
            files_to_process = self._pending_files.copy()
            self._pending_files.clear()
        
        # 批量处理
        await self._process_files_batch(files_to_process)
    
    async def _process_files_batch(self, file_paths: List[str]):
        """批量处理多个文件（优化版：所有文件统一推理）"""
        if not file_paths:
            return
        
        batch_size = len(file_paths)
        filenames = [os.path.basename(f) for f in file_paths]
        
        await websocket_manager.broadcast({
            "type": "monitor_log",
            "data": {
                "level": "info",
                "message": f"🚀 开始批量处理 {batch_size} 个文件: {', '.join(filenames[:3])}{'...' if batch_size > 3 else ''}"
            }
        })
        
        try:
            # 第1阶段：分析所有文件，收集匹配片段
            print(f"[Monitor] 第1阶段：分析所有文件，收集匹配片段...")
            all_segments = []  # 存储所有文件的片段信息
            files_with_matches = []  # 有匹配片段的文件
            files_no_matches = []  # 无匹配片段的文件
            
            for file_path in file_paths:
                if file_path in self.processed_files:
                    continue
                    
                filename = os.path.basename(file_path)
                segments = await self._analyze_file_only(file_path, filename)
                
                if segments:
                    all_segments.extend(segments)
                    files_with_matches.append((file_path, filename, len(segments)))
                else:
                    files_no_matches.append((file_path, filename))
                    
            print(f"[Monitor] 第1阶段完成：{len(files_with_matches)} 个文件有匹配片段，共 {len(all_segments)} 个片段")
            
            # 处理无匹配片段的文件
            for file_path, filename in files_no_matches:
                await self._handle_no_match(file_path, filename)
                self.processed_files.add(file_path)
                self.total_processed += 1
            
            # 如果有匹配片段，统一处理
            if all_segments:
                await websocket_manager.broadcast({
                    "type": "monitor_log",
                    "data": {
                        "level": "info",
                        "message": f"🔍 发现 {len(all_segments)} 个匹配片段，开始统一切分和推理..."
                    }
                })
                
                # 第2阶段：统一切分并生成频谱图
                print(f"[Monitor] 第2阶段：统一切分并生成频谱图...")
                spectrogram_data = await self._generate_spectrograms_batch(all_segments)
                print(f"[Monitor] 第2阶段完成：生成了 {len(spectrogram_data)} 个频谱图")
                
                # 第3阶段：统一执行模型推理（只执行一次）
                print(f"[Monitor] 第3阶段：统一执行模型推理...")
                await websocket_manager.broadcast({
                    "type": "monitor_log",
                    "data": {
                        "level": "info",
                        "message": f"🧠 开始批量推理，共 {len(spectrogram_data)} 个样本..."
                    }
                })
                
                detection_results = await self._run_batch_inference(spectrogram_data)
                print(f"[Monitor] 第3阶段完成：获得 {len(detection_results)} 个推理结果")
                
                # 第4阶段：处理所有结果
                print(f"[Monitor] 第4阶段：处理推理结果...")
                await self._process_all_detection_results(spectrogram_data, detection_results)
                
                # 标记所有文件为已处理
                for file_path, filename, _ in files_with_matches:
                    self.processed_files.add(file_path)
                    self.total_processed += 1
            
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "success",
                    "message": f"✅ 批量处理完成: {batch_size} 个文件（{len(files_with_matches)} 个有匹配，{len(files_no_matches)} 个无匹配）"
                }
            })
            
            print(f"[Monitor] 批量处理完成: {batch_size} 个文件")
            
        except Exception as e:
            print(f"[Monitor] 批量处理失败: {e}")
            import traceback
            traceback.print_exc()
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "error",
                    "message": f"❌ 批量处理失败: {str(e)}"
                }
            })
        
        finally:
            # 确保所有文件被记录
            for file_path in file_paths:
                if file_path not in self.processed_files:
                    self.processed_files.add(file_path)
                    self.total_processed += 1
    
    async def _analyze_file_only(self, file_path: str, filename: str) -> List[Dict]:
        """仅分析文件，返回匹配片段信息（不切分、不推理）"""
        print(f"[Monitor Debug] 分析文件: {filename}")
        
        await websocket_manager.broadcast({
            "type": "monitor_log",
            "data": {
                "level": "info",
                "message": f"🎵 分析长音频: {filename}"
            }
        })
        
        try:
            if self._analyzer is None:
                raise Exception("长音频分析器未初始化")
            
            loop = asyncio.get_event_loop()
            
            def progress_callback(message: str, progress: float):
                if progress in [0.05, 0.25, 0.50, 0.90, 1.0]:
                    asyncio.run_coroutine_threadsafe(
                        websocket_manager.broadcast({
                            "type": "monitor_log",
                            "data": {
                                "level": "info",
                                "message": f"[{filename}] {message}"
                            }
                        }),
                        loop
                    )
            
            # 在线程池中执行分析
            result = await loop.run_in_executor(
                None,
                lambda: self._analyzer.analyze(file_path, progress_callback)
            )
            
            # 收集片段信息
            segments = []
            if result.segment_matches:
                for segment in result.segment_matches:
                    segments.append({
                        'file_path': file_path,
                        'filename': filename,
                        'segment': segment
                    })
                print(f"[Monitor Debug] {filename}: 发现 {len(segments)} 个匹配片段")
            else:
                print(f"[Monitor Debug] {filename}: 未找到匹配片段")
            
            return segments
            
        except Exception as e:
            print(f"[Monitor] 分析文件失败 {filename}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _handle_no_match(self, file_path: str, filename: str):
        """处理无匹配片段的文件"""
        await websocket_manager.broadcast({
            "type": "monitor_log",
            "data": {
                "level": "warning",
                "message": f"⚠️ {filename}: 未找到匹配的参考音频"
            }
        })
        
        no_match_result = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "filepath": file_path,
            "anomaly_score": 0,
            "is_anomaly": False,
            "status": "未匹配",
            "original_path": None,
            "overlay_path": None,
            "heatmap_path": None,
            "segment_info": None
        }
        self.detection_results.append(no_match_result)
        
        await websocket_manager.broadcast({
            "type": "monitor_update",
            "data": {
                "total_processed": self.total_processed,
                "anomaly_count": self.anomaly_count,
                "latest_result": no_match_result
            }
        })
    
    async def _generate_spectrograms_batch(self, all_segments: List[Dict]) -> List[Dict]:
        """批量生成所有片段的频谱图"""
        import librosa
        import soundfile as sf
        from preprocessing import plot_spectrogram
        
        temp_dir = os.path.join("slice", "monitor")
        pic_dir = os.path.join("slice", "monitor", "picture")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(pic_dir, exist_ok=True)
        
        spectrogram_data = []
        
        for seg_info in all_segments:
            file_path = seg_info['file_path']
            filename = seg_info['filename']
            segment = seg_info['segment']
            
            # 切分音频
            start_time = segment.start_time
            end_time = segment.end_time
            duration = end_time - start_time
            
            audio, sr = librosa.load(file_path, sr=22050, offset=start_time, duration=duration)
            
            segment_filename = f"{os.path.splitext(filename)[0]}_{segment.music_name}_{int(start_time)}s_{int(end_time)}s.wav"
            segment_path = os.path.join(temp_dir, segment_filename)
            sf.write(segment_path, audio, sr)
            
            # 生成频谱图
            base_name = os.path.splitext(segment_filename)[0]
            spectrogram_path = os.path.join(pic_dir, f"{base_name}.png")
            plot_spectrogram(segment_path, spectrogram_path)
            
            spectrogram_data.append({
                'file_path': file_path,
                'filename': filename,
                'segment': segment,
                'segment_filename': segment_filename,
                'segment_path': segment_path,
                'spectrogram_path': spectrogram_path
            })
            
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "info",
                    "message": f"🎵 切分片段: {segment_filename}"
                }
            })
        
        return spectrogram_data
    
    async def _run_batch_inference(self, spectrogram_data: List[Dict]) -> List:
        """执行批量模型推理（只执行一次）"""
        if not spectrogram_data:
            return []
        
        spectrogram_paths = [d['spectrogram_path'] for d in spectrogram_data]
        
        # 确保检测器已初始化
        if self._detector is None:
            await self._init_detector()
        
        print(f"[Monitor Debug] 检测器就绪，开始批量推理 {len(spectrogram_paths)} 个样本...")
        
        # 批量推理（只调用一次 predict_batch）
        detection_results = self._detector.predict_batch(spectrogram_paths)
        
        print(f"[Monitor Debug] 批量推理完成: 获得 {len(detection_results)} 个结果")
        
        await websocket_manager.broadcast({
            "type": "monitor_log",
            "data": {
                "level": "success",
                "message": f"✅ 批量推理完成: {len(detection_results)} 个样本"
            }
        })
        
        return detection_results
    
    async def _process_all_detection_results(self, spectrogram_data: List[Dict], detection_results: List):
        """处理所有检测结果"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        for seg_info, detection_result_obj in zip(spectrogram_data, detection_results):
            filename = seg_info['filename']
            file_path = seg_info['file_path']
            segment = seg_info['segment']
            segment_filename = seg_info['segment_filename']
            
            # 从metadata中获取图片路径
            original_path = None
            overlay_path = None
            heatmap_path = None
            if detection_result_obj.metadata:
                original_path = detection_result_obj.metadata.get('original_path')
                overlay_path = detection_result_obj.metadata.get('overlay_path')
                heatmap_path = detection_result_obj.metadata.get('heatmap_path')
                
                # 转换为相对路径
                if original_path and original_path.startswith(project_root):
                    original_path = original_path[len(project_root)+1:].replace('\\', '/')
                if overlay_path and overlay_path.startswith(project_root):
                    overlay_path = overlay_path[len(project_root)+1:].replace('\\', '/')
                if heatmap_path and heatmap_path.startswith(project_root):
                    heatmap_path = heatmap_path[len(project_root)+1:].replace('\\', '/')
            
            # 创建检测结果
            detection_result = {
                "timestamp": datetime.now().isoformat(),
                "filename": segment_filename,
                "filepath": file_path,
                "anomaly_score": float(detection_result_obj.anomaly_score),
                "is_anomaly": bool(detection_result_obj.is_anomaly),
                "status": "异常" if detection_result_obj.is_anomaly else "正常",
                "original_path": original_path,
                "overlay_path": overlay_path,
                "heatmap_path": heatmap_path,
                "segment_info": {
                    "music_name": segment.music_name,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "confidence": segment.confidence,
                    "match_ratio": segment.match_ratio
                }
            }
            
            self.detection_results.append(detection_result)
            
            if detection_result_obj.is_anomaly:
                self.anomaly_count += 1
            
            # 发送日志
            status_icon = "⚠️" if detection_result_obj.is_anomaly else "✅"
            status_text = "异常" if detection_result_obj.is_anomaly else "正常"
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "warning" if detection_result_obj.is_anomaly else "success",
                    "message": f"{status_icon} {segment_filename}: {status_text} (分数: {detection_result_obj.anomaly_score:.4f})"
                }
            })
            
            # 广播更新
            await websocket_manager.broadcast({
                "type": "monitor_update",
                "data": {
                    "total_processed": self.total_processed,
                    "anomaly_count": self.anomaly_count,
                    "latest_result": detection_result
                }
            })
    
    def _get_audio_files(self) -> List[str]:
        """获取目录下的音频文件"""
        if not self.directory or not os.path.exists(self.directory):
            return []
        
        audio_files = []
        
        try:
            for filename in os.listdir(self.directory):
                filepath = os.path.join(self.directory, filename)
                if os.path.isfile(filepath):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in self.file_extensions:
                        audio_files.append(filepath)
        except Exception as e:
            print(f"[Monitor] 扫描目录失败: {e}")
        
        return audio_files
    
    def get_status(self) -> dict:
        """获取监控状态"""
        return {
            "is_running": self.is_running,
            "directory": self.directory,
            "interval": self.interval,
            "algorithm": self.algorithm,
            "device": self.device,
            "total_processed": self.total_processed,
            "anomaly_count": self.anomaly_count,
            "start_time": self.start_time.isoformat() if self.start_time else None
        }
    
    def get_results(self, limit: int = 100, offset: int = 0) -> List[dict]:
        """获取检测结果"""
        results = list(self.detection_results)
        results.reverse()  # 最新的在前面
        return results[offset:offset + limit]
    
    def clear_results(self):
        """清空结果"""
        self.detection_results.clear()
        self.total_processed = 0
        self.anomaly_count = 0
    
    async def update_reference_audios(self, reference_audios: List[str]) -> dict:
        """动态更新参考音频（运行时生效）"""
        print(f"[Monitor] [更新参考音频] 开始更新参考音频列表...")
        print(f"[Monitor] [更新参考音频] 新参考音频数量: {len(reference_audios)}")
        
        if not self.is_running:
            print(f"[Monitor] [更新参考音频] 监控未运行，只更新配置")
            self.reference_audios = reference_audios or []
            return {
                "success": True,
                "message": "监控未运行，配置已更新",
                "added": [],
                "removed": [],
                "failed": []
            }
        
        # 计算差异
        old_refs = set(self.reference_audios)
        new_refs = set(reference_audios or [])
        
        to_add = new_refs - old_refs
        to_remove = old_refs - new_refs
        
        print(f"[Monitor] [更新参考音频] 需要添加: {len(to_add)} 个")
        print(f"[Monitor] [更新参考音频] 需要移除: {len(to_remove)} 个")
        
        # 检查索引中是否包含未指定的参考音频（首次启动时加载了所有歌曲的情况）
        if not old_refs and self._analyzer:
            index_music_count = len(self._analyzer.matching_engine.index.music_info)
            if index_music_count > 0:
                print(f"[Monitor] [更新参考音频] 警告: 当前索引包含 {index_music_count} 首歌曲（启动时加载了所有歌曲）")
                print(f"[Monitor] [更新参考音频] 警告: 建议停止监控后重新启动，以仅使用指定的参考音频")
        
        added = []
        removed = []
        failed = []
        
        try:
            from core.shazam.database.connector import MySQLConnector
            db_connector = MySQLConnector()
            
            # 移除不再需要的参考音频
            for ref_path in to_remove:
                # 注意：LongAudioAnalyzer 目前没有 remove_reference 方法
                # 这里只是记录，实际需要从索引中移除
                removed.append(ref_path)
                print(f"[Monitor] [更新参考音频] 标记移除: {ref_path}")
            
            # 添加新的参考音频
            for ref_path in to_add:
                music_id = None
                
                # 首先尝试通过完整路径查找
                if os.path.exists(ref_path):
                    music_id = db_connector.find_music_by_music_path(ref_path)
                else:
                    # 文件不存在，尝试通过文件名（不含扩展名）查找
                    ref_name = os.path.splitext(os.path.basename(ref_path))[0]
                    print(f"[Monitor] [更新参考音频] 参考音频文件不存在，尝试使用名称查找: {ref_name}")
                    music_id = db_connector.find_music_by_music_name(ref_name)
                    if music_id:
                        print(f"[Monitor] [更新参考音频] 通过名称找到音乐ID: {music_id}")
                
                if music_id:
                    music_name = db_connector.find_music_name_by_music_id(music_id)
                    if self._analyzer:
                        # 获取数据库中存储的路径
                        stored_path = db_connector.find_music_path_by_music_id(music_id)
                        if stored_path and os.path.exists(stored_path):
                            success = self._analyzer.add_reference(stored_path, music_id, music_name)
                        else:
                            # 如果数据库中的路径也不存在，尝试从数据库直接加载指纹
                            print(f"[Monitor] [更新参考音频] 数据库路径不存在，尝试从数据库加载指纹")
                            success = self._add_reference_from_db(db_connector, music_id, music_name)
                        
                        if success:
                            added.append({"path": ref_path, "name": music_name})
                            print(f"[Monitor] [更新参考音频] 成功添加: {music_name}")
                        else:
                            failed.append({"path": ref_path, "reason": "添加到索引失败"})
                            print(f"[Monitor] [更新参考音频] 添加失败: {ref_path}")
                    else:
                        failed.append({"path": ref_path, "reason": "分析器未初始化"})
                        print(f"[Monitor] [更新参考音频] 分析器未初始化: {ref_path}")
                else:
                    failed.append({"path": ref_path, "reason": "数据库中未找到"})
                    print(f"[Monitor] [更新参考音频] 数据库中未找到: {ref_path}")
            
            # 更新配置
            self.reference_audios = reference_audios or []
            
            print(f"[Monitor] [更新参考音频] 更新完成: 成功添加 {len(added)} 个")
            
            # 发送WebSocket通知
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "info",
                    "message": f"🔄 参考音频已更新: 添加 {len(added)} 个"
                }
            })
            
            return {
                "success": True,
                "message": f"参考音频已更新: 添加 {len(added)} 个",
                "added": added,
                "removed": removed,
                "failed": failed
            }
            
        except Exception as e:
            print(f"[Monitor] [更新参考音频] 更新失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"更新失败: {str(e)}",
                "added": added,
                "removed": removed,
                "failed": failed
            }
    
    def _add_reference_from_db(self, db_connector, music_id: int, music_name: str) -> bool:
        """
        从数据库直接添加参考音频到索引（不依赖音频文件）
        
        Args:
            db_connector: 数据库连接对象
            music_id: 音乐ID
            music_name: 音乐名称
            
        Returns:
            是否成功
        """
        try:
            # 获取该歌曲的所有指纹
            sql = "SELECT hash, offset FROM finger_prints WHERE music_id_fk = %s"
            db_connector.cursor.execute(sql, (music_id,))
            fingerprints = db_connector.cursor.fetchall()
            
            if not fingerprints:
                print(f"[Monitor] [分析器初始化] 数据库中未找到音乐ID {music_id} 的指纹")
                return False
            
            hashes = [(h, int(o)) for h, o in fingerprints]
            self._analyzer.matching_engine.add_reference(music_id, hashes, music_name)
            
            print(f"[Monitor] [分析器初始化] 从数据库加载参考音频: {music_name} (ID: {music_id}, 指纹数: {len(hashes)})")
            return True
            
        except Exception as e:
            print(f"[Monitor] [分析器初始化] 从数据库加载参考音频失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """清理临时文件"""
        slice_dir = "slice"
        if not os.path.exists(slice_dir):
            return 0
        
        current_time = time.time()
        deleted_count = 0
        
        try:
            for filename in os.listdir(slice_dir):
                filepath = os.path.join(slice_dir, filename)
                if os.path.isfile(filepath):
                    file_mtime = os.path.getmtime(filepath)
                    age_hours = (current_time - file_mtime) / 3600
                    
                    if age_hours > max_age_hours:
                        try:
                            os.remove(filepath)
                            deleted_count += 1
                        except Exception as e:
                            print(f"[Monitor] 删除文件失败 {filepath}: {e}")
        
        except Exception as e:
            print(f"[Monitor] 清理临时文件失败: {e}")
        
        return deleted_count
    
    async def _init_analyzer(self):
        """初始化长音频分析器"""
        try:
            from core.long_audio_analyzer import LongAudioAnalyzer, AnalyzerConfig
            from core.shazam.database.connector import MySQLConnector
            
            # 创建配置（使用固定默认值）
            config = AnalyzerConfig(
                window_size=10.0,      # 固定窗口大小 10秒
                step_size=5.0,         # 固定步长 5秒
                match_threshold=10,    # 固定匹配阈值 10
                use_parallel=True,
                max_workers=4
            )
            
            # 创建数据库连接
            db_connector = MySQLConnector()
            
            # 创建分析器（不传递db_connector，避免自动构建全量索引）
            self._analyzer = LongAudioAnalyzer(config=config, db_connector=None)
            
            # 如果用户指定了参考音频，添加到索引
            if self.reference_audios:
                print(f"[Monitor] [分析器初始化] 用户指定了 {len(self.reference_audios)} 个参考音频，只加载这些音频到索引")
                for ref_path in self.reference_audios:
                    music_id = None
                    music_name = None
                    
                    # 首先尝试通过完整路径查找
                    if os.path.exists(ref_path):
                        music_id = db_connector.find_music_by_music_path(ref_path)
                    else:
                        # 文件不存在，尝试通过文件名（不含扩展名）查找
                        ref_name = os.path.splitext(os.path.basename(ref_path))[0]
                        print(f"[Monitor] [分析器初始化] 参考音频文件不存在，尝试使用名称查找: {ref_name}")
                        music_id = db_connector.find_music_by_music_name(ref_name)
                        if music_id:
                            print(f"[Monitor] [分析器初始化] 通过名称找到音乐ID: {music_id}")
                    
                    if music_id:
                        music_name = db_connector.find_music_name_by_music_id(music_id)
                        # 使用数据库中存储的路径替代用户提供的（可能不存在的）路径
                        stored_path = db_connector.find_music_path_by_music_id(music_id)
                        if stored_path and os.path.exists(stored_path):
                            success = self._analyzer.add_reference(stored_path, music_id, music_name)
                        else:
                            # 如果数据库中的路径也不存在，尝试使用用户提供的名称直接添加
                            print(f"[Monitor] [分析器初始化] 数据库路径不存在，尝试从数据库重新构建索引项")
                            success = self._add_reference_from_db(db_connector, music_id, music_name)
                        
                        if success:
                            print(f"[Monitor] [分析器初始化] 添加参考音频到索引: {music_name} (ID: {music_id})")
                        else:
                            print(f"[Monitor] [分析器初始化] 添加参考音频失败: {ref_path}")
                    else:
                        print(f"[Monitor] [分析器初始化] 未找到参考音频: {ref_path}")
            else:
                # 用户未指定参考音频，加载所有歌曲到索引
                print(f"[Monitor] [分析器初始化] 用户未指定参考音频，加载所有歌曲到索引")
                self._analyzer.matching_engine.build_index_from_database(db_connector)
            
            print(f"[Monitor] [分析器初始化] 长音频分析器初始化完成")
            
        except Exception as e:
            print(f"[Monitor] [分析器初始化] 初始化分析器失败: {e}")
            import traceback
            traceback.print_exc()
            self._analyzer = None
    
    async def _init_detector(self):
        """初始化检测器（常驻内存）"""
        try:
            from algorithms.factory import create_detector
            import torch
            import time
            
            print(f"[Monitor] [模型初始化] 开始初始化检测器...")
            print(f"[Monitor] [模型初始化] 目标算法: {self.algorithm}")
            print(f"[Monitor] [模型初始化] 目标设备: {self.device}")
            print(f"[Monitor] [模型初始化] 当前检测器状态: {'已加载' if self._detector else '未加载'}")
            if self._detector:
                print(f"[Monitor] [模型初始化] 当前算法: {self._current_algorithm}")
            
            # 如果检测器已存在且算法相同，直接返回
            if self._detector is not None and self._current_algorithm == self.algorithm:
                print(f"[Monitor] [模型初始化] 检测器已加载且算法匹配，复用现有模型")
                return
            
            # 释放旧检测器
            if self._detector is not None:
                print(f"[Monitor] [模型初始化] 开始释放旧检测器: {self._current_algorithm}")
                release_start = time.time()
                self._detector.release()
                release_time = time.time() - release_start
                print(f"[Monitor] [模型初始化] 旧检测器释放完成，耗时: {release_time:.3f}s")
                
                print(f"[Monitor] [模型初始化] 清理GPU缓存...")
                cache_start = time.time()
                torch.cuda.empty_cache()
                cache_time = time.time() - cache_start
                print(f"[Monitor] [模型初始化] GPU缓存清理完成，耗时: {cache_time:.3f}s")
                self._detector = None
                self._current_algorithm = None
            
            # 创建新检测器
            print(f"[Monitor] [模型初始化] 开始创建检测器: {self.algorithm}")
            create_start = time.time()
            self._detector = create_detector(self.algorithm, device=self.device)
            create_time = time.time() - create_start
            print(f"[Monitor] [模型初始化] 检测器创建完成，耗时: {create_time:.3f}s")
            
            # 加载模型
            print(f"[Monitor] [模型初始化] 开始加载模型权重...")
            load_start = time.time()
            self._detector.load_model()
            load_time = time.time() - load_start
            print(f"[Monitor] [模型初始化] 模型权重加载完成，耗时: {load_time:.3f}s")
            
            self._current_algorithm = self.algorithm
            total_time = create_time + load_time
            print(f"[Monitor] [模型初始化] 检测器初始化完成!")
            print(f"[Monitor] [模型初始化] 算法: {self.algorithm}")
            print(f"[Monitor] [模型初始化] 设备: {self.device}")
            print(f"[Monitor] [模型初始化] 总耗时: {total_time:.3f}s")
            
        except Exception as e:
            print(f"[Monitor] [模型初始化] 初始化检测器失败: {e}")
            import traceback
            traceback.print_exc()
            self._detector = None
            self._current_algorithm = None
    
    async def _process_long_audio(self, file_path: str):
        """使用长音频分析器处理单个文件，并对匹配片段进行异常检测"""
        print(f"[Monitor Debug] _process_long_audio 被调用: {file_path}")
        
        if file_path in self.processed_files:
            print(f"[Monitor Debug] 文件已处理过，跳过: {file_path}")
            return
        
        filename = os.path.basename(file_path)
        print(f"[Monitor Debug] 开始处理文件: {filename}")
        
        await websocket_manager.broadcast({
            "type": "monitor_log",
            "data": {
                "level": "info",
                "message": f"🎵 开始处理长音频: {filename}"
            }
        })
        
        try:
            print(f"[Monitor Debug] 检查分析器: analyzer={self._analyzer is not None}")
            if self._analyzer is None:
                raise Exception("长音频分析器未初始化")
            
            # 获取当前事件循环，用于线程安全地调用异步函数
            loop = asyncio.get_event_loop()
            
            # 使用长音频分析器分析
            def progress_callback(message: str, progress: float):
                # 只在关键进度点发送日志
                if progress in [0.05, 0.25, 0.50, 0.90, 1.0]:
                    # 使用 call_soon_threadsafe 在主事件循环中执行
                    future = asyncio.run_coroutine_threadsafe(
                        websocket_manager.broadcast({
                            "type": "monitor_log",
                            "data": {
                                "level": "info",
                                "message": f"[{filename}] {message}"
                            }
                        }),
                        loop
                    )
            
            # 在线程池中执行分析（避免阻塞事件循环）
            print(f"[Monitor Debug] 开始执行长音频分析...")
            result = await loop.run_in_executor(
                None,  # 使用默认线程池
                lambda: self._analyzer.analyze(file_path, progress_callback)
            )
            print(f"[Monitor Debug] 分析完成: segments={len(result.segment_matches) if result.segment_matches else 0}")
            
            # 处理每个匹配的片段
            if result.segment_matches:
                print(f"[Monitor Debug] 发现 {len(result.segment_matches)} 个匹配片段")
                await websocket_manager.broadcast({
                    "type": "monitor_log",
                    "data": {
                        "level": "info",
                        "message": f"🔍 发现 {len(result.segment_matches)} 个匹配片段，开始批量异常检测..."
                    }
                })
                
                # 批量处理所有片段（只加载一次模型）
                await self._process_segments_batch(file_path, filename, result.segment_matches)
            else:
                print(f"[Monitor Debug] 未找到匹配片段")
                await websocket_manager.broadcast({
                    "type": "monitor_log",
                    "data": {
                        "level": "warning",
                        "message": f"⚠️ {filename}: 未找到匹配的参考音频"
                    }
                })
                
                # 添加未匹配记录到结果列表
                no_match_result = {
                    "timestamp": datetime.now().isoformat(),
                    "filename": filename,
                    "filepath": file_path,
                    "anomaly_score": 0,
                    "is_anomaly": False,
                    "status": "未匹配",
                    "original_path": None,
                    "overlay_path": None,
                    "heatmap_path": None,
                    "segment_info": None
                }
                self.detection_results.append(no_match_result)
                self.total_processed += 1
                
                # 广播更新，包含未匹配结果
                await websocket_manager.broadcast({
                    "type": "monitor_update",
                    "data": {
                        "total_processed": self.total_processed,
                        "anomaly_count": self.anomaly_count,
                        "latest_result": no_match_result
                    }
                })
            
            # 标记为已处理
            print(f"[Monitor Debug] 标记文件为已处理: {file_path}")
            self.processed_files.add(file_path)
            
        except Exception as e:
            print(f"[Monitor] 处理长音频失败: {e}")
            import traceback
            traceback.print_exc()
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "error",
                    "message": f"❌ {filename} 处理失败: {str(e)}"
                }
            })
            # 即使失败也标记为已处理，避免重复尝试
            self.processed_files.add(file_path)
    
    async def _process_segment(self, file_path: str, filename: str, segment):
        """处理单个匹配的片段：切分、异常检测、生成热力图"""
        print(f"[Monitor Debug] _process_segment 被调用: {filename}, segment={segment.music_name}")
        try:
            import os
            import librosa
            import soundfile as sf
            
            # 1. 切分音频片段
            start_time = segment.start_time
            end_time = segment.end_time
            duration = end_time - start_time
            print(f"[Monitor Debug] 切分参数: start={start_time}, end={end_time}, duration={duration}")
            
            # 加载音频并切分
            print(f"[Monitor Debug] 开始加载音频...")
            audio, sr = librosa.load(file_path, sr=22050, offset=start_time, duration=duration)
            print(f"[Monitor Debug] 音频加载完成: len={len(audio)}, sr={sr}")
            
            # 创建临时文件保存切分后的音频
            temp_dir = os.path.join("slice", "monitor")
            os.makedirs(temp_dir, exist_ok=True)
            print(f"[Monitor Debug] 临时目录: {temp_dir}")
            
            segment_filename = f"{os.path.splitext(filename)[0]}_{segment.music_name}_{int(start_time)}s_{int(end_time)}s.wav"
            segment_path = os.path.join(temp_dir, segment_filename)
            print(f"[Monitor Debug] 保存切分音频: {segment_path}")
            sf.write(segment_path, audio, sr)
            print(f"[Monitor Debug] 音频保存完成")
            
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "info",
                    "message": f"🎵 切分片段: {segment_filename}"
                }
            })
            
            # 2. 生成频谱图
            print(f"[Monitor Debug] 开始生成频谱图...")
            from preprocessing import plot_spectrogram
            pic_dir = os.path.join("slice", "monitor", "picture")
            os.makedirs(pic_dir, exist_ok=True)
            
            base_name = os.path.splitext(segment_filename)[0]
            spectrogram_path = os.path.join(pic_dir, f"{base_name}.png")
            plot_spectrogram(segment_path, spectrogram_path)
            print(f"[Monitor Debug] 频谱图生成完成: {spectrogram_path}")
            
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "info",
                    "message": f"📊 生成频谱图: {os.path.basename(spectrogram_path)}"
                }
            })
            
            # 3. 进行异常检测
            print(f"[Monitor Debug] 开始异常检测...")
            # 使用工厂函数创建检测器
            from algorithms.factory import create_detector
            detector = create_detector(self.algorithm, device=self.device)
            detector.load_model()
            print(f"[Monitor Debug] 检测器创建并加载成功: {self.algorithm}")
            
            # 执行推理 - 使用predict_batch方法（单张图像也使用batch方法，因为它会生成可视化图像）
            detection_results = detector.predict_batch([spectrogram_path])
            detection_result_obj = detection_results[0] if detection_results else None
            
            if detection_result_obj is None:
                raise ValueError("检测器返回空结果")
            
            print(f"[Monitor Debug] 推理完成: score={detection_result_obj.anomaly_score}, is_anomaly={detection_result_obj.is_anomaly}")
            
            # 4. 从metadata中获取生成的图片路径
            original_path = None
            overlay_path = None
            heatmap_path = None
            if detection_result_obj.metadata:
                original_path = detection_result_obj.metadata.get('original_path')
                overlay_path = detection_result_obj.metadata.get('overlay_path')
                heatmap_path = detection_result_obj.metadata.get('heatmap_path')
                
                # 将绝对路径转换为相对路径
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                if original_path and original_path.startswith(project_root):
                    original_path = original_path[len(project_root)+1:].replace('\\', '/')
                if overlay_path and overlay_path.startswith(project_root):
                    overlay_path = overlay_path[len(project_root)+1:].replace('\\', '/')
                if heatmap_path and heatmap_path.startswith(project_root):
                    heatmap_path = heatmap_path[len(project_root)+1:].replace('\\', '/')
            
            print(f"[Monitor Debug] 图片路径: original={original_path}, overlay={overlay_path}, heatmap={heatmap_path}")
            
            # 检查文件是否生成成功
            if heatmap_path:
                full_heatmap_path = os.path.join(project_root, heatmap_path) if not os.path.isabs(heatmap_path) else heatmap_path
                exists = os.path.exists(full_heatmap_path)
                size = os.path.getsize(full_heatmap_path) if exists else 0
                print(f"[Monitor Debug] 热力图文件: exists={exists}, size={size} bytes")
            
            # 4. 创建检测结果
            detection_result = {
                "timestamp": datetime.now().isoformat(),
                "filename": segment_filename,
                "filepath": segment_path,
                "anomaly_score": float(detection_result_obj.anomaly_score),
                "is_anomaly": bool(detection_result_obj.is_anomaly),
                "status": "异常" if detection_result_obj.is_anomaly else "正常",
                "original_path": original_path,
                "overlay_path": overlay_path,
                "heatmap_path": heatmap_path,
                "segment_info": {
                    "music_name": segment.music_name,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "confidence": segment.confidence,
                    "match_ratio": segment.match_ratio
                }
            }
            
            print(f"[Monitor Debug] 创建检测结果: {detection_result}")
            
            self.detection_results.append(detection_result)
            self.total_processed += 1
            print(f"[Monitor Debug] 结果已添加到列表, total_processed={self.total_processed}")
            
            if detection_result_obj.is_anomaly:
                self.anomaly_count += 1
            
            # 发送日志
            status_icon = "⚠️" if detection_result_obj.is_anomaly else "✅"
            status_text = "异常" if detection_result_obj.is_anomaly else "正常"
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "warning" if detection_result_obj.is_anomaly else "success",
                    "message": f"{status_icon} {segment_filename}: {status_text} (分数: {detection_result_obj.anomaly_score:.4f})"
                }
            })
            
            # 广播更新
            await websocket_manager.broadcast({
                "type": "monitor_update",
                "data": {
                    "total_processed": self.total_processed,
                    "anomaly_count": self.anomaly_count,
                    "latest_result": detection_result
                }
            })
            
        except Exception as e:
            print(f"[Monitor] 处理片段失败: {e}")
            import traceback
            traceback.print_exc()
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "error",
                    "message": f"❌ 片段处理失败: {str(e)}"
                }
            })

    async def _process_segments_batch(self, file_path: str, filename: str, segments):
        """批量处理多个片段：只加载一次模型，批量推理"""
        print(f"[Monitor Debug] _process_segments_batch 被调用: {filename}, 片段数={len(segments)}")
        
        try:
            import os
            import librosa
            import soundfile as sf
            from preprocessing import plot_spectrogram
            
            # 第1步：切分所有音频片段并生成频谱图
            print(f"[Monitor Debug] 步骤1: 切分音频并生成频谱图...")
            
            temp_dir = os.path.join("slice", "monitor")
            pic_dir = os.path.join("slice", "monitor", "picture")
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(pic_dir, exist_ok=True)
            
            segment_infos = []  # 存储片段信息
            spectrogram_paths = []  # 存储频谱图路径
            
            for segment in segments:
                # 切分音频
                start_time = segment.start_time
                end_time = segment.end_time
                duration = end_time - start_time
                
                audio, sr = librosa.load(file_path, sr=22050, offset=start_time, duration=duration)
                
                segment_filename = f"{os.path.splitext(filename)[0]}_{segment.music_name}_{int(start_time)}s_{int(end_time)}s.wav"
                segment_path = os.path.join(temp_dir, segment_filename)
                sf.write(segment_path, audio, sr)
                
                # 生成频谱图
                base_name = os.path.splitext(segment_filename)[0]
                spectrogram_path = os.path.join(pic_dir, f"{base_name}.png")
                plot_spectrogram(segment_path, spectrogram_path)
                
                segment_infos.append({
                    'segment': segment,
                    'segment_filename': segment_filename,
                    'segment_path': segment_path,
                    'spectrogram_path': spectrogram_path
                })
                spectrogram_paths.append(spectrogram_path)
                
                await websocket_manager.broadcast({
                    "type": "monitor_log",
                    "data": {
                        "level": "info",
                        "message": f"🎵 切分片段: {segment_filename}"
                    }
                })
            
            print(f"[Monitor Debug] 步骤1完成: 生成了 {len(spectrogram_paths)} 个频谱图")
            
            # 第2步：使用常驻检测器进行批量推理
            print(f"[Monitor Debug] 步骤2: 使用常驻检测器批量推理...")
            
            # 确保检测器已初始化
            if self._detector is None:
                await self._init_detector()
            
            print(f"[Monitor Debug] 检测器就绪，开始批量推理 {len(spectrogram_paths)} 个样本...")
            
            # 批量推理
            detection_results = self._detector.predict_batch(spectrogram_paths)
            print(f"[Monitor Debug] 批量推理完成: 获得 {len(detection_results)} 个结果")
            
            # 第3步：处理每个片段的结果
            print(f"[Monitor Debug] 步骤3: 处理推理结果...")
            
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            
            for i, (segment_info, detection_result_obj) in enumerate(zip(segment_infos, detection_results)):
                segment = segment_info['segment']
                segment_filename = segment_info['segment_filename']
                segment_path = segment_info['segment_path']
                
                # 从metadata中获取图片路径
                original_path = None
                overlay_path = None
                heatmap_path = None
                if detection_result_obj.metadata:
                    original_path = detection_result_obj.metadata.get('original_path')
                    overlay_path = detection_result_obj.metadata.get('overlay_path')
                    heatmap_path = detection_result_obj.metadata.get('heatmap_path')
                    
                    # 转换为相对路径
                    if original_path and original_path.startswith(project_root):
                        original_path = original_path[len(project_root)+1:].replace('\\', '/')
                    if overlay_path and overlay_path.startswith(project_root):
                        overlay_path = overlay_path[len(project_root)+1:].replace('\\', '/')
                    if heatmap_path and heatmap_path.startswith(project_root):
                        heatmap_path = heatmap_path[len(project_root)+1:].replace('\\', '/')
                
                # 创建检测结果
                detection_result = {
                    "timestamp": datetime.now().isoformat(),
                    "filename": segment_filename,
                    "filepath": segment_path,
                    "anomaly_score": float(detection_result_obj.anomaly_score),
                    "is_anomaly": bool(detection_result_obj.is_anomaly),
                    "status": "异常" if detection_result_obj.is_anomaly else "正常",
                    "original_path": original_path,
                    "overlay_path": overlay_path,
                    "heatmap_path": heatmap_path,
                    "segment_info": {
                        "music_name": segment.music_name,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "confidence": segment.confidence,
                        "match_ratio": segment.match_ratio
                    }
                }
                
                self.detection_results.append(detection_result)
                self.total_processed += 1
                
                if detection_result_obj.is_anomaly:
                    self.anomaly_count += 1
                
                # 发送日志
                status_icon = "⚠️" if detection_result_obj.is_anomaly else "✅"
                status_text = "异常" if detection_result_obj.is_anomaly else "正常"
                await websocket_manager.broadcast({
                    "type": "monitor_log",
                    "data": {
                        "level": "warning" if detection_result_obj.is_anomaly else "success",
                        "message": f"{status_icon} {segment_filename}: {status_text} (分数: {detection_result_obj.anomaly_score:.4f})"
                    }
                })
                
                # 广播更新
                await websocket_manager.broadcast({
                    "type": "monitor_update",
                    "data": {
                        "total_processed": self.total_processed,
                        "anomaly_count": self.anomaly_count,
                        "latest_result": detection_result
                    }
                })
            
            print(f"[Monitor Debug] 批量处理完成: 处理了 {len(segment_infos)} 个片段")
            
        except Exception as e:
            print(f"[Monitor] 批量处理片段失败: {e}")
            import traceback
            traceback.print_exc()
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "error",
                    "message": f"❌ 批量处理失败: {str(e)}"
                }
            })


# 全局监控服务实例
monitor_service = MonitorService()
