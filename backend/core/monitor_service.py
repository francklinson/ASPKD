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
            print(f"[Monitor] 检测到新文件: {file_path}")
            # 使用线程安全的方式提交到主事件循环
            asyncio.run_coroutine_threadsafe(
                self._delayed_process(file_path), 
                self.loop
            )
    
    async def _delayed_process(self, file_path: str, delay: float = 2.0):
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
        
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[AudioFileHandler] = None
        
        self.processed_files: Set[str] = set()
        self.detection_results: deque = deque(maxlen=1000)
        self.total_processed: int = 0
        self.anomaly_count: int = 0
        self.start_time: Optional[datetime] = None
        
        self._scan_task: Optional[asyncio.Task] = None
    
    async def start(
        self,
        directory: str,
        interval: int = 30,
        algorithm: str = "dinomaly_dinov3_small",
        device: str = "auto",
        detect_existing: bool = False,
        file_extensions: List[str] = None
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
        self.start_time = datetime.now()
        
        # 扫描已有文件
        existing_files = self._get_audio_files()
        
        if detect_existing:
            # 创建批量检测任务
            if existing_files:
                await task_manager.create_batch_task(
                    file_paths=existing_files,
                    algorithm=algorithm,
                    device=device
                )
                self.total_processed += len(existing_files)
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
                    for file_path in new_files:
                        await self.process_file(file_path)
                
            except Exception as e:
                print(f"[Monitor] 扫描出错: {e}")
    
    async def process_file(self, file_path: str):
        """处理单个文件"""
        if file_path in self.processed_files:
            return
        
        result = None
        try:
            # 创建检测任务
            task_id = await task_manager.create_batch_task(
                file_paths=[file_path],
                algorithm=self.algorithm,
                device=self.device
            )
            
            # 等待任务完成
            while True:
                result = task_manager.get_task_result(task_id)
                if result["status"] in ["completed", "failed"]:
                    break
                await asyncio.sleep(0.5)
            
            # 记录结果
            if result["status"] == "completed" and result["results"]:
                for r in result["results"]:
                    self.detection_results.append({
                        "timestamp": datetime.now().isoformat(),
                        **r
                    })
                    
                    if r.get("is_anomaly"):
                        self.anomaly_count += 1
            
            print(f"[Monitor] 文件处理完成: {os.path.basename(file_path)}, 结果: {result.get('status')}")
            
        except Exception as e:
            print(f"[Monitor] 处理文件失败 {file_path}: {e}")
            result = {"status": "failed", "error": str(e), "results": []}
        
        finally:
            # 确保文件被记录，避免重复处理
            self.processed_files.add(file_path)
            self.total_processed += 1
        
        # 广播更新
        await websocket_manager.broadcast({
            "type": "monitor_update",
            "data": {
                "total_processed": self.total_processed,
                "anomaly_count": self.anomaly_count,
                "latest_result": result.get("results", [{}])[0] if result and result.get("results") else None
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


# 全局监控服务实例
monitor_service = MonitorService()
