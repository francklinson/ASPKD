"""
任务管理器
管理检测任务的队列和执行
"""
import os
import time
import uuid
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from backend.core.websocket import websocket_manager


@dataclass
class Task:
    """任务数据类"""
    id: str
    status: str  # pending, running, completed, failed, cancelled
    algorithm: str
    device: str
    files: List[str]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_file: Optional[str] = None
    results: List[dict] = field(default_factory=list)
    error: Optional[str] = None
    save_results: bool = True
    reference_audio: Optional[str] = None  # 指定的参考音频路径


class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running: bool = False
        self.current_task: Optional[str] = None
        self.executor = ThreadPoolExecutor(max_workers=1)  # 串行执行，避免GPU内存溢出
        self.worker_task: Optional[asyncio.Task] = None
        
        # 导入项目模块
        self._import_modules()
    
    def _import_modules(self):
        """导入项目依赖模块"""
        try:
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys.path.insert(0, project_root)
            
            # 添加虚拟环境 site-packages 到路径
            venv_site_packages = os.path.join(project_root, ".venv", "lib", "python3.12", "site-packages")
            if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
                sys.path.insert(0, venv_site_packages)
            
            from algorithms import create_detector
            from core import ConfigManager
            from preprocessing import Preprocessor
            
            self.create_detector = create_detector
            self.ConfigManager = ConfigManager
            self.Preprocessor = Preprocessor
            
            # 加载配置
            config_path = os.path.join(project_root, "config", "config.yaml")
            self.config = ConfigManager(config_path)
            
            ref_file = self.config.config.get('preprocessing', {}).get('ref_file', 'ref/渡口片段10s.wav')
            if not os.path.isabs(ref_file):
                ref_file = os.path.join(project_root, ref_file)
            
            # 获取预处理配置（强制使用 shazam 模式，非 shazam 模式已屏蔽）
            split_method = 'shazam'
            shazam_config = self.config.config.get('preprocessing', {}).get('shazam', {})
            shazam_threshold = shazam_config.get('threshold', 10)
            shazam_auto_match = shazam_config.get('auto_match', False)
            max_workers = shazam_config.get('max_workers', 1)
            
            self.preprocessor = Preprocessor(
                ref_file=ref_file,
                split_method=split_method,
                shazam_threshold=shazam_threshold,
                shazam_auto_match=shazam_auto_match,
                max_workers=max_workers
            )
            self.detector = None
            self.current_algorithm = None
            
        except Exception as e:
            print(f"[TaskManager] 导入模块失败: {e}")
            raise
    
    async def initialize(self):
        """初始化任务管理器"""
        self.running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        print("[TaskManager] 任务管理器已启动")
    
    async def cleanup(self):
        """清理资源"""
        self.running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        if self.detector:
            self.detector.release()
            import torch
            torch.cuda.empty_cache()
        
        self.executor.shutdown(wait=True)
        print("[TaskManager] 任务管理器已关闭")
    
    async def create_task(
        self,
        files: List,
        algorithm: str,
        device: str,
        save_results: bool = True,
        reference_audio: Optional[str] = None
    ) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        
        # 保存上传的文件
        upload_dir = os.path.join("uploads", task_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        file_paths = []
        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            file_paths.append(file_path)
        
        # 创建任务
        task = Task(
            id=task_id,
            status="pending",
            algorithm=algorithm,
            device=device,
            files=file_paths,
            created_at=datetime.now(),
            save_results=save_results,
            reference_audio=reference_audio
        )
        
        self.tasks[task_id] = task
        await self.queue.put(task_id)
        
        print(f"[TaskManager] 创建任务: {task_id}, 队列长度: {self.queue.qsize()}")
        
        return task_id
    
    async def create_batch_task(
        self,
        file_paths: List[str],
        algorithm: str,
        device: str,
        save_results: bool = True
    ) -> str:
        """创建批量任务（用于目录监控）"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            status="pending",
            algorithm=algorithm,
            device=device,
            files=file_paths,
            created_at=datetime.now(),
            save_results=save_results
        )
        
        self.tasks[task_id] = task
        await self.queue.put(task_id)
        
        return task_id
    
    async def _worker_loop(self):
        """工作线程循环"""
        while self.running:
            try:
                # 获取任务
                task_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                
                if task.status == "cancelled":
                    continue
                
                # 更新状态
                task.status = "running"
                task.started_at = datetime.now()
                self.current_task = task_id
                
                print(f"[TaskManager] 开始执行任务: {task_id}")
                
                # 执行检测
                try:
                    await self._execute_task(task)
                    task.status = "completed"
                    # 发送最终进度
                    await websocket_manager.send_progress(task_id, {
                        "progress": 100,
                        "status": "completed",
                        "message": f"检测完成！共 {len(task.results)} 个文件"
                    })
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    print(f"[TaskManager] 任务执行失败: {e}")
                    # 发送错误信息
                    await websocket_manager.send_progress(task_id, {
                        "progress": 0,
                        "status": "failed",
                        "message": f"检测失败: {str(e)}"
                    })

                task.completed_at = datetime.now()
                self.current_task = None

                # 发送完成通知
                await websocket_manager.send_result(task_id, {
                    "task_id": task_id,
                    "status": task.status,
                    "results": task.results,
                    "error": task.error,
                    "file_count": len(task.files),
                    "processed_count": len(task.results)
                })
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[TaskManager] 工作循环出错: {e}")
    
    async def _execute_task(self, task: Task):
        """执行检测任务"""
        total_files = len(task.files)
        print(f"[TaskManager] 开始执行任务 {task.id}, 算法: {task.algorithm}, 文件数: {total_files}")

        # 发送开始信息
        await websocket_manager.send_progress(task.id, {
            "progress": 0,
            "status": "preprocessing",
            "message": f"开始处理 {total_files} 个文件，算法: {task.algorithm}"
        })

        # 1. 加载模型（如果需要切换算法）
        if self.current_algorithm != task.algorithm or self.detector is None:
            import time
            print(f"[TaskManager] [模型初始化] 开始初始化检测器...")
            print(f"[TaskManager] [模型初始化] 目标算法: {task.algorithm}")
            print(f"[TaskManager] [模型初始化] 目标设备: {task.device}")
            print(f"[TaskManager] [模型初始化] 当前检测器状态: {'已加载' if self.detector else '未加载'}")
            if self.detector:
                print(f"[TaskManager] [模型初始化] 当前算法: {self.current_algorithm}")

            await websocket_manager.send_progress(task.id, {
                "progress": 0,
                "status": "preprocessing",
                "message": f"正在加载模型: {task.algorithm}..."
            })

            if self.detector:
                print(f"[TaskManager] [模型初始化] 开始释放旧检测器: {self.current_algorithm}")
                release_start = time.time()
                self.detector.release()
                release_time = time.time() - release_start
                print(f"[TaskManager] [模型初始化] 旧检测器释放完成，耗时: {release_time:.3f}s")

                print(f"[TaskManager] [模型初始化] 清理GPU缓存...")
                cache_start = time.time()
                import torch
                torch.cuda.empty_cache()
                cache_time = time.time() - cache_start
                print(f"[TaskManager] [模型初始化] GPU缓存清理完成，耗时: {cache_time:.3f}s")

            print(f"[TaskManager] [模型初始化] 开始创建检测器: {task.algorithm}")
            create_start = time.time()
            self.detector = self.create_detector(
                algorithm_name=task.algorithm,
                config_manager=self.config,
                device=task.device
            )
            create_time = time.time() - create_start
            print(f"[TaskManager] [模型初始化] 检测器创建完成，耗时: {create_time:.3f}s")

            print(f"[TaskManager] [模型初始化] 开始加载模型权重...")
            load_start = time.time()
            self.detector.load_model()
            load_time = time.time() - load_start
            print(f"[TaskManager] [模型初始化] 模型权重加载完成，耗时: {load_time:.3f}s")

            self.current_algorithm = task.algorithm
            total_time = create_time + load_time
            print(f"[TaskManager] [模型初始化] 检测器初始化完成!")
            print(f"[TaskManager] [模型初始化] 算法: {task.algorithm}")
            print(f"[TaskManager] [模型初始化] 设备: {task.device}")
            print(f"[TaskManager] [模型初始化] 总耗时: {total_time:.3f}s")

            await websocket_manager.send_progress(task.id, {
                "progress": 2,
                "status": "preprocessing",
                "message": f"模型加载完成，耗时: {total_time:.3f}s"
            })

        # 2. 预处理音频文件（批量并行处理）
        all_images = []
        file_image_map = {}

        await websocket_manager.send_progress(task.id, {
            "progress": 5,
            "status": "preprocessing",
            "message": f"开始音频预处理，共 {total_files} 个文件，使用多线程并行处理..."
        })

        # 一次性传入所有文件进行批量并行处理
        task.current_file = f"批量处理 {total_files} 个文件"
        
        # 构建原始文件名映射，避免使用 task_id 作为文件名前缀
        original_names = {}
        for file_path in task.files:
            original_names[file_path] = os.path.basename(file_path)
        
        # 根据任务指定的参考音频创建预处理器
        ref_file = task.reference_audio
        if ref_file:
            print(f"[TaskManager] 使用指定的参考音频: {ref_file}")
            await websocket_manager.send_progress(task.id, {
                "progress": 5,
                "status": "preprocessing",
                "message": f"使用参考音频: {os.path.basename(ref_file)}"
            })
        else:
            # 使用默认参考音频
            ref_file = self.config.config.get('preprocessing', {}).get('ref_file', 'ref/渡口片段10s.wav')
            if not os.path.isabs(ref_file):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                ref_file = os.path.join(project_root, ref_file)
            print(f"[TaskManager] 使用默认参考音频: {ref_file}")
        
        # 创建任务特定的预处理器（强制使用 shazam 模式，非 shazam 模式已屏蔽）
        split_method = 'shazam'
        shazam_config = self.config.config.get('preprocessing', {}).get('shazam', {})
        shazam_threshold = shazam_config.get('threshold', 10)
        shazam_auto_match = shazam_config.get('auto_match', False)
        max_workers = shazam_config.get('max_workers', 1)
        
        try:
            task_preprocessor = self.Preprocessor(
                ref_file=ref_file,
                split_method=split_method,
                shazam_threshold=shazam_threshold,
                shazam_auto_match=shazam_auto_match,
                max_workers=max_workers
            )
        except Exception as e:
            print(f"[TaskManager] 创建预处理器失败: {e}")
            await websocket_manager.send_progress(task.id, {
                "progress": 0,
                "status": "failed",
                "message": f"❌ 创建预处理器失败: {str(e)}"
            })
            raise
        
        try:
            # 批量预处理（自动使用多线程），传入 original_names 避免 task_id 作为文件名前缀
            print(f"[TaskManager] 开始预处理，使用参考音频: {ref_file}")
            result = task_preprocessor.process_audio(task.files, save_dir="slice", original_names=original_names)
            print(f"[TaskManager] 预处理完成，结果数量: {len(result)}")
            
            # 处理结果
            processed_count = 0
            failed_files = []  # 记录处理失败的文件
            
            # 发送预处理完成消息
            await websocket_manager.send_progress(task.id, {
                "progress": 25,
                "status": "preprocessing",
                "message": f"预处理完成，共 {len(result)} 个文件结果"
            })
            
            for audio_file, file_result in result.items():
                print(f"[TaskManager] 处理结果: {os.path.basename(audio_file)} -> {file_result}")
                images = []
                # 检查是否标记为失败（未找到片段）
                is_failed = isinstance(file_result, dict) and file_result.get("failed")
                
                if isinstance(file_result, dict):
                    if file_result.get("dk"):
                        images.append(file_result["dk"])
                    if file_result.get("qzgy"):
                        images.append(file_result["qzgy"])
                
                if images and not is_failed:
                    file_image_map[audio_file] = images
                    all_images.extend(images)
                    processed_count += 1
                else:
                    # 记录未找到片段的文件
                    failed_files.append(audio_file)
                    # 获取详细的错误信息
                    error_msg = ""
                    error_detail = None
                    if isinstance(file_result, dict):
                        error_detail = file_result.get("error")
                        if error_detail:
                            error_msg = f" ({error_detail})"
                    
                    # 发送失败消息到前端
                    message_text = ""
                    if error_detail and "参考音频不匹配" in error_detail:
                        message_text = f"⚠️ {os.path.basename(audio_file)}: {error_detail}"
                    else:
                        message_text = f"⚠️ {os.path.basename(audio_file)}: 未找到指定片段{error_msg}"
                    
                    print(f"[TaskManager] 发送消息到前端: {message_text}")
                    await websocket_manager.send_progress(task.id, {
                        "progress": 25,
                        "status": "preprocessing",
                        "message": message_text
                    })
                    
                    # 将失败信息添加到任务结果中（以便前端在WebSocket重连后获取）
                    task.results.append({
                        "filename": os.path.basename(audio_file),
                        "filepath": audio_file,
                        "anomaly_score": 0,
                        "is_anomaly": False,
                        "status": "预处理失败",
                        "error": error_detail or "未找到指定片段",
                        "original_path": None,
                        "overlay_path": None,
                        "heatmap_path": None
                    })
            
            # 检查哪些文件完全没有被处理（不在 result 中）
            processed_files = set(result.keys())
            for audio_file in task.files:
                if audio_file not in processed_files:
                    failed_files.append(audio_file)
                    await websocket_manager.send_progress(task.id, {
                        "progress": 25,
                        "status": "preprocessing",
                        "message": f"⚠️ {os.path.basename(audio_file)}: 未找到指定片段（使用参考音频: {os.path.basename(ref_file)}）"
                    })
            
            # 汇总处理结果
            if failed_files:
                failed_names = [os.path.basename(f) for f in failed_files[:5]]  # 最多显示5个
                failed_summary = ", ".join(failed_names)
                if len(failed_files) > 5:
                    failed_summary += f" 等共 {len(failed_files)} 个文件"
                
                await websocket_manager.send_progress(task.id, {
                    "progress": 50,
                    "status": "preprocessing",
                    "message": f"✅ 预处理完成: {processed_count}/{total_files} 个文件成功，{len(failed_files)} 个文件未找到片段（{failed_summary}），共生成 {len(all_images)} 张图片"
                })
            else:
                await websocket_manager.send_progress(task.id, {
                    "progress": 50,
                    "status": "preprocessing",
                    "message": f"✅ 预处理完成: {processed_count}/{total_files} 个文件成功，共生成 {len(all_images)} 张图片"
                })
            
        except Exception as e:
            await websocket_manager.send_progress(task.id, {
                "progress": 0,
                "status": "failed",
                "message": f"❌ 预处理失败: {str(e)}"
            })
            # 清理任务预处理器资源
            if 'task_preprocessor' in locals():
                task_preprocessor.__exit__(None, None, None)
            raise
        finally:
            # 清理任务预处理器资源
            if 'task_preprocessor' in locals():
                try:
                    task_preprocessor.__exit__(None, None, None)
                    print(f"[TaskManager] 任务预处理器资源已释放")
                except Exception as e:
                    print(f"[TaskManager] 释放预处理器资源时出错: {e}")

        print(f"[TaskManager] 预处理完成，共生成 {len(all_images)} 张图片")

        # 3. 异常检测
        if all_images:
            task.progress = 50
            await websocket_manager.send_progress(task.id, {
                "progress": task.progress,
                "status": "detecting",
                "message": f"开始异常检测，共 {len(all_images)} 张图片..."
            })
            
            # 批量推理
            await websocket_manager.send_progress(task.id, {
                "progress": 55,
                "status": "detecting",
                "message": f"正在进行批量推理 ({len(all_images)} 张图片)..."
            })

            detection_results = self.detector.predict_batch(all_images)

            # 整理结果
            current_idx = 0
            anomaly_count = 0

            for audio_file, images in file_image_map.items():
                file_results = detection_results[current_idx:current_idx + len(images)]
                current_idx += len(images)

                # 找出最高异常分数
                max_score = 0
                is_anomaly = False
                original_path = None
                overlay_path = None
                heatmap_path = None

                for result in file_results:
                    if result.anomaly_score > max_score:
                        max_score = result.anomaly_score
                        is_anomaly = result.is_anomaly
                        # 获取三种图像路径并转换为相对路径
                        if result.metadata:
                            original_path = result.metadata.get('original_path')
                            overlay_path = result.metadata.get('overlay_path')
                            heatmap_path = result.metadata.get('heatmap_path')
                            
                            # 将绝对路径转换为相对路径，便于前端访问
                            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                            
                            if original_path:
                                if original_path.startswith(project_root):
                                    original_path = original_path[len(project_root)+1:]
                                original_path = original_path.replace('\\', '/')
                            
                            if overlay_path:
                                if overlay_path.startswith(project_root):
                                    overlay_path = overlay_path[len(project_root)+1:]
                                overlay_path = overlay_path.replace('\\', '/')
                            
                            if heatmap_path:
                                if heatmap_path.startswith(project_root):
                                    heatmap_path = heatmap_path[len(project_root)+1:]
                                heatmap_path = heatmap_path.replace('\\', '/')
                                print(f"[TaskManager] 生成图像: original={original_path}, overlay={overlay_path}, heatmap={heatmap_path}, 异常: {is_anomaly}")
                            else:
                                print(f"[TaskManager] 无热力图: metadata={result.metadata is not None}, score={result.anomaly_score:.4f}")

                if is_anomaly:
                    anomaly_count += 1

                task.results.append({
                    "filename": os.path.basename(audio_file),
                    "filepath": audio_file,
                    "anomaly_score": max_score,
                    "is_anomaly": is_anomaly,
                    "status": "异常" if is_anomaly else "正常",
                    "original_path": original_path,
                    "overlay_path": overlay_path,
                    "heatmap_path": heatmap_path
                })

                task.progress = 55 + (current_idx / len(all_images)) * 40  # 检测占55%-95%
                await websocket_manager.send_progress(task.id, {
                    "progress": task.progress,
                    "current": current_idx,
                    "total": len(all_images),
                    "status": "detecting",
                    "message": f"检测中 [{current_idx}/{len(all_images)}]: {os.path.basename(audio_file)}"
                })

            # 检测完成统计
            await websocket_manager.send_progress(task.id, {
                "progress": 95,
                "status": "detecting",
                "message": f"检测完成，发现 {anomaly_count} 个异常文件，正在生成结果..."
            })

        task.progress = 100
        print(f"[TaskManager] 任务完成: {task.id}, 处理了 {len(task.results)} 个文件")
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status == "completed":
            return False
        
        task.status = "cancelled"
        return True
    
    def get_task_result(self, task_id: str) -> Optional[dict]:
        """获取任务结果"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        return {
            "task_id": task.id,
            "status": task.status,
            "progress": task.progress,
            "current_file": task.current_file,
            "results": task.results,
            "error": task.error,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        }
    
    def get_queue_position(self, task_id: str) -> int:
        """获取任务在队列中的位置"""
        # 将队列转为列表查找位置
        queue_list = list(self.queue._queue)
        try:
            return queue_list.index(task_id)
        except ValueError:
            return -1
    
    def list_tasks(self, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[dict]:
        """列出任务"""
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        # 按时间倒序
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        
        tasks = tasks[offset:offset + limit]
        
        return [{
            "id": t.id,
            "status": t.status,
            "algorithm": t.algorithm,
            "file_count": len(t.files),
            "progress": t.progress,
            "created_at": t.created_at.isoformat() if t.created_at else None
        } for t in tasks]
    
    def get_task_count(self) -> int:
        """获取总任务数"""
        return len(self.tasks)
    
    def get_running_count(self) -> int:
        """获取运行中任务数"""
        return sum(1 for t in self.tasks.values() if t.status == "running")
    
    def get_queued_count(self) -> int:
        """获取排队任务数"""
        return sum(1 for t in self.tasks.values() if t.status == "pending")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total = len(self.tasks)
        running = self.get_running_count()
        queued = self.get_queued_count()
        completed = sum(1 for t in self.tasks.values() if t.status == "completed")
        failed = sum(1 for t in self.tasks.values() if t.status == "failed")
        
        return {
            "total": total,
            "running": running,
            "queued": queued,
            "completed": completed,
            "failed": failed
        }
    
    async def cleanup_old_tasks(self, keep_days: int = 7) -> int:
        """清理旧任务"""
        cutoff_time = time.time() - (keep_days * 24 * 3600)
        to_remove = []
        
        for task_id, task in self.tasks.items():
            if task.completed_at and task.completed_at.timestamp() < cutoff_time:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        return len(to_remove)
    
    async def clear_all_tasks(self) -> int:
        """清理所有已完成/失败/取消的任务"""
        to_remove = []
        
        for task_id, task in self.tasks.items():
            # 只保留运行中和待处理的任务
            if task.status in ['completed', 'failed', 'cancelled']:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        return len(to_remove)
    
    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        if task_id not in self.tasks:
            return False
        
        del self.tasks[task_id]
        return True


# 全局任务管理器实例
task_manager = TaskManager()
