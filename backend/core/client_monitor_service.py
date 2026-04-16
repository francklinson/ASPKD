"""
客户端检测服务
统一处理客户端上传的音频文件，与实时检测使用相同的处理流程

处理流程：
1. 接收客户端上传的文件
2. 使用长音频分析器分析文件（Shazam定位）
3. 精确定位并切分片段
4. 批量推理检测
5. 返回结果并通知客户端
"""
import os
import sys
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from collections import deque

# 确保导入路径正确
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# 从配置文件读取虚拟环境路径
config_path = os.path.join(project_root, "config", "config.yaml")
venv_site_packages = None
if os.path.exists(config_path):
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        venv_config = config.get('python_venv', {})

        # 优先使用直接指定的路径
        if venv_config.get('site_packages_path'):
            venv_site_packages = os.path.join(project_root, venv_config['site_packages_path'])
        else:
            # 使用模板构建路径
            venv_dir = venv_config.get('venv_dir', '.venv')
            template = venv_config.get('site_packages_template', 'lib/python{python_version}/site-packages')
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            site_packages_relative = template.format(python_version=python_version)
            venv_site_packages = os.path.join(project_root, venv_dir, site_packages_relative)
    except Exception as e:
        print(f"[ClientMonitor] Warning: Failed to load venv config: {e}")

# 如果配置读取失败，使用默认路径
if not venv_site_packages:
    venv_site_packages = os.path.join(project_root, ".venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")

if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

from backend.core.websocket import websocket_manager


class ClientDetectionService:
    """客户端检测服务 - 与实时检测使用相同的处理流程"""
    
    def __init__(self):
        # 配置
        self.algorithm: str = "dinomaly_dinov3_small"
        self.device: str = "auto"
        self.reference_audios: List[str] = []
        
        # 长音频分析器
        self._analyzer = None
        
        # 检测器（常驻内存）
        self._detector = None
        self._current_algorithm: Optional[str] = None
        
        # 结果存储
        self.detection_results: deque = deque(maxlen=1000)
        self.total_processed: int = 0
        self.anomaly_count: int = 0
        
        # 批量处理
        self._pending_files: List[Dict] = []  # 存储文件信息和客户端信息
        self._batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
        self._batch_delay: float = 3.0  # 等待3秒收集文件
    
    async def initialize(self, algorithm: str, device: str, reference_audios: List[str]):
        """初始化服务"""
        self.algorithm = algorithm
        self.device = device
        self.reference_audios = reference_audios or []
        
        # 初始化长音频分析器
        await self._init_analyzer()
        
        # 初始化检测器
        await self._init_detector()
        
        print(f"[ClientDetection] 服务初始化完成: algorithm={algorithm}, device={device}")
    
    async def _init_analyzer(self):
        """初始化长音频分析器"""
        try:
            from core.precise_segment_locator.adapter import PreciseSegmentLocatorAdapter
            from core.long_audio_analyzer import AnalyzerConfig
            from core.shazam.database.connector import MySQLConnector

            # 创建配置（使用固定默认值）
            # 注意：AnalyzerConfig 参数与 SegmentLocatorConfig 不同
            config = AnalyzerConfig(
                window_size=10.0,       # 窗口大小（秒）
                step_size=5.0,          # 步长（秒）
                match_threshold=10,     # 匹配阈值
                min_match_ratio=0.05,   # 最小匹配比例
                time_tolerance=2.0,     # 时间容差（秒）
                min_segment_duration=5.0,  # 最小片段时长
                use_parallel=True,      # 是否使用并行处理
                max_workers=4,          # 最大线程数
                batch_size=10           # 批处理大小
            )

            # 创建数据库连接
            db_connector = MySQLConnector()

            # 创建分析器
            self._analyzer = PreciseSegmentLocatorAdapter(config=config, db_connector=db_connector)

            # 如果用户指定了参考音频，添加到索引
            if self.reference_audios:
                print(f"[ClientDetection] [分析器初始化] 用户指定了 {len(self.reference_audios)} 个参考音频，只加载这些音频到索引")
                for ref_path in self.reference_audios:
                    music_id = None
                    music_name = None

                    # 首先尝试通过完整路径查找
                    if os.path.exists(ref_path):
                        music_id = db_connector.find_music_by_music_path(ref_path)
                    else:
                        # 文件不存在，尝试通过文件名（不含扩展名）查找
                        ref_name = os.path.splitext(os.path.basename(ref_path))[0]
                        print(f"[ClientDetection] [分析器初始化] 参考音频文件不存在，尝试使用名称查找: {ref_name}")
                        music_id = db_connector.find_music_by_music_name(ref_name)
                        if music_id:
                            print(f"[ClientDetection] [分析器初始化] 通过名称找到音乐ID: {music_id}")

                    if music_id:
                        music_name = db_connector.find_music_name_by_music_id(music_id)
                        # PreciseSegmentLocator 直接使用数据库加载
                        success = self._analyzer.locator.add_reference(music_id, music_name)

                        if success:
                            print(f"[ClientDetection] [分析器初始化] 添加参考音频到索引: {music_name} (ID: {music_id})")
                        else:
                            print(f"[ClientDetection] [分析器初始化] 添加参考音频失败: {ref_path}")
                    else:
                        print(f"[ClientDetection] [分析器初始化] 未找到参考音频: {ref_path}")
            else:
                # 用户未指定参考音频，加载所有歌曲到索引
                print(f"[ClientDetection] [分析器初始化] 用户未指定参考音频，加载所有歌曲到索引")
                # 获取所有参考音频ID并添加
                sql = "SELECT music_id, music_name FROM music"
                db_connector.cursor.execute(sql)
                musics = db_connector.cursor.fetchall()
                for music_id, music_name in musics:
                    self._analyzer.locator.add_reference(music_id, music_name)

            print("[ClientDetection] 长音频分析器初始化完成")

        except Exception as e:
            print(f"[ClientDetection] 初始化长音频分析器失败: {e}")
            import traceback
            traceback.print_exc()
            self._analyzer = None
    
    async def _init_detector(self):
        """初始化检测器（常驻内存）"""
        try:
            # 延迟导入，避免启动时加载
            from algorithms.factory import create_detector
            import torch
            
            if self._detector is None or self._current_algorithm != self.algorithm:
                print(f"[ClientDetection] 初始化检测器: {self.algorithm}")
                
                # 解析设备
                device = self.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # 创建检测器
                self._detector = create_detector(self.algorithm, device=device)
                self._current_algorithm = self.algorithm
                print("[ClientDetection] 检测器初始化完成")
            
        except Exception as e:
            print(f"[ClientDetection] 初始化检测器失败: {e}")
            import traceback
            traceback.print_exc()
            self._detector = None
    
    async def process_client_files(self, files: List, client_id: str, client_name: str) -> str:
        """
        处理客户端上传的文件
        
        Args:
            files: 上传的文件列表
            client_id: 客户端ID
            client_name: 客户端名称
            
        Returns:
            task_id: 任务ID
        """
        # 保存上传的文件到临时目录
        temp_dir = os.path.join("uploads", "clients", client_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        saved_files = []
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append({
                "path": file_path,
                "filename": file.filename,
                "client_id": client_id,
                "client_name": client_name
            })
        
        # 加入批量处理队列
        async with self._batch_lock:
            self._pending_files.extend(saved_files)
            queue_length = len(self._pending_files)
            
            # 启动批量处理任务
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._batch_process())
        
        await websocket_manager.broadcast({
            "type": "monitor_log",
            "data": {
                "level": "info",
                "message": f"📥 客户端 {client_name} 上传 {len(files)} 个文件，队列长度: {queue_length}"
            }
        })
        
        return f"client_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def _batch_process(self):
        """批量处理队列中的文件"""
        # 等待一段时间收集文件
        await asyncio.sleep(self._batch_delay)
        
        async with self._batch_lock:
            if not self._pending_files:
                return
            
            files_to_process = self._pending_files.copy()
            self._pending_files.clear()
        
        # 批量处理
        await self._process_files_batch(files_to_process)
    
    async def _process_files_batch(self, file_infos: List[Dict]):
        """批量处理文件（与实时检测一致的处理流程）"""
        if not file_infos:
            return
        
        batch_size = len(file_infos)
        filenames = [f["filename"] for f in file_infos]
        
        await websocket_manager.broadcast({
            "type": "monitor_log",
            "data": {
                "level": "info",
                "message": f"🚀 开始处理客户端文件 {batch_size} 个: {', '.join(filenames[:3])}{'...' if batch_size > 3 else ''}"
            }
        })
        
        try:
            # 第1阶段：分析所有文件，收集匹配片段
            print(f"[ClientDetection] 第1阶段：分析文件，收集匹配片段...")
            all_segments = []
            files_with_matches = []
            files_no_matches = []
            
            for file_info in file_infos:
                file_path = file_info["path"]
                filename = file_info["filename"]
                client_name = file_info["client_name"]
                client_id = file_info.get("client_id")
                
                segments = await self._analyze_file_only(file_path, filename, client_name, client_id)
                
                if segments:
                    all_segments.extend(segments)
                    files_with_matches.append((file_info, len(segments)))
                else:
                    files_no_matches.append(file_info)
            
            print(f"[ClientDetection] 第1阶段完成：{len(files_with_matches)} 个文件有匹配，共 {len(all_segments)} 个片段")
            
            # 处理无匹配片段的文件
            for file_info in files_no_matches:
                await self._handle_no_match(file_info)
                self.total_processed += 1
            
            # 如果有匹配片段，统一处理
            if all_segments:
                # 第2阶段：统一切分并生成频谱图
                print(f"[ClientDetection] 第2阶段：切分并生成频谱图...")
                spectrogram_data = await self._generate_spectrograms_batch(all_segments)
                
                # 第3阶段：统一执行模型推理
                print(f"[ClientDetection] 第3阶段：批量推理...")
                detection_results = await self._run_batch_inference(spectrogram_data)
                
                # 第4阶段：处理所有结果
                print(f"[ClientDetection] 第4阶段：处理结果...")
                await self._process_all_detection_results(spectrogram_data, detection_results)
                
                # 更新统计
                for file_info, _ in files_with_matches:
                    self.total_processed += 1
            
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "success",
                    "message": f"✅ 客户端文件处理完成: {batch_size} 个文件"
                }
            })
            
        except Exception as e:
            print(f"[ClientDetection] 批量处理失败: {e}")
            import traceback
            traceback.print_exc()
            await websocket_manager.broadcast({
                "type": "monitor_log",
                "data": {
                    "level": "error",
                    "message": f"❌ 客户端文件处理失败: {str(e)}"
                }
            })
    
    async def _analyze_file_only(self, file_path: str, filename: str, client_name: str, client_id: str = None) -> List[Dict]:
        """仅分析文件，返回匹配片段信息"""
        print(f"[ClientDetection] 分析文件: {filename}")
        
        await websocket_manager.broadcast({
            "type": "monitor_log",
            "data": {
                "level": "info",
                "message": f"🎵 [{client_name}] 分析: {filename}"
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
                        'client_name': client_name,
                        'client_id': client_id,
                        'segment': segment
                    })
                print(f"[ClientDetection] {filename}: 发现 {len(segments)} 个匹配片段")
            else:
                print(f"[ClientDetection] {filename}: 未找到匹配片段")
            
            return segments
            
        except Exception as e:
            print(f"[ClientDetection] 分析文件失败 {filename}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _handle_no_match(self, file_info: Dict):
        """处理无匹配片段的文件"""
        filename = file_info["filename"]
        client_name = file_info["client_name"]
        file_path = file_info["path"]
        
        await websocket_manager.broadcast({
            "type": "monitor_log",
            "data": {
                "level": "warning",
                "message": f"⚠️ [{client_name}] {filename}: 未找到匹配的参考音频"
            }
        })
        
        no_match_result = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "filepath": file_path,
            "client_name": client_name,
            "anomaly_score": 0.0,
            "is_anomaly": False,
            "status": "未匹配",
            "original_path": None,
            "overlay_path": None,
            "heatmap_path": None,
            "segment_info": None
        }
        self.detection_results.append(no_match_result)
        
        # 通知客户端 - 使用 monitor_update 与实时检测对齐
        await websocket_manager.broadcast({
            "type": "monitor_update",
            "data": {
                "total_processed": self.total_processed,
                "anomaly_count": self.anomaly_count,
                "latest_result": no_match_result
            }
        })
    
    async def _generate_spectrograms_batch(self, all_segments: List[Dict]) -> List[Dict]:
        """批量生成频谱图"""
        import librosa
        import soundfile as sf
        from preprocessing import plot_spectrogram
        from core.shazam.api import AudioFingerprinter
        
        temp_dir = os.path.join("uploads", "clients", "segments")
        pic_dir = os.path.join("uploads", "clients", "pictures")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(pic_dir, exist_ok=True)
        
        spectrogram_data = []
        fingerprinter = AudioFingerprinter()
        
        try:
            for seg_info in all_segments:
                file_path = seg_info['file_path']
                filename = seg_info['filename']
                segment = seg_info['segment']
                
                # 使用 Shazam 精确定位
                ref_name = segment.music_name
                location = fingerprinter.locate(
                    long_audio_path=file_path,
                    reference_name=ref_name,
                    threshold=10
                )
                
                if location.found:
                    if location.start_time < 0:
                        start_time = -location.start_time
                    else:
                        start_time = location.start_time
                else:
                    start_time = segment.start_time
                
                if start_time < 0:
                    start_time = 0.0
                
                # 固定切分10秒
                duration = 10.0
                
                try:
                    audio, sr = librosa.load(file_path, sr=22050, offset=start_time, duration=duration)
                    
                    segment_filename = f"{os.path.splitext(filename)[0]}_{ref_name}_{int(start_time)}s.wav"
                    segment_path = os.path.join(temp_dir, segment_filename)
                    sf.write(segment_path, audio, sr)
                    
                    # 生成频谱图
                    base_name = os.path.splitext(segment_filename)[0]
                    spectrogram_path = os.path.join(pic_dir, f"{base_name}.png")
                    plot_spectrogram(segment_path, spectrogram_path)
                    
                    spectrogram_data.append({
                        **seg_info,
                        'segment_filename': segment_filename,
                        'segment_path': segment_path,
                        'spectrogram_path': spectrogram_path,
                        'start_time': start_time,
                        'duration': duration
                    })
                except Exception as e:
                    print(f"[ClientDetection] 切分文件失败 {filename}: {e}")
                    continue
        finally:
            fingerprinter.close()
        
        return spectrogram_data
    
    async def _run_batch_inference(self, spectrogram_data: List[Dict]) -> List:
        """批量推理"""
        if not self._detector:
            raise Exception("检测器未初始化")
        
        image_paths = [item['spectrogram_path'] for item in spectrogram_data]
        
        await websocket_manager.broadcast({
            "type": "monitor_log",
            "data": {
                "level": "info",
                "message": f"🧠 批量推理 {len(image_paths)} 个样本..."
            }
        })
        
        # 使用 predict_batch 进行批量推理（与 monitor_service 保持一致）
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._detector.predict_batch(image_paths)
        )
        
        return results
    
    async def _process_all_detection_results(self, spectrogram_data: List[Dict], detection_results: List):
        """处理所有检测结果"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # 导入 client_manager 用于更新客户端统计
        from backend.api.client_monitor import client_manager
        
        # 用于统计每个客户端的异常数
        client_anomaly_counts = {}
        
        for spec_info, detection_result_obj in zip(spectrogram_data, detection_results):
            try:
                filename = spec_info['filename']
                client_name = spec_info['client_name']
                client_id = spec_info.get('client_id')
                file_path = spec_info['file_path']
                segment = spec_info['segment']
                
                # 解析结果 - 支持多种返回类型
                metadata = {}
                if isinstance(detection_result_obj, dict):
                    anomaly_score = detection_result_obj.get('anomaly_score', 0)
                    is_anomaly = detection_result_obj.get('is_anomaly', False)
                    metadata = detection_result_obj.get('metadata', {})
                elif hasattr(detection_result_obj, 'anomaly_score') and hasattr(detection_result_obj, 'is_anomaly'):
                    # DetectionResult 对象
                    anomaly_score = detection_result_obj.anomaly_score
                    is_anomaly = detection_result_obj.is_anomaly
                    metadata = getattr(detection_result_obj, 'metadata', {}) or {}
                else:
                    # 简单数值
                    anomaly_score = float(detection_result_obj)
                    is_anomaly = anomaly_score > 0.5
                
                # 从metadata中获取图片路径并转换为相对路径
                original_path = None
                overlay_path = None
                heatmap_path = None
                if metadata:
                    original_path = metadata.get('original_path')
                    overlay_path = metadata.get('overlay_path')
                    heatmap_path = metadata.get('heatmap_path')
                    
                    # 转换为相对路径
                    if original_path and original_path.startswith(project_root):
                        original_path = original_path[len(project_root)+1:].replace('\\', '/')
                    if overlay_path and overlay_path.startswith(project_root):
                        overlay_path = overlay_path[len(project_root)+1:].replace('\\', '/')
                    if heatmap_path and heatmap_path.startswith(project_root):
                        heatmap_path = heatmap_path[len(project_root)+1:].replace('\\', '/')
                
                # 更新统计
                if is_anomaly:
                    self.anomaly_count += 1
                    # 统计每个客户端的异常数
                    if client_id:
                        client_anomaly_counts[client_id] = client_anomaly_counts.get(client_id, 0) + 1
                
                # 构建结果 - 与实时检测对齐
                detection_result = {
                    "timestamp": datetime.now().isoformat(),
                    "filename": filename,
                    "filepath": file_path,
                    "client_name": client_name,
                    "anomaly_score": float(anomaly_score),
                    "is_anomaly": bool(is_anomaly),
                    "status": "异常" if is_anomaly else "正常",
                    "original_path": original_path,
                    "overlay_path": overlay_path,
                    "heatmap_path": heatmap_path,
                    "segment_info": {
                        "music_name": segment.music_name,
                        "start_time": spec_info.get('start_time', segment.start_time),
                        "end_time": segment.end_time,
                        "confidence": segment.confidence,
                        "match_ratio": segment.match_ratio
                    }
                }
                
                self.detection_results.append(detection_result)
                
                # 发送日志
                status_icon = "⚠️" if is_anomaly else "✅"
                status_text = "异常" if is_anomaly else "正常"
                await websocket_manager.broadcast({
                    "type": "monitor_log",
                    "data": {
                        "level": "warning" if is_anomaly else "success",
                        "message": f"{status_icon} [{client_name}] {filename}: {status_text} (分数: {anomaly_score:.4f})"
                    }
                })
                
                # 广播更新 - 使用 monitor_update 与实时检测完全对齐
                await websocket_manager.broadcast({
                    "type": "monitor_update",
                    "data": {
                        "total_processed": self.total_processed,
                        "anomaly_count": self.anomaly_count,
                        "latest_result": detection_result
                    }
                })
                
            except Exception as e:
                print(f"[ClientDetection] 处理结果失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 更新每个客户端的异常数统计
        for client_id, anomaly_count in client_anomaly_counts.items():
            try:
                # 获取当前客户端信息
                client = client_manager.get_client(client_id)
                if client:
                    # 累加异常数（而不是覆盖）
                    new_anomaly_count = client.anomaly_detected + anomaly_count
                    await client_manager.update_stats(client_id, anomaly_detected=new_anomaly_count)
                    print(f"[ClientDetection] 更新客户端 {client_id} 异常数: +{anomaly_count} = {new_anomaly_count}")
            except Exception as e:
                print(f"[ClientDetection] 更新客户端异常数失败 {client_id}: {e}")
    
    async def update_config(self, algorithm: str = None, device: str = None, reference_audios: List[str] = None):
        """更新配置"""
        need_reinit_detector = False
        need_reinit_analyzer = False

        if algorithm and algorithm != self.algorithm:
            self.algorithm = algorithm
            need_reinit_detector = True

        if device and device != self.device:
            self.device = device
            need_reinit_detector = True

        if reference_audios is not None:
            # 检查参考音频是否发生变化
            old_refs = set(self.reference_audios)
            new_refs = set(reference_audios)
            if old_refs != new_refs:
                self.reference_audios = reference_audios
                need_reinit_analyzer = True
                print(f"[ClientDetection] 参考音频变更，从 {len(old_refs)} 个变为 {len(new_refs)} 个")

        # 如果算法或设备改变，重新初始化检测器
        if need_reinit_detector:
            print(f"[ClientDetection] 配置变更，重新初始化检测器: {self.algorithm}")
            # 释放旧检测器
            if self._detector:
                try:
                    self._detector.release()
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
                self._detector = None
                self._current_algorithm = None

            # 重新初始化检测器
            await self._init_detector()

        # 如果参考音频改变，重新初始化分析器
        if need_reinit_analyzer:
            print(f"[ClientDetection] 参考音频变更，重新初始化分析器")
            # 释放旧分析器
            if self._analyzer:
                try:
                    self._analyzer.close()
                except:
                    pass
                self._analyzer = None

            # 重新初始化分析器
            await self._init_analyzer()
    
    def get_status(self) -> Dict:
        """获取服务状态"""
        return {
            "algorithm": self.algorithm,
            "device": self.device,
            "total_processed": self.total_processed,
            "anomaly_count": self.anomaly_count,
            "reference_audios": self.reference_audios,
            "detector_loaded": self._detector is not None,
            "analyzer_loaded": self._analyzer is not None
        }


# 全局客户端检测服务实例
client_detection_service = ClientDetectionService()
