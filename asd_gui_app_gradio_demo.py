"""
统一接口版本的Web GUI应用程序
使用新的统一算法接口
"""

import gc
import os
import threading
import time
import zipfile
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional

import gradio as gr
import pandas as pd
import torch
import torch.cuda as cuda

# 可选依赖：内存监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_available_devices():
    """获取可用的计算设备列表"""
    devices = [("auto", "自动选择 (GPU优先)"), ("cpu", "CPU (纯CPU运行)")]
    
    # 添加CPU选项

    # 如果有CUDA，添加各个GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            # 截取显卡名称，避免过长
            short_name = gpu_name[:30] + "..." if len(gpu_name) > 30 else gpu_name
            devices.append((f"cuda:{i}", f"GPU {i}: {short_name}"))
    
    return devices

# 使用统一接口
from algorithms import create_detector
from core import ConfigManager
from preprocessing import Preprocessor


class LogManager:
    """日志管理模块"""
    
    def __init__(self, max_rows=8):
        self.log_messages = ""
        self.max_rows = max_rows
    
    def generate_log(self, msg):
        """给日志打时间戳，截断到最大行数"""
        self.log_messages += f"\n[{time.strftime('%H:%M:%S')}] {msg}"
        lines = self.log_messages.split('\n')
        if len(lines) > self.max_rows:
            lines = lines[-self.max_rows:]
        self.log_messages = '\n'.join(lines)
    
    def update_log(self, msg):
        """更新日志"""
        self.generate_log(msg)
        yield self.log_messages, gr.update(visible=False), None, None, None, gr.update(interactive=False), gr.update()
    
    def update_log_and_action(self, msg, run_button_action):
        """更新日志，同时传递GUI控件响应"""
        self.generate_log(msg)
        yield self.log_messages, gr.update(visible=False), None, None, None, run_button_action, gr.update()


class Export:
    """文件导出类"""
    
    @classmethod
    def create_excel_report(cls, results: List[Dict], save_dir) -> str:
        """创建Excel报告"""
        df = pd.DataFrame(results)
        df_for_excel = df[['filename', 'anomaly_score', 'is_anomaly']]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(save_dir, f"anomaly_detection_results_{timestamp}.xlsx")
        df_for_excel.to_excel(excel_path, index=False)
        return excel_path
    
    @classmethod
    def create_zip_with_results(cls, zip_path: str, excel_path, images: List[str]):
        """将Excel和图像打包成ZIP文件"""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(excel_path, os.path.basename(excel_path))
            for img in images:
                zipf.write(img, os.path.basename(img))
        return zip_path


class UnifiedAlgorithmManager:
    """
    统一算法管理器
    使用新的统一接口管理所有算法
    """
    
    def __init__(self, device: str = 'auto'):
        print(f"\n{'='*60}")
        print(f"[模型管理器] 🚀 初始化 UnifiedAlgorithmManager")
        print(f"[模型管理器] 初始设备设置: {device}")
        print(f"{'='*60}")
        
        self.detector = None
        self.algorithm_chosen = ""
        self.device = device  # 运行设备
        self._model_loaded = False  # 模型加载状态标志
        
        # 使用绝对路径加载配置
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "algorithms.yaml")
        print(f"[模型管理器] 配置文件路径: {config_path}")
        print(f"[模型管理器] 配置文件存在: {os.path.exists(config_path)}")
        
        self.config = ConfigManager(config_path)
        print(f"[模型管理器] 配置加载完成，base_dir={self.config.base_dir}")
        print(f"[模型管理器] 配置中的模型: {list(self.config.config.get('models', {}).keys())}")
        
        # 获取可用算法列表
        self.available_algorithms = self._get_gui_algorithms()
        print(f"[模型管理器] 可用算法列表: {self.available_algorithms}")
        print(f"[模型管理器] ✓ 初始化完成，当前模型: None")
        print(f"{'='*60}\n")
    
    def _get_gui_algorithms(self) -> List[str]:
        """获取GUI可用的算法列表"""
        # 只保留轻量级模型
        return [
            "dinomaly_dinov2_small",
            "dinomaly_dinov3_small",
        ]
    
    def update_algorithm(self, algorithm_choice, device: str = None):
        """更新算法"""
        print(f"\n{'='*60}")
        print(f"[模型管理器] 🔄 开始更新算法")
        print(f"[模型管理器] 目标算法: {algorithm_choice}")
        print(f"[模型管理器] 当前算法: {self.algorithm_chosen if self.algorithm_chosen else 'None'}")
        print(f"[模型管理器] 新设备: {device if device else '未指定（使用当前）'}")
        print(f"[模型管理器] 当前设备: {self.device}")
        print(f"{'='*60}")
        
        assert algorithm_choice in self.available_algorithms
        
        if self.detector is not None and algorithm_choice == self.algorithm_chosen:
            print(f"[模型管理器] ℹ 算法未改变，无需更新")
            print(f"{'='*60}\n")
            return
        
        # 如果指定了新设备，更新设备设置
        if device is not None:
            old_device = self.device
            self.device = device
            print(f"[模型管理器] 设备更新: {old_device} -> {device}")
        
        # 清理旧模型
        if self.detector is not None:
            print(f"[模型管理器] 🧹 开始清理旧模型: {self.algorithm_chosen}")
            try:
                self.detector.release()
                print(f"[模型管理器] ✓ 旧模型已释放")
            except Exception as e:
                print(f"[模型管理器] ⚠ 旧模型释放失败: {e}")
            self.detector = None
            self._model_loaded = False
            print(f"[模型管理器] ✓ 检测器引用已清空，模型状态: _model_loaded=False")
            self._clear_cuda_cache()
        else:
            print(f"[模型管理器] ℹ 无旧模型需要清理")
        
        # 使用统一接口创建新检测器
        try:
            print(f"[模型管理器] 🏗️ 开始创建新检测器...")
            model_path = self.config.get_model_path('dinomaly', 'dinov3_small')
            print(f"[模型管理器] 配置 base_dir: {self.config.base_dir}")
            print(f"[模型管理器] 模型路径: {model_path}")
            print(f"[模型管理器] 使用设备: {self.device}")
            
            self.detector = create_detector(
                algorithm_name=algorithm_choice,
                config_manager=self.config,
                device=self.device
            )
            self.algorithm_chosen = algorithm_choice
            print(f"[模型管理器] ✓ 检测器创建成功")
            print(f"[模型管理器] 算法: {self.algorithm_chosen}")
            print(f"[模型管理器] 模型路径: {self.detector.model_path}")
            print(f"[模型管理器] 实际使用设备: {self.detector.device}")
        except Exception as e:
            print(f"[模型管理器] ✗ 加载算法失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            print(f"{'='*60}\n")
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model_loaded and self.detector is not None
    
    def load_model(self):
        """加载模型并更新状态"""
        if self.detector is None:
            raise RuntimeError("检测器未初始化，请先调用 update_algorithm")
        self.detector.load_model()
        self._model_loaded = True
        print(f"[模型管理器] ✓ 模型加载状态已更新: _model_loaded=True")
    
    def unload_model(self):
        """卸载模型并更新状态"""
        if self.detector is not None:
            self.detector.release()
        self._model_loaded = False
        print(f"[模型管理器] ✓ 模型卸载状态已更新: _model_loaded=False")
    
    @staticmethod
    def _clear_cuda_cache():
        """清除GPU缓存"""
        print(f"[模型管理器] 🧹 开始清理 CUDA 缓存...")
        if torch.cuda.is_available():
            try:
                # 清理前显存状态
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**2
                    reserved = torch.cuda.memory_reserved(i) / 1024**2
                    print(f"[模型管理器]   GPU {i} 清理前: 已分配={allocated:.1f}MB, 预留={reserved:.1f}MB")
                
                torch.cuda.empty_cache()
                cuda.empty_cache()
                cuda.synchronize()
                
                # 清理后显存状态
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**2
                    reserved = torch.cuda.memory_reserved(i) / 1024**2
                    print(f"[模型管理器]   GPU {i} 清理后: 已分配={allocated:.1f}MB, 预留={reserved:.1f}MB")
                print(f"[模型管理器] ✓ CUDA 缓存清理完成")
            except Exception as e:
                print(f"[模型管理器] ⚠ 清除CUDA缓存失败: {e}")
        else:
            print(f"[模型管理器] ℹ 无 CUDA 设备，跳过显存清理")
        
        gc.collect()
        print(f"[模型管理器] ✓ 垃圾回收已执行")


class DirectoryMonitor:
    """
    目录监控器
    用于监控指定目录下的新增音频文件，自动进行异常检测
    """
    
    def __init__(self, algorithm_manager: UnifiedAlgorithmManager, device: str = 'auto'):
        self.algorithm_manager = algorithm_manager
        self.device = device  # 运行设备
        self.preprocessor = None
        self.monitor_thread = None
        self.is_monitoring = False
        self.monitor_dir = None
        self.interval = 30  # 默认监控间隔30秒
        self.processed_files = set()  # 已处理的文件集合
        self.detection_results = deque(maxlen=1000)  # 保存最近的检测结果
        self.status_callback = None  # 状态回调函数
        self.detect_existing_on_start = False  # 启动时是否检测已有文件
        
    def set_preprocessor(self, ref_file: str, split_method: str = 'mfcc_dtw', 
                        shazam_threshold: int = 10, shazam_auto_match: bool = False,
                        max_workers: int = 1):
        """设置预处理器"""
        self.preprocessor = Preprocessor(
            ref_file=ref_file,
            split_method=split_method,
            shazam_threshold=shazam_threshold,
            shazam_auto_match=shazam_auto_match,
            max_workers=max_workers
        )
        
    def set_monitor_params(self, monitor_dir: str, interval: int = 5):
        """设置监控参数"""
        self.monitor_dir = monitor_dir
        self.interval = interval
        
    def update_interval(self, interval: int):
        """动态更新检测间隔"""
        self.interval = interval
        self._log(f"检测间隔已更新为 {interval} 秒")
        
    def update_algorithm(self, algorithm: str, config):
        """动态更新检测算法（会重新加载模型）"""
        print(f"\n{'='*60}")
        print(f"[在线模式] 🔄 动态切换算法请求")
        print(f"[在线模式] 目标算法: {algorithm}")
        print(f"[在线模式] 监控状态: {'运行中' if self.is_monitoring else '未运行'}")
        print(f"{'='*60}")
        
        if self.is_monitoring:
            self._log(f"正在切换算法到: {algorithm}...")
            try:
                # 释放旧模型（使用封装方法，会更新状态标志）
                if self.algorithm_manager.is_model_loaded():
                    print(f"[在线模式] 🧹 释放旧模型...")
                    self.algorithm_manager.unload_model()
                    print(f"[在线模式] ✓ 旧模型已释放")
                
                # 切换算法
                print(f"[在线模式] 🏗️ 调用 update_algorithm 切换算法...")
                self.algorithm_manager.update_algorithm(algorithm)
                
                print(f"[在线模式] 🏗️ 调用 load_model() 加载新模型...")
                self.algorithm_manager.load_model()  # 使用封装方法
                
                self._log(f"算法切换成功: {algorithm}")
                print(f"[在线模式] ✓ 算法切换完成")
                print(f"{'='*60}\n")
                return True
            except Exception as e:
                self._log(f"算法切换失败: {e}")
                print(f"[在线模式] ✗ 算法切换失败: {e}")
                import traceback
                traceback.print_exc()
                print(f"{'='*60}\n")
                return False
        print(f"[在线模式] ℹ 监控未运行，跳过算法切换")
        print(f"{'='*60}\n")
        return False
        
    def set_status_callback(self, callback):
        """设置状态回调函数"""
        self.status_callback = callback
        
    def _log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        if self.status_callback:
            self.status_callback(log_msg)
            
    def get_audio_files(self) -> List[str]:
        """获取监控目录下的所有音频文件"""
        if not self.monitor_dir or not os.path.exists(self.monitor_dir):
            return []
        
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
        audio_files = []
        
        try:
            for filename in os.listdir(self.monitor_dir):
                filepath = os.path.join(self.monitor_dir, filename)
                if os.path.isfile(filepath):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in audio_extensions:
                        audio_files.append(filepath)
        except Exception as e:
            self._log(f"扫描目录失败: {e}")
            
        return audio_files
        
    def detect_new_files(self) -> List[str]:
        """检测新文件"""
        current_files = set(self.get_audio_files())
        new_files = list(current_files - self.processed_files)
        return new_files
        
    def preprocess_files(self, audio_files: List[str]) -> Dict[str, List[str]]:
        """
        批量预处理音频文件
        返回: {audio_file: [image_paths]}
        """
        file_images_map = {}
        
        for audio_file in audio_files:
            try:
                filename = os.path.basename(audio_file)
                print(f"[预处理] 处理文件: {filename}")
                
                # 音频预处理
                picture_file_dict = self.preprocessor.process_audio([audio_file], save_dir="slice")
                
                if not picture_file_dict:
                    print(f"[预处理] ✗ 预处理失败: {filename}")
                    continue
                
                # 收集图像列表
                picture_file_list = []
                for k, v in picture_file_dict.items():
                    if isinstance(v, dict):
                        if v.get("dk"):
                            picture_file_list.append(v["dk"])
                        if v.get("qzgy"):
                            picture_file_list.append(v["qzgy"])
                
                if picture_file_list:
                    file_images_map[audio_file] = picture_file_list
                    print(f"[预处理] ✓ {filename}: 生成 {len(picture_file_list)} 个图像")
                else:
                    print(f"[预处理] ✗ {filename}: 未找到目标音频")
                    
            except Exception as e:
                print(f"[预处理] ✗ 处理失败 {audio_file}: {str(e)}")
                
        return file_images_map
    
    def process_batch_files(self, audio_files: List[str]) -> List[Dict]:
        """
        批量处理多个音频文件
        先统一预处理所有文件，再一次性调用模型检测
        """
        if not audio_files:
            return []
        
        print(f"\n[批量检测] >>> 开始处理 {len(audio_files)} 个文件")
        self._log(f"开始批量处理 {len(audio_files)} 个新文件...")
        
        # 1. 批量预处理所有文件
        print(f"[批量检测] 步骤1/2: 预处理 {len(audio_files)} 个音频文件...")
        start_time = time.time()
        file_images_map = self.preprocess_files(audio_files)
        preprocess_time = time.time() - start_time
        
        if not file_images_map:
            print("[批量检测] ✗ 所有文件预处理失败")
            self._log("预处理失败，没有有效的图像")
            return []
        
        # 统计有效文件和图像数量
        valid_files = list(file_images_map.keys())
        total_images = sum(len(images) for images in file_images_map.values())
        print(f"[批量检测] ✓ 预处理完成: {len(valid_files)}/{len(audio_files)} 个文件有效，共 {total_images} 个图像，耗时 {preprocess_time:.2f}s")
        self._log(f"预处理完成: {len(valid_files)} 个文件，{total_images} 个图像")
        
        # 2. 收集所有图像，一次性调用检测模型
        all_images = []
        file_image_ranges = {}  # 记录每个文件对应的图像范围
        current_idx = 0
        
        for audio_file, images in file_images_map.items():
            file_image_ranges[audio_file] = (current_idx, current_idx + len(images))
            all_images.extend(images)
            current_idx += len(images)
        
        # 3. 批量异常检测
        print(f"[批量检测] 步骤2/2: 执行批量异常检测，共 {len(all_images)} 个图像...")
        start_time = time.time()
        try:
            all_detection_results = self.algorithm_manager.detector.predict_batch(all_images)
            detect_time = time.time() - start_time
            print(f"[批量检测] ✓ 检测完成，返回 {len(all_detection_results)} 个结果，耗时 {detect_time:.2f}s")
        except Exception as e:
            print(f"[批量检测] ✗ 检测失败: {str(e)}")
            self._log(f"批量检测失败: {str(e)}")
            return []
        
        # 4. 整理每个文件的结果
        results = []
        print(f"\n[批量检测] >>> 整理检测结果:")
        
        for audio_file, (start_idx, end_idx) in file_image_ranges.items():
            filename = os.path.basename(audio_file)
            file_results = all_detection_results[start_idx:end_idx]
            file_images = all_images[start_idx:end_idx]
            
            # 找出该文件的最高异常分数
            max_score = 0
            is_anomaly = False
            heatmap_path = None
            
            for img_path, result in zip(file_images, file_results):
                if result.anomaly_score > max_score:
                    max_score = result.anomaly_score
                    is_anomaly = result.is_anomaly
                    heatmap_path = result.metadata.get('heatmap_path') if result.metadata else None
            
            status_str = '异常' if is_anomaly else '正常'
            
            result_dict = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': filename,
                'filepath': audio_file,
                'anomaly_score': max_score,
                'is_anomaly': is_anomaly,
                'status': status_str,
                'heatmap_path': heatmap_path,
                'processed_images': file_images
            }
            results.append(result_dict)
            
            print(f"[批量检测]   - {filename}: 分数={max_score:.4f}, 状态={status_str}")
        
        # 打印统计
        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        normal_count = len(results) - anomaly_count
        print(f"\n[批量检测] ✅ 批量处理完成: 总计 {len(results)} 个文件, 异常 {anomaly_count} 个, 正常 {normal_count} 个")
        self._log(f"批量检测完成: {len(results)} 个文件, 异常 {anomaly_count} 个")
        
        return results
    
    def process_single_file(self, audio_file: str) -> Optional[Dict]:
        """处理单个音频文件（兼容旧接口，内部调用批量处理）"""
        results = self.process_batch_files([audio_file])
        return results[0] if results else None
            
    def monitoring_loop(self):
        """监控主循环"""
        print("\n" + "=" * 60)
        print("[监控] 监控线程已启动")
        print(f"[监控] 监控目录: {self.monitor_dir}")
        print(f"[监控] 检测间隔: {self.interval}秒")
        print(f"[监控] 使用算法: {self.algorithm_manager.algorithm_chosen}")
        print(f"[监控] 运行设备: {self.device}")
        print("=" * 60)
        
        self._log("=" * 50)
        self._log("监控线程已启动")
        self._log(f"监控目录: {self.monitor_dir}")
        self._log(f"检测间隔: {self.interval}秒")
        self._log("=" * 50)
        
        # 加载模型（仅当模型未加载时）
        try:
            if self.algorithm_manager.is_model_loaded():
                print(f"\n{'='*60}")
                print(f"[监控] ℹ 模型已经加载，跳过重复加载")
                print(f"[监控] 算法: {self.algorithm_manager.algorithm_chosen}")
                print(f"[监控] 设备: {self.device}")
                print(f"{'='*60}\n")
                self._log("模型已加载，直接使用")
            else:
                print(f"\n{'='*60}")
                print(f"[监控] 🏗️ 正在加载模型...")
                print(f"[监控] 算法: {self.algorithm_manager.algorithm_chosen}")
                print(f"[监控] 设备: {self.device}")
                print(f"{'='*60}")
                self.algorithm_manager.load_model()
                print(f"[监控] ✓ 模型加载成功")
                print(f"[监控] 模型状态: 已加载并就绪")
                print(f"{'='*60}\n")
                self._log("模型加载成功")
        except Exception as e:
            print(f"[监控] ✗ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self._log(f"模型加载失败: {e}")
            self.is_monitoring = False
            return
            
        # 初始化时扫描已有文件
        print(f"[监控] 扫描监控目录: {self.monitor_dir}")
        existing_files = self.get_audio_files()
        print(f"[监控] 发现 {len(existing_files)} 个现有音频文件")
        
        # 如果设置了检测已有文件标志，使用批量处理处理已有文件
        if self.detect_existing_on_start and existing_files:
            print(f"[监控] 开始批量检测 {len(existing_files)} 个已有文件...")
            self._log(f"开始对 {len(existing_files)} 个已有文件进行批量检测...")
            
            # 使用新的批量处理方法
            batch_results = self.process_batch_files(existing_files)
            
            # 将所有结果添加到检测队列
            for result in batch_results:
                self.detection_results.append(result)
            
            # 标记所有文件为已处理
            for audio_file in existing_files:
                self.processed_files.add(audio_file)
            
            print(f"\n[监控] ✅ 已有文件检测完成: 处理 {len(existing_files)} 个，有效结果 {len(batch_results)} 个")
            self._log(f"✅ 已有文件检测完成，共处理 {len(existing_files)} 个，发现 {len(batch_results)} 个有效结果")
        else:
            # 否则只是记录已有文件，不进行检测
            self.processed_files = set(existing_files)
            print(f"[监控] 初始化完成，已记录 {len(existing_files)} 个现有文件（跳过检测）")
            self._log(f"初始化完成，已记录 {len(existing_files)} 个现有文件（跳过检测）")
        
        print(f"\n[监控] 进入实时监控循环，每 {self.interval} 秒扫描一次...")
        scan_count = 0
        last_cleanup_time = time.time()
        
        while self.is_monitoring:
            try:
                scan_count += 1
                print(f"\n[监控] 第 {scan_count} 次扫描目录...")
                
                # 检测新文件
                new_files = self.detect_new_files()
                
                if new_files:
                    print(f"[监控] >>> 发现 {len(new_files)} 个新文件:")
                    for i, f in enumerate(new_files, 1):
                        print(f"[监控]   {i}. {os.path.basename(f)}")
                    self._log(f"发现 {len(new_files)} 个新文件")
                    
                    # 使用批量处理处理所有新文件
                    print(f"[监控] 开始批量处理 {len(new_files)} 个新文件...")
                    batch_results = self.process_batch_files(new_files)
                    
                    # 将所有结果添加到检测队列
                    for result in batch_results:
                        self.detection_results.append(result)
                    
                    # 标记所有文件为已处理
                    for audio_file in new_files:
                        self.processed_files.add(audio_file)
                    
                    print(f"[监控] ✅ 批量处理完成: {len(batch_results)} 个有效结果")
                else:
                    print(f"[监控] 未发现新文件，等待 {self.interval} 秒后再次扫描...")
                        
                # 定期清理临时文件（每小时检查一次）
                current_time = time.time()
                if current_time - last_cleanup_time > 3600:  # 3600秒 = 1小时
                    self._cleanup_temp_files(max_age_hours=24)
                    last_cleanup_time = current_time
                
                # 等待下一次检测
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"[监控] ✗ 监控循环出错: {e}")
                import traceback
                traceback.print_exc()
                self._log(f"监控循环出错: {e}")
                time.sleep(self.interval)
        
        print("\n" + "=" * 60)
        print("[监控] 监控线程已停止")
        print("=" * 60)
        self._log("监控线程已停止")
        
    def start_monitoring(self) -> bool:
        """开始监控"""
        if self.is_monitoring:
            self._log("监控已在运行")
            return False
            
        if not self.monitor_dir:
            self._log("错误: 未设置监控目录")
            return False
            
        if not os.path.exists(self.monitor_dir):
            self._log(f"错误: 目录不存在 {self.monitor_dir}")
            return False
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        return True
        
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self._log("监控已停止")
        
    def get_results_df(self) -> pd.DataFrame:
        """获取检测结果DataFrame"""
        if not self.detection_results:
            return pd.DataFrame(columns=['时间', '文件名', '异常分数', '是否异常', '状态'])
            
        data = []
        for result in self.detection_results:
            data.append({
                '时间': result['timestamp'],
                '文件名': result['filename'],
                '异常分数': result['anomaly_score'],
                '是否异常': result['is_anomaly'],
                '状态': result['status']
            })
            
        df = pd.DataFrame(data)
        # 按时间倒序排列
        df = df.iloc[::-1].reset_index(drop=True)
        return df
        
    def get_anomaly_count(self) -> int:
        """获取异常文件数量"""
        return sum(1 for r in self.detection_results if r['is_anomaly'])
        
    def get_total_count(self) -> int:
        """获取总检测数量"""
        return len(self.detection_results)
        
    def clear_results(self):
        """清空结果"""
        self.detection_results.clear()
        self._log("结果已清空")
        
    def cleanup(self):
        """清理资源，防止内存泄漏"""
        print(f"\n{'='*60}")
        print(f"[Cleanup] 🧹 开始清理监控资源...")
        print(f"{'='*60}")
        
        # 1. 停止监控
        if self.is_monitoring:
            print(f"[Cleanup] ⏹️ 停止监控线程...")
            self.is_monitoring = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
                print(f"[Cleanup] ✓ 监控线程已停止")
        else:
            print(f"[Cleanup] ℹ 监控未运行，无需停止线程")
        
        # 2. 释放模型和显存
        if self.algorithm_manager and self.algorithm_manager.is_model_loaded():
            print(f"[Cleanup] 🏗️ 准备释放模型...")
            print(f"[Cleanup] 当前算法: {self.algorithm_manager.algorithm_chosen}")
            try:
                self.algorithm_manager.unload_model()  # 使用封装方法
                print(f"[Cleanup] ✓ 模型已释放")
            except Exception as e:
                print(f"[Cleanup] ⚠ 模型释放失败: {e}")
            
            # 强制清空CUDA缓存
            try:
                if torch.cuda.is_available():
                    print(f"[Cleanup] 🧹 清空 CUDA 缓存...")
                    for i in range(torch.cuda.device_count()):
                        allocated_before = torch.cuda.memory_allocated(i) / 1024**2
                        print(f"[Cleanup]   GPU {i} 清理前: {allocated_before:.1f}MB")
                    
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    for i in range(torch.cuda.device_count()):
                        allocated_after = torch.cuda.memory_allocated(i) / 1024**2
                        print(f"[Cleanup]   GPU {i} 清理后: {allocated_after:.1f}MB")
                    print(f"[Cleanup] ✓ CUDA缓存已清空")
            except Exception as e:
                print(f"[Cleanup] ⚠ CUDA缓存清空失败: {e}")
        else:
            print(f"[Cleanup] ℹ 无模型需要释放")
        
        # 3. 清理数据集合
        result_count = len(self.detection_results)
        self.detection_results.clear()
        self.processed_files.clear()
        print(f"[Cleanup] ✓ 已清空 {result_count} 条检测结果和已处理文件记录")
        
        # 4. 删除预处理器引用
        if self.preprocessor:
            del self.preprocessor
            self.preprocessor = None
            print(f"[Cleanup] ✓ 预处理器已释放")
        
        # 5. 强制垃圾回收
        gc.collect()
        print(f"[Cleanup] ✓ 垃圾回收已执行")
        print(f"[Cleanup] ✓ 资源清理完成")
        print(f"{'='*60}\n")
        
    def _cleanup_temp_files(self, max_age_hours=24):
        """
        清理临时文件（slice目录下的旧文件）
        max_age_hours: 超过此时间的文件将被删除
        """
        slice_dir = "slice"
        if not os.path.exists(slice_dir):
            return
            
        current_time = time.time()
        deleted_count = 0
        deleted_size = 0
        
        try:
            for filename in os.listdir(slice_dir):
                filepath = os.path.join(slice_dir, filename)
                if os.path.isfile(filepath):
                    # 检查文件修改时间
                    file_mtime = os.path.getmtime(filepath)
                    age_hours = (current_time - file_mtime) / 3600
                    
                    if age_hours > max_age_hours:
                        try:
                            file_size = os.path.getsize(filepath)
                            os.remove(filepath)
                            deleted_count += 1
                            deleted_size += file_size
                        except Exception as e:
                            print(f"[Cleanup] 删除临时文件失败 {filepath}: {e}")
            
            if deleted_count > 0:
                print(f"[Cleanup] 清理 {deleted_count} 个临时文件，释放 {deleted_size/1024/1024:.2f} MB")
                
        except Exception as e:
            print(f"[Cleanup] 临时文件清理失败: {e}")
    
    def cleanup_all_slice_files(self) -> tuple:
        """
        清理slice目录下的所有临时文件（无时间限制）
        返回: (删除文件数量, 释放空间字节数)
        """
        slice_dir = "slice"
        if not os.path.exists(slice_dir):
            return 0, 0
        
        deleted_count = 0
        deleted_size = 0
        
        try:
            for filename in os.listdir(slice_dir):
                filepath = os.path.join(slice_dir, filename)
                if os.path.isfile(filepath):
                    try:
                        file_size = os.path.getsize(filepath)
                        os.remove(filepath)
                        deleted_count += 1
                        deleted_size += file_size
                    except Exception as e:
                        print(f"[Cleanup] 删除临时文件失败 {filepath}: {e}")
            
            print(f"[Cleanup] 清理所有临时文件完成: {deleted_count} 个文件，释放 {deleted_size/1024/1024:.2f} MB")
            return deleted_count, deleted_size
                
        except Exception as e:
            print(f"[Cleanup] 临时文件清理失败: {e}")
            return deleted_count, deleted_size
    
    @staticmethod
    def cleanup_all_temp_files() -> dict:
        """
        清理所有临时文件（slice/、exports/ 和 visualize/ 目录）
        递归删除目录中的所有文件和子目录
        返回: {'slice': (file_count, dir_count, size), 'exports': (...), 'visualize': (...)}
        """
        result = {'slice': (0, 0, 0), 'exports': (0, 0, 0), 'visualize': (0, 0, 0)}
        
        def cleanup_directory_recursive(dir_path: str) -> tuple:
            """
            递归清理目录中的所有文件和子目录
            返回: (文件数量, 目录数量, 释放空间字节数)
            """
            file_count, dir_count, total_size = 0, 0, 0
            
            if not os.path.exists(dir_path):
                return file_count, dir_count, total_size
            
            try:
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    
                    if os.path.isfile(item_path):
                        # 删除文件
                        try:
                            file_size = os.path.getsize(item_path)
                            os.remove(item_path)
                            file_count += 1
                            total_size += file_size
                        except Exception as e:
                            print(f"[Cleanup] 删除文件失败 {item_path}: {e}")
                    
                    elif os.path.isdir(item_path):
                        # 递归清理子目录
                        sub_file_count, sub_dir_count, sub_size = cleanup_directory_recursive(item_path)
                        file_count += sub_file_count
                        dir_count += sub_dir_count
                        total_size += sub_size
                        
                        # 删除空目录
                        try:
                            os.rmdir(item_path)
                            dir_count += 1
                        except Exception as e:
                            print(f"[Cleanup] 删除目录失败 {item_path}: {e}")
                
            except Exception as e:
                print(f"[Cleanup] 清理目录失败 {dir_path}: {e}")
            
            return file_count, dir_count, total_size
        
        # 清理 slice 目录
        slice_dir = "slice"
        if os.path.exists(slice_dir):
            file_count, dir_count, size = cleanup_directory_recursive(slice_dir)
            result['slice'] = (file_count, dir_count, size)
            print(f"[Cleanup] slice目录: 清理 {file_count} 个文件, {dir_count} 个目录, 释放 {size/1024/1024:.2f} MB")
        
        # 清理 exports 目录
        exports_dir = "exports"
        if os.path.exists(exports_dir):
            file_count, dir_count, size = cleanup_directory_recursive(exports_dir)
            result['exports'] = (file_count, dir_count, size)
            print(f"[Cleanup] exports目录: 清理 {file_count} 个文件, {dir_count} 个目录, 释放 {size/1024/1024:.2f} MB")
        
        # 清理 visualize 目录
        visualize_dir = "visualize"
        if os.path.exists(visualize_dir):
            file_count, dir_count, size = cleanup_directory_recursive(visualize_dir)
            result['visualize'] = (file_count, dir_count, size)
            print(f"[Cleanup] visualize目录: 清理 {file_count} 个文件, {dir_count} 个目录, 释放 {size/1024/1024:.2f} MB")
        
        total_files = result['slice'][0] + result['exports'][0] + result['visualize'][0]
        total_dirs = result['slice'][1] + result['exports'][1] + result['visualize'][1]
        total_size = result['slice'][2] + result['exports'][2] + result['visualize'][2]
        print(f"[Cleanup] 总计: 清理 {total_files} 个文件, {total_dirs} 个目录, 释放 {total_size/1024/1024:.2f} MB")
        
        return result


def run_asd_btn_func(audio_files, algorithm_choice: str, device_choice: str, progress=gr.Progress()):
    """主处理函数 - 离线模式批量检测"""
    global model_manager
    
    print("\n" + "=" * 60)
    print("[离线模式] 开始批量异常检测")
    print(f"[离线模式] 选择设备: {device_choice}")
    print("=" * 60)
    
    # 加载配置（使用绝对路径）
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "config", "algorithms.yaml")
    config = ConfigManager(config_path)
    ref_file = config.config.get('preprocessing', {}).get('ref_file', 'ref/渡口片段10s.wav')
    # 如果ref_file是相对路径，转为绝对路径
    if not os.path.isabs(ref_file):
        ref_file = os.path.join(project_root, ref_file)
    
    # 获取预处理配置
    split_method = config.config.get('preprocessing', {}).get('split_method', 'mfcc_dtw')
    shazam_config = config.config.get('preprocessing', {}).get('shazam', {})
    shazam_threshold = shazam_config.get('threshold', 10)
    shazam_auto_match = shazam_config.get('auto_match', False)
    max_workers = shazam_config.get('max_workers', 1)
    
    print(f"[离线模式] 上传文件数量: {len(audio_files) if audio_files else 0}")
    print(f"[离线模式] 选择算法: {algorithm_choice}")
    print(f"[离线模式] 参考音频: {ref_file}")
    print(f"[离线模式] 切分方法: {split_method}")
    
    # 初始化组件
    p = Preprocessor(
        ref_file=ref_file,
        split_method=split_method,
        shazam_threshold=shazam_threshold,
        shazam_auto_match=shazam_auto_match,
        max_workers=max_workers
    )
    lm = LogManager()
    # 使用全局模型实例 model_manager，与在线模式共享
    
    # 资源释放标志（仅用于清理，不销毁模型）
    model_loaded = False
    
    try:
        if not audio_files:
            print("[离线模式] ✗ 错误: 未上传音频文件")
            yield from lm.update_log_and_action("请上传至少一个音频文件", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()
        
        # 加载算法（使用全局实例 model_manager）
        print(f"\n{'='*60}")
        print(f"[离线模式] 步骤1/4: 加载算法模型...")
        print(f"[离线模式] 算法: {algorithm_choice}")
        print(f"[离线模式] 设备: {device_choice}")
        print(f"{'='*60}")
        try:
            model_manager.update_algorithm(algorithm_choice, device=device_choice)
            print(f"[离线模式] 🏗️ 调用 load_model() 加载模型权重...")
            model_manager.load_model()  # 使用封装方法，会设置 _model_loaded 标志
            model_loaded = True
            print(f"[离线模式] ✓ 算法模型加载成功: {algorithm_choice}")
            print(f"[离线模式] 模型状态: 已加载 (与在线模式共享实例)")
            print(f"{'='*60}")
        except Exception as e:
            print(f"[离线模式] ✗ 算法加载失败: {e}")
            import traceback
            traceback.print_exc()
            yield from lm.update_log_and_action(f"算法加载失败: {str(e)}", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()
        
        # [1] 音频预处理
        print(f"\n[离线模式] 步骤2/4: 音频预处理...")
        yield from lm.update_log("执行音频预处理...")
        try:
            picture_file_dict = p.process_audio(audio_files, save_dir="slice")
            print(f"[离线模式] ✓ 预处理完成，生成 {len(picture_file_dict)} 个音频片段")
        except Exception as e:
            print(f"[离线模式] ✗ 音频预处理失败: {e}")
            import traceback
            traceback.print_exc()
            yield from lm.update_log_and_action(f"音频预处理失败: {str(e)}", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()
        
        if picture_file_dict is None or not picture_file_dict:
            print("[离线模式] ✗ 预处理返回空结果")
            yield from lm.update_log_and_action("音频预处理返回空结果，请检查音频文件格式", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()
        
        yield from lm.update_log("完成目标音频搜索和切分！！")
        
        # [2] 准备图像列表
        print(f"\n[离线模式] 步骤3/4: 查找目标音频片段...")
        picture_file_list = []
        for k, v in picture_file_dict.items():
            if isinstance(v, dict):
                if v.get("dk"):
                    picture_file_list.append(v["dk"])
                if v.get("qzgy"):
                    picture_file_list.append(v["qzgy"])
        
        print(f"[离线模式] ✓ 找到 {len(picture_file_list)} 个目标图像")
        
        if not picture_file_list:
            print("[离线模式] ✗ 未找到目标音频片段")
            yield from lm.update_log_and_action("上传的素材中没有找到目标音频，请检查!", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()
        
        # [3] 执行异常检测
        print(f"\n[离线模式] 步骤4/4: 执行异常检测...")
        print(f"[离线模式] 使用算法: {model_manager.algorithm_chosen}")
        print(f"[离线模式] 待检测图像数量: {len(picture_file_list)}")
        yield from lm.update_log(f"使用算法: {model_manager.algorithm_chosen}")
        yield from lm.update_log("执行异常预测...")
        
        try:
            # 使用统一接口批量推理
            detection_results = model_manager.detector.predict_batch(picture_file_list)
            print(f"[离线模式] ✓ 检测完成，返回 {len(detection_results)} 个结果")
            
            # 构建结果字典和热力图路径字典
            pred_res_dict = {}
            heatmap_paths_dict = {}
            print(f"\n[离线模式] >>> 详细检测结果:")
            for idx, (img_path, result) in enumerate(zip(picture_file_list, detection_results), 1):
                filename = os.path.basename(img_path)
                pred_res_dict[filename] = (result.anomaly_score, 1 if result.is_anomaly else 0)
                # 从metadata中获取热力图路径
                heatmap_path = result.metadata.get('heatmap_path') if result.metadata else None
                if heatmap_path:
                    heatmap_paths_dict[filename] = heatmap_path
                
                status = "异常" if result.is_anomaly else "正常"
                print(f"[离线模式]   {idx}. {filename}")
                print(f"[离线模式]      - 异常分数: {result.anomaly_score:.4f}")
                print(f"[离线模式]      - 检测状态: {status}")
                if heatmap_path:
                    print(f"[离线模式]      - 热力图: {heatmap_path}")
            
            print(f"\n[离线模式] 统计: 异常 {sum(1 for r in detection_results if r.is_anomaly)} 个, 正常 {sum(1 for r in detection_results if not r.is_anomaly)} 个")
            yield from lm.update_log("检测完成!")
        except Exception as e:
            yield from lm.update_log_and_action(f"检测失败: {str(e)}", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()
        
        # [4] 输出结果
        print(f"\n[离线模式] 整理输出结果...")
        yield from lm.update_log("整理输出结果中...")
        
        results = []
        for k, v in pred_res_dict.items():
            results.append({"filename": k, "anomaly_score": v[0], "is_anomaly": v[1]})
        
        df_results = pd.DataFrame(results)
        df_results['状态'] = df_results['is_anomaly'].apply(lambda x: '异常' if x == 1 else '正常')
        df_results = df_results[['filename', 'anomaly_score', 'is_anomaly', '状态']]
        df_results.columns = ['文件名', '异常分数', '是否异常', '状态']
        
        print(f"[离线模式] 生成结果表格: {len(results)} 行数据")
        
        # 使用可控的 exports 目录存放导出文件
        exports_dir = "exports"
        os.makedirs(exports_dir, exist_ok=True)
        
        excel_path = Export.create_excel_report(results, exports_dir)
        print(f"[离线模式] ✓ Excel报告生成: {excel_path}")
        
        zip_path = os.path.join(exports_dir, f'asd_results_{model_manager.algorithm_chosen}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')
        images_for_zip = picture_file_list
        zip_path = Export.create_zip_with_results(zip_path, excel_path, images_for_zip)
        print(f"[离线模式] ✓ ZIP包生成: {zip_path}")
        
        # 为Gallery准备带caption的数据 (热力图路径, caption)
        gallery_data = []
        for img_path in picture_file_list:
            filename = os.path.basename(img_path)
            heatmap_path = heatmap_paths_dict.get(filename)
            
            if heatmap_path and os.path.exists(heatmap_path):
                # 使用热力图
                caption = os.path.splitext(filename)[0]
                gallery_data.append((heatmap_path, caption))
                print(f"[离线模式]      ✓ 使用热力图: {heatmap_path}")
            elif os.path.exists(img_path):
                # 回退到原图
                caption = os.path.splitext(filename)[0]
                gallery_data.append((img_path, caption))
                print(f"[离线模式]      ℹ 无热力图，使用原图: {img_path}")
            else:
                print(f"[离线模式]      ✗ 图像文件不存在: {img_path}")

        print(f"\n[离线模式] 输出统计:")
        print(f"[离线模式]   - Gallery图像: {len(gallery_data)} 个")
        print(f"[离线模式]   - 结果表格: {len(df_results)} 行")
        print(f"[离线模式]   - 总文件数: {len(results)} 个")

        # 确保gallery_data不为空，如果为空则提供一个提示
        if not gallery_data:
            print("[离线模式] ⚠ 警告: gallery_data为空，没有图像可显示")

        print(f"\n[离线模式] >>> 处理完成! 共处理 {len(results)} 个文件")
        print("=" * 60)
        
        lm.generate_log(f"处理完成! 共处理 {len(results)} 个文件，请查看下方结果!")

        # 返回结果，并同步算法选择到在线模式
        yield lm.log_messages, gr.update(visible=True), df_results, gallery_data, zip_path, gr.update(interactive=True), gr.update(value=algorithm_choice)
        
    finally:
        # 模型由全局实例管理，离线模式检测完成后不释放模型
        # 这样可以与在线模式共享模型，避免重复加载
        if model_loaded:
            print(f"\n{'='*60}")
            print(f"[离线模式] 检测流程完成")
            print(f"[离线模式] 模型状态: 保持加载（与在线模式共享）")
            print(f"[离线模式] 当前算法: {model_manager.algorithm_chosen}")
            print(f"[离线模式] 当前设备: {model_manager.device}")
            print(f"{'='*60}")


# 加载全局配置（使用绝对路径）
_project_root = os.path.dirname(os.path.abspath(__file__))
_config_path = os.path.join(_project_root, "config", "algorithms.yaml")
config = ConfigManager(_config_path)

# 设置Dinomaly所需的环境变量
_dinomaly_config_path = os.path.join(_project_root, "config", "asd_gui_config.yaml")
if os.path.exists(_dinomaly_config_path):
    import yaml
    with open(_dinomaly_config_path, 'r', encoding='utf-8') as f:
        dinomaly_config = yaml.safe_load(f) or {}
    env_config = dinomaly_config.get('environments', {})
    for key, value in env_config.items():
        if value:
            os.environ[key] = str(value)
            print(f"[DEBUG] Set environment variable: {key}={value}")

model_manager = UnifiedAlgorithmManager()

# 页面加载计数器和最后活跃时间（用于检测用户是否已离开）
_session_count = 0
_last_active_time = time.time()
_session_lock = threading.Lock()

# 监控日志使用deque限制大小，防止内存无限增长
monitor_logs = deque(maxlen=100)

def on_page_load():
    """页面加载时调用"""
    global _session_count, _last_active_time
    with _session_lock:
        _session_count += 1
        _last_active_time = time.time()
        print(f"[Session] 页面加载，当前会话数: {_session_count}")
    # 不返回任何值，避免Gradio警告

def on_page_unload():
    """页面卸载/刷新时调用"""
    global _session_count, _last_active_time
    with _session_lock:
        _session_count = max(0, _session_count - 1)
        _last_active_time = time.time()
        print(f"\n[Session] 页面离开，当前会话数: {_session_count}")
        
        # 如果没有会话了，彻底清理资源
        if _session_count == 0 and monitor:
            print(f"[Session] 所有会话已离开，执行完整资源清理...")
            monitor.cleanup()  # 使用新的cleanup方法
        else:
            print(f"[Session] 仍有 {_session_count} 个活跃会话，保持资源加载")
    # 不返回任何值

# Gradio界面定义
# JavaScript 用于自动滚动监控日志到底部
js_autoscroll = """
<script>
(function() {
    function scrollMonitorLogToBottom() {
        const logBox = document.querySelector('#monitor_log_box textarea, #monitor_log_box input');
        if (logBox) {
            logBox.scrollTop = logBox.scrollHeight;
        }
    }

    // 使用 MutationObserver 监听日志内容变化
    let observer = null;

    function initObserver() {
        const logBox = document.querySelector('#monitor_log_box textarea, #monitor_log_box input');
        if (logBox && !observer) {
            observer = new MutationObserver(function(mutations) {
                scrollMonitorLogToBottom();
            });
            observer.observe(logBox.parentElement || logBox, { childList: true, subtree: true, characterData: true });
        }
    }

    // 页面加载完成后开始监听
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initObserver);
    } else {
        initObserver();
    }

    // 定期检查元素是否存在（处理Gradio动态加载的情况）
    setInterval(function() {
        if (!observer) {
            initObserver();
        }
        scrollMonitorLogToBottom();
    }, 1000);
})();
</script>
"""

with gr.Blocks(title="音频异常检测工具") as demo:
    gr.Markdown("# 🎵 音频异常检测工具")
    gr.Markdown("支持离线手动检测和在线实时监测两种模式")

    gr.Markdown("---")

    # ==================== 离线模式 Tab（默认）====================
    with gr.Tab("💻 离线模式", id=0) as offline_tab:
        gr.Markdown("## 📤 手动上传音频检测")
        gr.Markdown("上传WAV格式音频文件进行批量异常检测")

        # 参考音频区域（放在离线模式中）
        with gr.Accordion("📁 参考音频文件（点击展开）", open=False):
            gr.Markdown("点击下方链接下载示例音频文件用于测试（渡口+青藏高原片段）")
            gr.Markdown("💡 目前仅适用于使用该标准音频得到的测试音频，其余功能请静候开发!")

            example_audio_file = config.config.get('preprocessing', {}).get('ref_file', 'ref/渡口片段10s.wav')
            if not os.path.isabs(example_audio_file):
                example_audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), example_audio_file)
            gr.Audio(
                label="示例音频试听",
                value=example_audio_file if os.path.exists(example_audio_file) else None,
                type="filepath",
                elem_classes=["audio-player"]
            )

        with gr.Row():
            with gr.Column():
                audio_inputs = gr.Files(
                    file_count="multiple",
                    file_types=[".wav"],
                    label="📤 上传音频文件"
                )

                algorithm_dropdown = gr.Dropdown(
                    choices=model_manager.available_algorithms,
                    value=model_manager.available_algorithms[0] if model_manager.available_algorithms else None,
                    label="🔧 选择异常检测算法"
                )

                # 设备选择 - 动态生成实际可用的设备列表
                available_devices = get_available_devices()
                device_dropdown = gr.Dropdown(
                    choices=[d[0] for d in available_devices],  # 只使用value值
                    value="auto",
                    label="🖥️ 选择运行设备",
                    info=", ".join([f"{d[0]}={d[1]}" for d in available_devices]),
                    allow_custom_value=True
                )

                run_button = gr.Button("🚀 开始异常检测", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(label="📊 处理状态", max_lines=9, interactive=True, autoscroll=True)
                download_output = gr.File(label="📥 下载结果文件", file_count="single")

        # 检测结果展示区域 - 初始隐藏
        results_section = gr.Column(visible=False)
        with results_section:
            gr.Markdown("---")
            gr.Markdown("## 📊 检测结果展示")
            gr.Markdown("💡 提示: 点击表格中的文件名可在下方热力图中定位")

            with gr.Row():
                # 结果表格
                results_table = gr.DataFrame(
                    label="检测结果明细 (点击文件名定位热力图)",
                    headers=["文件名", "异常分数", "是否异常", "状态"],
                    interactive=False,
                    wrap=True
                )

            with gr.Row():
                # 热力图画廊 - 使用caption显示文件名
                heatmap_gallery = gr.Gallery(
                    label="全部异常热力图 (点击表格行可定位到对应热力图)",
                    show_label=True,
                    elem_id="heatmap_gallery",
                    columns=4,
                    rows=2,
                    height="auto",
                    object_fit="contain",
                    preview=True
                )

        gr.Markdown("---")
        gr.Markdown("### 使用说明")
        gr.Markdown("""
        1. 点击"上传音频文件"按钮选择一个或多个WAV格式的音频文件
        2. 从下拉菜单中选择要使用的异常检测算法
        3. 点击"开始异常检测"按钮开始处理
        4. 处理完成后会显示状态信息，检测结果表格和热力图会在下方展示
        5. 点击表格中的文件名可在下方热力图画廊中定位到对应图像
        6. 可下载ZIP文件包含Excel报告和所有热力图图像
        """)

        # 绑定按钮事件（事件绑定移到文件末尾，确保所有组件已定义）
        # run_button.click 将在在线模式定义后统一绑定
        
        # 算法选择同步：离线模式改变时更新在线模式
        # 注意：algorithm_dropdown.change 将在在线模式定义后统一绑定

        def on_select_row(evt: gr.SelectData, gallery_data):
            """
            处理表格行选择事件
            evt.index: 选中的行索引
            evt.row_value: 选中的行数据
            返回: 将Gallery的selected_index设置为对应位置
            """
            if evt.index is None or not gallery_data:
                return gr.update()

            # 获取选中的文件名
            selected_filename = evt.row_value[0] if isinstance(evt.row_value, (list, tuple)) else None
            if not selected_filename:
                return gr.update()

            # 在gallery_data中查找匹配的热力图索引
            # gallery_data格式: [(img_path, caption), ...]
            for idx, (img_path, caption) in enumerate(gallery_data):
                # 检查caption或文件名是否匹配
                if caption == selected_filename or selected_filename in caption or caption in selected_filename:
                    # 返回Gallery更新，设置selected_index
                    return gr.update(selected_index=idx)

            return gr.update()

        # 绑定表格选择事件
        results_table.select(
            fn=on_select_row,
            inputs=[heatmap_gallery],
            outputs=[heatmap_gallery]
        )

    # ==================== 在线模式 Tab（实时监控）====================
    with gr.Tab("📡 在线模式", id=1) as online_tab:
        gr.Markdown("## 📡 目录实时监控")
        gr.Markdown("监控指定目录下的新增音频文件，自动进行异常检测")
        
        # 创建监控器实例（全局）
        monitor = DirectoryMonitor(model_manager)
        # monitor_logs 已在全局定义为 deque(maxlen=100)
        
        def update_monitor_status(message: str):
            """更新监控状态日志 - message 已经包含时间戳"""
            # monitor_logs 是 deque(maxlen=100)，自动限制大小
            monitor_logs.append(message)  # message 已由 _log() 添加时间戳
            return "\n".join(monitor_logs)
        
        monitor.set_status_callback(update_monitor_status)
        
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=350):
                # 监控设置
                monitor_dir_input = gr.Textbox(
                    label="📁 监控目录路径",
                    placeholder="输入要监控的目录绝对路径",
                    value=""
                )
                
                monitor_interval = gr.Slider(
                    label="⏱️ 检测间隔（秒）",
                    minimum=5,
                    maximum=60,
                    value=30,
                    step=5
                )
                
                monitor_algorithm = gr.Dropdown(
                    choices=model_manager.available_algorithms,
                    value=model_manager.available_algorithms[0] if model_manager.available_algorithms else None,
                    label="🔧 检测算法"
                )
                
                # 在线模式设备选择 - 动态生成实际可用的设备列表
                monitor_device = gr.Dropdown(
                    choices=[d[0] for d in available_devices],
                    value="auto",
                    label="🖥️ 运行设备",
                    info=", ".join([f"{d[0]}={d[1]}" for d in available_devices]),
                    allow_custom_value=True
                )
                
                with gr.Row():
                    start_monitor_btn = gr.Button("▶️ 开始监控", variant="primary")
                    stop_monitor_btn = gr.Button("⏹️ 停止监控", variant="secondary")
                    cleanup_temp_btn = gr.Button("🧹 清理临时文件", variant="secondary")
                    export_zip_btn = gr.Button("📦 导出结果", variant="secondary")
                
                # 隐藏的 HTML 组件，用于触发自动下载
                auto_download_html = gr.HTML(value="")
                
                # 统计信息
                monitor_stats = gr.Textbox(
                    label="📊 统计信息",
                    value="总检测: 0 | 异常: 0 | 正常: 0",
                    interactive=False
                )
                
                # 内存使用情况
                memory_stats = gr.Textbox(
                    label="🧠 内存/显存使用",
                    value="CPU内存: -- | GPU显存: --",
                    interactive=False
                )

            with gr.Column(scale=3, min_width=450):
                # 监控日志
                monitor_log_box = gr.Textbox(
                    label="📝 监控日志",
                    lines=8,
                    max_lines=12,
                    interactive=False,
                    autoscroll=True,
                    elem_id="monitor_log_box"
                )
                
                # 实时结果表格
                monitor_results_table = gr.DataFrame(
                    label="实时检测结果（最近100条）- 点击行跳转到对应热力图",
                    headers=["时间", "文件名", "异常分数", "是否异常", "状态"],
                    interactive=False,
                    wrap=True
                )

        # 确认对话框（初始隐藏）- 用于处理路径下已有音频文件的情况
        confirm_dialog = gr.Row(visible=False)
        with confirm_dialog:
            with gr.Column():
                gr.Markdown("---")
                gr.Markdown("⚠️ **检测到目录下已有音频文件，请选择处理方式：**")
                confirm_count = gr.Markdown("音频文件数量: 0")
                with gr.Row():
                    detect_existing_btn = gr.Button("🔍 检测已有文件", variant="primary")
                    skip_existing_btn = gr.Button("⏭️ 跳过已有文件", variant="secondary")

        # 最近异常文件展示
        with gr.Row():
            recent_anomaly_gallery = gr.Gallery(
                label="最近异常文件热力图",
                show_label=True,
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                interactive=False
            )
        
        # 存储临时参数（用于确认对话框）
        pending_monitor_params = {}
        
        def get_audio_files_in_dir(dir_path):
            """获取目录下的音频文件列表"""
            if not dir_path or not os.path.exists(dir_path):
                return []
            audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
            audio_files = []
            try:
                for filename in os.listdir(dir_path):
                    filepath = os.path.join(dir_path, filename)
                    if os.path.isfile(filepath):
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in audio_extensions:
                            audio_files.append(filepath)
            except Exception as e:
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 扫描目录失败: {e}")
            return audio_files
        
        def start_monitoring_fn(dir_path, interval, algorithm, device):
            """开始监控 - 先检查是否有已有文件"""
            if not dir_path or not os.path.exists(dir_path):
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 错误: 目录不存在")
                return "\n".join(list(monitor_logs)), gr.update(), gr.update(), gr.update(visible=False), gr.update(), gr.update()
            
            # 检查是否已在监控中
            if monitor.is_monitoring:
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 监控已在运行中")
                return "\n".join(list(monitor_logs)), gr.update(), gr.update(), gr.update(visible=False), gr.update(), gr.update()
            
            # 扫描目录下的音频文件
            existing_files = get_audio_files_in_dir(dir_path)
            
            if existing_files:
                # 有已有文件，显示确认对话框
                global pending_monitor_params
                pending_monitor_params = {
                    'dir_path': dir_path,
                    'interval': interval,
                    'algorithm': algorithm,
                    'device': device,
                    'existing_files': existing_files
                }
                count_msg = f"发现 {len(existing_files)} 个音频文件"
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {count_msg}，等待用户选择...")
                # 注意：confirm_count是Markdown，通过gr.update(value=...)更新
                return "\n".join(list(monitor_logs)), gr.update(), gr.update(), gr.update(visible=True), gr.update(value=f"**音频文件数量: {len(existing_files)} 个**"), gr.update(value=algorithm)
            
            # 没有已有文件，直接开始监控
            result = _do_start_monitoring(dir_path, interval, algorithm, device, detect_existing=False)
            # _do_start_monitoring 返回6个值（包括 algorithm_dropdown）
            return result[0], result[1], result[2], result[3], result[4], result[5]
        
        def _do_start_monitoring(dir_path, interval, algorithm, device='auto', detect_existing=False):
            """执行实际开始监控的操作"""
            # 更新监控器的设备设置
            monitor.device = device
            
            # 设置预处理器
            project_root = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(project_root, "config", "algorithms.yaml")
            config = ConfigManager(config_path)
            ref_file = config.config.get('preprocessing', {}).get('ref_file', 'ref/渡口片段10s.wav')
            if not os.path.isabs(ref_file):
                ref_file = os.path.join(project_root, ref_file)
            
            # 获取预处理配置
            split_method = config.config.get('preprocessing', {}).get('split_method', 'mfcc_dtw')
            shazam_config = config.config.get('preprocessing', {}).get('shazam', {})
            shazam_threshold = shazam_config.get('threshold', 10)
            shazam_auto_match = shazam_config.get('auto_match', False)
            max_workers = shazam_config.get('max_workers', 1)
            
            monitor.set_preprocessor(
                ref_file=ref_file,
                split_method=split_method,
                shazam_threshold=shazam_threshold,
                shazam_auto_match=shazam_auto_match,
                max_workers=max_workers
            )
            monitor.set_monitor_params(dir_path, interval)
            
            # 切换算法并指定设备
            try:
                print(f"\n[在线模式] 🔄 监控启动前切换算法...")
                model_manager.update_algorithm(algorithm, device=device)
                print(f"[在线模式] ✓ 算法切换完成，准备加载模型...")
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 使用设备: {device}")
            except Exception as e:
                print(f"[在线模式] ✗ 算法切换失败: {e}")
                import traceback
                traceback.print_exc()
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 算法切换失败: {e}")
                return "\n".join(list(monitor_logs)), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=False), gr.update()

            # 设置是否检测已有文件的标志
            monitor.detect_existing_on_start = detect_existing
            
            # 如果有已有文件且选择跳过，则添加到已处理列表
            if not detect_existing and pending_monitor_params.get('existing_files'):
                for f in pending_monitor_params['existing_files']:
                    monitor.processed_files.add(f)
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 已跳过 {len(pending_monitor_params['existing_files'])} 个已有文件")
            
            # 开始监控
            success = monitor.start_monitoring()

            if success:
                return "\n".join(list(monitor_logs)), gr.update(interactive=False), gr.update(interactive=True), gr.update(visible=False), gr.update(interactive=False), gr.update(value=algorithm)
            else:
                return "\n".join(list(monitor_logs)), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=True), gr.update()
        
        def confirm_detect_existing_fn():
            """确认检测已有文件"""
            global pending_monitor_params
            
            if not pending_monitor_params:
                return "\n".join(list(monitor_logs)), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=True), gr.update()
            
            monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 选择：检测已有文件")
            
            params = pending_monitor_params
            pending_monitor_params = {}  # 清空
            
            result = _do_start_monitoring(
                params['dir_path'], 
                params['interval'], 
                params['algorithm'],
                params.get('device', 'auto'),
                detect_existing=True
            )
            # 添加 algorithm_dropdown 同步
            return result[0], result[1], result[2], result[3], result[4], gr.update(value=params['algorithm'])
        
        def confirm_skip_existing_fn():
            """确认跳过已有文件"""
            global pending_monitor_params
            
            if not pending_monitor_params:
                return "\n".join(list(monitor_logs)), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=True), gr.update()
            
            monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 选择：跳过已有文件")
            
            params = pending_monitor_params
            pending_monitor_params = {}  # 清空
            
            result = _do_start_monitoring(
                params['dir_path'], 
                params['interval'], 
                params['algorithm'],
                params.get('device', 'auto'),
                detect_existing=False
            )
            # 添加 algorithm_dropdown 同步
            return result[0], result[1], result[2], result[3], result[4], gr.update(value=params['algorithm'])
        
        def stop_monitoring_fn():
            """停止监控并释放资源"""
            print(f"\n{'='*60}")
            print(f"[在线模式] ⏹️ 用户请求停止监控")
            print(f"{'='*60}")
            
            monitor.stop_monitoring()
            
            # 停止后释放模型资源（共享模型，也需要释放）
            if monitor.algorithm_manager and monitor.algorithm_manager.is_model_loaded():
                try:
                    print(f"[在线模式] 🧹 释放模型资源...")
                    print(f"[在线模式] 当前算法: {monitor.algorithm_manager.algorithm_chosen}")
                    monitor.algorithm_manager.unload_model()  # 使用封装方法
                    torch.cuda.empty_cache()
                    print(f"[在线模式] ✓ 模型资源已释放")
                    monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 模型资源已释放")
                except Exception as e:
                    print(f"[在线模式] ⚠ 资源释放失败: {e}")
                    monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 资源释放失败: {e}")
            else:
                print(f"[在线模式] ℹ 模型未加载，无需释放")
            
            print(f"{'='*60}\n")
            # 返回5个值（包括 algorithm_dropdown 保持当前值）
            return "\n".join(list(monitor_logs)), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True), gr.update()
        
        def cleanup_temp_files_fn():
            """手动清理所有临时文件（slice/、exports/ 和 visualize/ 目录）"""
            result = DirectoryMonitor.cleanup_all_temp_files()
            
            slice_files, slice_dirs, slice_size = result['slice']
            exports_files, exports_dirs, exports_size = result['exports']
            visualize_files, visualize_dirs, visualize_size = result['visualize']
            total_files = slice_files + exports_files + visualize_files
            total_dirs = slice_dirs + exports_dirs + visualize_dirs
            total_size = slice_size + exports_size + visualize_size
            
            if total_files > 0 or total_dirs > 0:
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 临时文件清理完成: {total_files}个文件, {total_dirs}个目录, 共释放 {total_size/1024/1024:.1f} MB")
            else:
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ 没有需要清理的临时文件")
            return "\n".join(list(monitor_logs))
        
        def export_monitor_results_fn():
            """导出监控结果为Excel"""
            print(f"\n{'='*60}")
            print(f"[在线模式] 📊 导出Excel报告...")
            
            results = list(monitor.detection_results)
            if not results:
                print(f"[在线模式] ⚠ 暂无检测结果可导出")
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 导出失败: 暂无检测结果")
                return "\n".join(list(monitor_logs)), None
            
            try:
                # 使用可控的 exports 目录存放导出文件
                exports_dir = "exports"
                os.makedirs(exports_dir, exist_ok=True)
                
                # 准备导出数据
                export_data = []
                for r in results:
                    export_data.append({
                        'filename': r['filename'],
                        'anomaly_score': r['anomaly_score'],
                        'is_anomaly': r['is_anomaly'],
                        'timestamp': r['timestamp'],
                        'filepath': r['filepath']
                    })
                
                # 创建Excel报告
                excel_path = Export.create_excel_report(export_data, exports_dir)
                print(f"[在线模式] ✓ Excel报告生成: {excel_path}")
                print(f"[在线模式]   记录数: {len(export_data)}")
                
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Excel报告导出成功: {len(export_data)}条记录")
                print(f"{'='*60}\n")
                
                return "\n".join(list(monitor_logs)), excel_path
            except Exception as e:
                print(f"[在线模式] ✗ Excel导出失败: {e}")
                import traceback
                traceback.print_exc()
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 导出失败: {str(e)}")
                return "\n".join(list(monitor_logs)), None
        
        def export_monitor_zip_fn():
            """打包下载Excel和所有热力图"""
            print(f"\n{'='*60}")
            print(f"[在线模式] 📦 打包下载全部结果...")
            
            results = list(monitor.detection_results)
            if not results:
                print(f"[在线模式] ⚠ 暂无检测结果可打包")
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 打包失败: 暂无检测结果")
                return "\n".join(list(monitor_logs)), None
            
            try:
                # 使用可控的 exports 目录存放导出文件
                exports_dir = "exports"
                os.makedirs(exports_dir, exist_ok=True)
                
                # 准备导出数据
                export_data = []
                heatmap_paths = []
                for r in results:
                    export_data.append({
                        'filename': r['filename'],
                        'anomaly_score': r['anomaly_score'],
                        'is_anomaly': r['is_anomaly'],
                        'timestamp': r['timestamp'],
                        'filepath': r['filepath']
                    })
                    # 收集热力图路径
                    if r.get('heatmap_path') and os.path.exists(r['heatmap_path']):
                        heatmap_paths.append(r['heatmap_path'])
                    # 收集所有处理过的图像
                    if r.get('processed_images'):
                        for img_path in r['processed_images']:
                            if os.path.exists(img_path) and img_path not in heatmap_paths:
                                heatmap_paths.append(img_path)
                
                # 创建Excel报告
                excel_path = Export.create_excel_report(export_data, exports_dir)
                print(f"[在线模式] ✓ Excel报告生成: {excel_path}")
                
                # 创建ZIP包
                zip_filename = f"monitor_results_{model_manager.algorithm_chosen}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                zip_path = os.path.join(exports_dir, zip_filename)
                Export.create_zip_with_results(zip_path, excel_path, heatmap_paths)
                print(f"[在线模式] ✓ ZIP包生成: {zip_path}")
                print(f"[在线模式]   记录数: {len(export_data)}, 图像数: {len(heatmap_paths)}")
                
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 打包下载成功: {len(export_data)}条记录, {len(heatmap_paths)}张图像")
                print(f"{'='*60}\n")
                
                # 生成自动下载的 HTML 脚本
                zip_filename = os.path.basename(zip_path)
                download_html = f"""<script>
setTimeout(function() {{
    var link = document.createElement('a');
    link.href = '/file={zip_path}';
    link.download = '{zip_filename}';
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    console.log('[AutoDownload] Triggered download for: {zip_filename}');
}}, 500);
</script>"""

                return "\n".join(list(monitor_logs)), download_html
            except Exception as e:
                print(f"[在线模式] ✗ 打包失败: {e}")
                import traceback
                traceback.print_exc()
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 打包失败: {str(e)}")
                return "\n".join(list(monitor_logs)), ""
        
        def get_memory_stats():
            """获取内存和显存使用情况"""
            # 获取当前设备信息
            if monitor.is_monitoring:
                current_device = monitor.device
            else:
                current_device = "未运行"
            
            device_info = f"设备: {current_device}"
            
            # 解析设备ID（如 "cuda:1" -> 1）
            gpu_id = 0  # 默认GPU
            if isinstance(current_device, str) and current_device.startswith("cuda:"):
                try:
                    gpu_id = int(current_device.split(":")[1])
                except:
                    gpu_id = 0
            
            if not PSUTIL_AVAILABLE:
                # GPU显存（不依赖psutil）
                gpu_mem_str = "未使用"
                if torch.cuda.is_available():
                    try:
                        gpu_mem = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
                        gpu_mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 / 1024
                        gpu_percent = (gpu_mem / gpu_mem_total) * 100
                        gpu_mem_str = f"{gpu_mem:.0f}MB/{gpu_mem_total:.0f}MB ({gpu_percent:.1f}%)"
                    except:
                        gpu_mem_str = "获取失败"
                return f"{device_info} | CPU: 需安装psutil | GPU: {gpu_mem_str}"
            
            # CPU内存
            mem = psutil.virtual_memory()
            cpu_mem_str = f"{mem.used/1024/1024/1024:.1f}GB/{mem.total/1024/1024/1024:.1f}GB ({mem.percent}%)"
            
            # GPU显存 - 根据当前使用的设备获取对应GPU
            gpu_mem_str = "未使用"
            if torch.cuda.is_available():
                try:
                    gpu_mem = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
                    gpu_mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 / 1024
                    gpu_percent = (gpu_mem / gpu_mem_total) * 100
                    gpu_mem_str = f"{gpu_mem:.0f}MB/{gpu_mem_total:.0f}MB ({gpu_percent:.1f}%)"
                except:
                    gpu_mem_str = "获取失败"
            
            return f"{device_info} | CPU: {cpu_mem_str} | GPU: {gpu_mem_str}"
        
        def refresh_monitor_status():
            """刷新监控状态"""
            # monitor_logs 是 deque，直接转换为 list 再 join
            logs_str = "\n".join(list(monitor_logs))
            
            # 获取结果表格
            df = monitor.get_results_df()
            
            # 更新统计
            total = monitor.get_total_count()
            anomaly = monitor.get_anomaly_count()
            normal = total - anomaly
            stats = f"总检测: {total} | 异常: {anomaly} | 正常: {normal}"
            
            # 获取内存使用情况
            mem_stats = get_memory_stats()
            
            # 获取最近异常的热力图（按时间倒序，最新的在最前面）
            anomaly_images = []
            # 取最后20个结果并倒序，使最新的显示在最前面
            recent_results = list(monitor.detection_results)[-20:]
            for result in reversed(recent_results):  # 倒序遍历
                if result['is_anomaly'] and result.get('heatmap_path') and os.path.exists(result['heatmap_path']):
                    anomaly_images.append((result['heatmap_path'], result['filename']))
            
            return logs_str, df, anomaly_images, stats, mem_stats
        
        # 绑定按钮事件
        start_monitor_btn.click(
            fn=start_monitoring_fn,
            inputs=[monitor_dir_input, monitor_interval, monitor_algorithm, monitor_device],
            outputs=[monitor_log_box, start_monitor_btn, stop_monitor_btn, confirm_dialog, confirm_count, algorithm_dropdown]
        )
        
        # 检测间隔动态更新 - 监控运行时实时生效
        def on_interval_change(interval):
            """当用户修改检测间隔时实时更新"""
            if monitor.is_monitoring:
                monitor.update_interval(interval)
            return interval
        
        monitor_interval.change(
            fn=on_interval_change,
            inputs=[monitor_interval],
            outputs=[monitor_interval]
        )
        
        # 绑定确认对话框按钮 - 注意这里不需要更新 confirm_count，所以只返回4个值
        detect_existing_btn.click(
            fn=confirm_detect_existing_fn,
            outputs=[monitor_log_box, start_monitor_btn, stop_monitor_btn, confirm_dialog, monitor_algorithm, algorithm_dropdown]
        )
        
        skip_existing_btn.click(
            fn=confirm_skip_existing_fn,
            outputs=[monitor_log_box, start_monitor_btn, stop_monitor_btn, confirm_dialog, monitor_algorithm, algorithm_dropdown]
        )
        
        stop_monitor_btn.click(
            fn=stop_monitoring_fn,
            outputs=[monitor_log_box, start_monitor_btn, stop_monitor_btn, monitor_algorithm, algorithm_dropdown]
        )
        
        cleanup_temp_btn.click(
            fn=cleanup_temp_files_fn,
            outputs=[monitor_log_box]
        )
        
        # 绑定导出按钮事件
        export_zip_btn.click(
            fn=export_monitor_zip_fn,
            outputs=[monitor_log_box, auto_download_html]
        )
        
        # 在线模式：点击表格行跳转到对应热力图
        def on_monitor_select_row(evt: gr.SelectData, gallery_data):
            """
            处理在线模式表格行选择事件
            根据文件名在热力图画廊中查找并跳转
            """
            if evt.index is None or not gallery_data:
                return gr.update()
            
            # 获取选中的文件名（第二列是文件名）
            selected_filename = None
            if isinstance(evt.row_value, (list, tuple)) and len(evt.row_value) >= 2:
                selected_filename = evt.row_value[1]  # 文件名在第二列
            
            if not selected_filename:
                return gr.update()
            
            # 在gallery_data中查找匹配的热力图索引
            # gallery_data格式: [(img_path, caption), ...]
            for idx, (img_path, caption) in enumerate(gallery_data):
                # 检查caption或文件名是否匹配
                if caption == selected_filename or selected_filename in caption or caption in selected_filename:
                    return gr.update(selected_index=idx)
            
            return gr.update()
        
        # 绑定在线模式表格选择事件
        monitor_results_table.select(
            fn=on_monitor_select_row,
            inputs=[recent_anomaly_gallery],
            outputs=[recent_anomaly_gallery]
        )
        
        # 算法切换 - 监控运行中时阻止切换并提示
        def on_algorithm_change(algorithm):
            """当用户修改算法时，如果正在监控则阻止并提示需要先停止"""
            if monitor.is_monitoring:
                current_algo = monitor.algorithm_manager.algorithm_chosen
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 警告: 监控运行中无法切换算法，请先停止监控")
                # 返回日志更新，并恢复算法选择器的值为当前使用的算法
                return "\n".join(list(monitor_logs)), gr.update(value=current_algo), gr.update()
            # 未监控时允许切换，同步到离线模式
            print(f"[算法同步] 在线模式 -> 离线模式: {algorithm}")
            return gr.update(), gr.update(value=algorithm), gr.update(value=algorithm)
        
        monitor_algorithm.change(
            fn=on_algorithm_change,
            inputs=[monitor_algorithm],
            outputs=[monitor_log_box, monitor_algorithm, algorithm_dropdown]
        )
        
        # 使用gr.Timer定期刷新状态（Gradio 4.0+支持）
        # 如果版本不支持，可以使用下面的轮询方式
        try:
            # 尝试使用定时器（Gradio 4.0+）
            timer = gr.Timer(value=2, active=True)
            timer.tick(
                fn=refresh_monitor_status,
                outputs=[monitor_log_box, monitor_results_table, recent_anomaly_gallery, monitor_stats, memory_stats]
            )
        except:
            # 旧版本使用按钮手动刷新
            gr.Markdown("---")
            refresh_btn = gr.Button("🔄 手动刷新状态")
            refresh_btn.click(
                fn=refresh_monitor_status,
                outputs=[monitor_log_box, monitor_results_table, recent_anomaly_gallery, monitor_stats, memory_stats]
            )

    # 统一绑定事件（确保所有组件已定义）
    # 离线模式按钮事件
    run_button.click(
        fn=run_asd_btn_func,
        inputs=[audio_inputs, algorithm_dropdown, device_dropdown],
        outputs=[output_text, results_section, results_table, heatmap_gallery, download_output, run_button, monitor_algorithm],
        queue=True
    )
    
    # 算法选择同步：离线模式改变时更新在线模式
    def sync_algorithm_to_online(algorithm):
        """同步算法选择到在线模式"""
        if algorithm and algorithm != model_manager.algorithm_chosen:
            print(f"[算法同步] 离线模式 -> 在线模式: {algorithm}")
        return gr.update(value=algorithm)
    
    algorithm_dropdown.change(
        fn=sync_algorithm_to_online,
        inputs=[algorithm_dropdown],
        outputs=[monitor_algorithm]
    )

    # 页面加载事件绑定（用于资源管理）
    demo.load(fn=on_page_load, inputs=None, outputs=None)
    
    # 页面加载时同步监控状态到前端控件
    def get_initial_monitor_state():
        """获取当前监控状态，用于页面刷新时恢复控件状态"""
        if monitor.is_monitoring:
            # 监控运行中，恢复运行时的参数
            monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 页面已刷新，监控正在运行中...")
            return (
                monitor.monitor_dir or "",  # 监控目录
                monitor.interval,  # 检测间隔
                gr.update(
                    value=monitor.algorithm_manager.algorithm_chosen,
                    interactive=False
                ),  # 当前算法（禁用选择）
                gr.update(interactive=False),  # 开始按钮禁用
                gr.update(interactive=True),   # 停止按钮启用
                "\n".join(list(monitor_logs))  # 日志（deque自动限制100条）
            )
        else:
            # 监控未运行，使用默认值
            return (
                "",  # 监控目录
                30,  # 默认间隔30秒
                gr.update(
                    value=model_manager.available_algorithms[0] if model_manager.available_algorithms else None,
                    interactive=True
                ),  # 默认算法（启用选择）
                gr.update(interactive=True),   # 开始按钮启用
                gr.update(interactive=False),  # 停止按钮禁用
                "\n".join(list(monitor_logs)) if monitor_logs else "准备就绪"
            )
    
    # 页面加载时同步状态
    demo.load(
        fn=get_initial_monitor_state,
        inputs=None,
        outputs=[
            monitor_dir_input,      # 监控目录路径
            monitor_interval,       # 检测间隔
            monitor_algorithm,      # 算法选择（值和交互状态）
            start_monitor_btn,      # 开始按钮交互状态
            stop_monitor_btn,       # 停止按钮交互状态
            monitor_log_box         # 日志显示
        ]
    )

# 全局资源清理函数
def cleanup_resources():
    """应用退出时清理资源"""
    print(f"\n{'='*60}")
    print(f"[全局清理] 🧹 应用退出，执行全局资源清理...")
    print(f"{'='*60}")
    
    global monitor
    if monitor:
        print(f"[全局清理] 调用 monitor.cleanup()...")
        monitor.cleanup()  # 使用完整的cleanup方法
    else:
        print(f"[全局清理] ℹ 监控器未初始化，执行备用清理...")
        # 备用清理
        if torch.cuda.is_available():
            try:
                print(f"[全局清理] 清空 CUDA 缓存...")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**2
                    print(f"[全局清理]   GPU {i}: {allocated:.1f}MB -> 0MB")
                torch.cuda.empty_cache()
                print(f"[全局清理] ✓ CUDA缓存已清空")
            except Exception as e:
                print(f"[全局清理] ⚠ CUDA缓存清空失败: {e}")
        gc.collect()
        print(f"[全局清理] ✓ 垃圾回收已执行")
    
    print(f"[全局清理] ✓ 全局资源清理完成")
    print(f"{'='*60}\n")

# 注册退出清理
import atexit
atexit.register(cleanup_resources)

if __name__ == "__main__":
    # 使用绝对路径加载配置
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "config", "algorithms.yaml")
    config = ConfigManager(config_path)
    
    server_config = config.config.get('server', {})
    try:
        demo.launch(
            server_name=server_config.get('server_name', '0.0.0.0'),
            server_port=server_config.get('port', 8002),
            share=server_config.get('share', False),
            inbrowser=server_config.get('inbrowser', True),
            show_error=True,
            head=js_autoscroll
        )
    finally:
        cleanup_resources()
