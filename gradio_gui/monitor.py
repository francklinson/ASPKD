"""
目录监控器 - 实时监控指定目录下的音频文件
"""
import os
import sys
import time
import threading
import gc
from datetime import datetime
from collections import deque
from typing import List, Dict, Optional, Callable

import torch
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing import Preprocessor


class DirectoryMonitor:
    """目录监控器 - 用于监控指定目录下的新增音频文件，自动进行异常检测"""

    def __init__(self, algorithm_manager, device: str = 'auto'):
        self.algorithm_manager = algorithm_manager
        self.device = device
        self.preprocessor = None
        self.monitor_thread = None
        self.is_monitoring = False
        self.monitor_dir = None
        self.interval = 30
        self.processed_files = set()
        self.detection_results = deque(maxlen=1000)
        self.status_callback = None
        self.detect_existing_on_start = False

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

    def set_status_callback(self, callback: Callable[[str], None]):
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
        return list(current_files - self.processed_files)

    def preprocess_files(self, audio_files: List[str]) -> Dict[str, List[str]]:
        """批量预处理音频文件，返回 {audio_file: [image_paths]}"""
        file_images_map = {}

        if not audio_files:
            return file_images_map

        try:
            print(f"[预处理] 批量处理 {len(audio_files)} 个文件...")

            # 一次性传入所有文件进行批量并行处理
            picture_file_dict = self.preprocessor.process_audio(audio_files, save_dir="slice")

            if not picture_file_dict:
                print(f"[预处理] ✗ 批量预处理失败")
                return file_images_map

            # 整理结果
            for audio_file in audio_files:
                filename = os.path.basename(audio_file)

                if audio_file not in picture_file_dict:
                    print(f"[预处理] ✗ {filename}: 未找到目标音频")
                    continue

                picture_file_list = []
                v = picture_file_dict[audio_file]
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
            print(f"[预处理] ✗ 批量处理失败: {str(e)}")
            import traceback
            traceback.print_exc()

        return file_images_map

    def process_batch_files(self, audio_files: List[str]) -> List[Dict]:
        """批量处理多个音频文件"""
        if not audio_files:
            return []

        print(f"\n[批量检测] >>> 开始处理 {len(audio_files)} 个文件")
        self._log(f"开始批量处理 {len(audio_files)} 个新文件...")

        # 1. 批量预处理
        start_time = time.time()
        file_images_map = self.preprocess_files(audio_files)
        preprocess_time = time.time() - start_time

        if not file_images_map:
            print("[批量检测] ✗ 所有文件预处理失败")
            self._log("预处理失败，没有有效的图像")
            return []

        valid_files = list(file_images_map.keys())
        total_images = sum(len(images) for images in file_images_map.values())
        print(f"[批量检测] ✓ 预处理完成: {len(valid_files)}/{len(audio_files)} 个文件，共 {total_images} 个图像")
        self._log(f"预处理完成: {len(valid_files)} 个文件，{total_images} 个图像")

        # 2. 收集所有图像
        all_images = []
        file_image_ranges = {}
        current_idx = 0

        for audio_file, images in file_images_map.items():
            file_image_ranges[audio_file] = (current_idx, current_idx + len(images))
            all_images.extend(images)
            current_idx += len(images)

        # 3. 批量异常检测
        print(f"[批量检测] 执行批量异常检测，共 {len(all_images)} 个图像...")
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
        for audio_file, (start_idx, end_idx) in file_image_ranges.items():
            filename = os.path.basename(audio_file)
            file_results = all_detection_results[start_idx:end_idx]

            max_score = 0
            is_anomaly = False
            heatmap_path = None

            for idx, result in enumerate(file_results):
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
                'processed_images': file_images_map[audio_file]
            }
            results.append(result_dict)
            print(f"[批量检测]   - {filename}: 分数={max_score:.4f}, 状态={status_str}")

        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        print(f"\n[批量检测] ✅ 完成: 总计 {len(results)} 个, 异常 {anomaly_count} 个")
        self._log(f"批量检测完成: {len(results)} 个文件, 异常 {anomaly_count} 个")

        return results

    def monitoring_loop(self):
        """监控主循环"""
        print("\n" + "=" * 60)
        print("[监控] 监控线程已启动")
        print(f"[监控] 监控目录: {self.monitor_dir}")
        print("=" * 60)

        self._log("=" * 50)
        self._log("监控线程已启动")
        self._log(f"监控目录: {self.monitor_dir}")
        self._log("=" * 50)

        # 加载模型
        try:
            if self.algorithm_manager.is_model_loaded():
                self._log("模型已加载，直接使用")
            else:
                self.algorithm_manager.load_model()
                self._log("模型加载成功")
        except Exception as e:
            self._log(f"模型加载失败: {e}")
            self.is_monitoring = False
            return

        # 初始化时扫描已有文件
        existing_files = self.get_audio_files()
        print(f"[监控] 发现 {len(existing_files)} 个现有音频文件")

        if self.detect_existing_on_start and existing_files:
            self._log(f"开始对 {len(existing_files)} 个已有文件进行批量检测...")
            batch_results = self.process_batch_files(existing_files)
            for result in batch_results:
                self.detection_results.append(result)
            for audio_file in existing_files:
                self.processed_files.add(audio_file)
            self._log(f"✅ 已有文件检测完成，共处理 {len(existing_files)} 个")
        else:
            self.processed_files = set(existing_files)
            self._log(f"初始化完成，已记录 {len(existing_files)} 个现有文件（跳过检测）")

        # 监控循环
        scan_count = 0
        while self.is_monitoring:
            try:
                scan_count += 1
                print(f"\n[监控] 第 {scan_count} 次扫描目录...")

                new_files = self.detect_new_files()

                if new_files:
                    print(f"[监控] >>> 发现 {len(new_files)} 个新文件")
                    self._log(f"发现 {len(new_files)} 个新文件")

                    batch_results = self.process_batch_files(new_files)
                    for result in batch_results:
                        self.detection_results.append(result)
                    for audio_file in new_files:
                        self.processed_files.add(audio_file)

                time.sleep(self.interval)

            except Exception as e:
                print(f"[监控] ✗ 监控循环出错: {e}")
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

        if not self.monitor_dir or not os.path.exists(self.monitor_dir):
            self._log("错误: 未设置监控目录或目录不存在")
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

        data = [{
            '时间': result['timestamp'],
            '文件名': result['filename'],
            '异常分数': result['anomaly_score'],
            '是否异常': result['is_anomaly'],
            '状态': result['status']
        } for result in self.detection_results]

        df = pd.DataFrame(data)
        return df.iloc[::-1].reset_index(drop=True)

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

        # 停止监控
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)

        # 释放模型
        if self.algorithm_manager and self.algorithm_manager.is_model_loaded():
            try:
                self.algorithm_manager.unload_model()
                print(f"[Cleanup] ✓ 模型已释放")
            except Exception as e:
                print(f"[Cleanup] ⚠ 模型释放失败: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # 清理数据
        self.detection_results.clear()
        self.processed_files.clear()

        if self.preprocessor:
            del self.preprocessor
            self.preprocessor = None

        gc.collect()
        print(f"[Cleanup] ✓ 资源清理完成")
        print(f"{'='*60}\n")
