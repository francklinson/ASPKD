"""
统一接口版本的Web GUI应用程序
使用新的统一算法接口
"""

import gc
import os
import sys
import tempfile
import time
import zipfile
import threading
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque

import torch
import torch.cuda as cuda
import gradio as gr
import pandas as pd

# 使用统一接口
from algorithms import create_detector, list_available_algorithms
from core import ConfigManager
from data_prepocessing import Preprocessor


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
        yield self.log_messages, gr.update(visible=False), None, None, None, gr.update(interactive=False)
    
    def update_log_and_action(self, msg, run_button_action):
        """更新日志，同时传递GUI控件响应"""
        self.generate_log(msg)
        yield self.log_messages, gr.update(visible=False), None, None, None, run_button_action


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
    
    def __init__(self):
        self.detector = None
        self.algorithm_chosen = ""
        # 使用绝对路径加载配置
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "algorithms.yaml")
        print(f"[DEBUG] UnifiedAlgorithmManager: config_path={config_path}")
        print(f"[DEBUG] Config file exists: {os.path.exists(config_path)}")
        self.config = ConfigManager(config_path)
        print(f"[DEBUG] Config loaded, base_dir={self.config.base_dir}")
        print(f"[DEBUG] Models in config: {list(self.config.config.get('models', {}).keys())}")
        # 获取可用算法列表
        self.available_algorithms = self._get_gui_algorithms()
    
    def _get_gui_algorithms(self) -> List[str]:
        """获取GUI可用的算法列表"""
        # 暂时只启用已完整实现的Dinomaly系列
        # 后续可以逐步添加其他算法
        return [
            "dinomaly_dinov3_small",
            "dinomaly_dinov3_large",
            "dinomaly_dinov2_small",
            "dinomaly_dinov2_large",
        ]
    
    def update_algorithm(self, algorithm_choice):
        """更新算法"""
        assert algorithm_choice in self.available_algorithms
        
        if self.detector is not None and algorithm_choice == self.algorithm_chosen:
            return
        
        # 清理旧模型
        if self.detector is not None:
            self.detector.release()
            self.detector = None
            self._clear_cuda_cache()
        
        # 使用统一接口创建新检测器
        try:
            # 调试信息
            model_path = self.config.get_model_path('dinomaly', 'dinov3_small')
            print(f"[DEBUG] Config base_dir: {self.config.base_dir}")
            print(f"[DEBUG] Model path from config: {model_path}")
            
            self.detector = create_detector(
                algorithm_name=algorithm_choice,
                config_manager=self.config
            )
            self.algorithm_chosen = algorithm_choice
            print(f"算法成功切换到: {self.algorithm_chosen}")
            print(f"[DEBUG] Detector model_path: {self.detector.model_path}")
        except Exception as e:
            print(f"加载算法失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @staticmethod
    def _clear_cuda_cache():
        """清除GPU缓存"""
        torch.cuda.empty_cache()
        if cuda.is_available():
            cuda.empty_cache()
            cuda.synchronize()
        gc.collect()


class DirectoryMonitor:
    """
    目录监控器
    用于监控指定目录下的新增音频文件，自动进行异常检测
    """
    
    def __init__(self, algorithm_manager: UnifiedAlgorithmManager):
        self.algorithm_manager = algorithm_manager
        self.preprocessor = None
        self.monitor_thread = None
        self.is_monitoring = False
        self.monitor_dir = None
        self.interval = 5  # 默认监控间隔5秒
        self.processed_files = set()  # 已处理的文件集合
        self.detection_results = deque(maxlen=1000)  # 保存最近的检测结果
        self.status_callback = None  # 状态回调函数
        self.detect_existing_on_start = False  # 启动时是否检测已有文件
        
    def set_preprocessor(self, ref_file: str):
        """设置预处理器"""
        self.preprocessor = Preprocessor(ref_file=ref_file)
        
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
        if self.is_monitoring:
            self._log(f"正在切换算法到: {algorithm}...")
            try:
                # 释放旧模型
                if self.algorithm_manager.detector:
                    self.algorithm_manager.detector.release()
                
                # 切换算法
                self.algorithm_manager.update_algorithm(algorithm)
                self.algorithm_manager.detector.load_model()
                self._log(f"算法切换成功: {algorithm}")
                return True
            except Exception as e:
                self._log(f"算法切换失败: {e}")
                return False
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
        
    def process_single_file(self, audio_file: str) -> Optional[Dict]:
        """处理单个音频文件"""
        try:
            filename = os.path.basename(audio_file)
            self._log(f"开始处理: {filename}")
            
            # 1. 音频预处理
            picture_file_dict = self.preprocessor.process_audio([audio_file], save_dir="slice")
            
            if not picture_file_dict:
                self._log(f"预处理失败: {filename}")
                return None
                
            # 2. 准备图像列表
            picture_file_list = []
            for k, v in picture_file_dict.items():
                if isinstance(v, dict):
                    if v.get("dk"):
                        picture_file_list.append(v["dk"])
                    if v.get("qzgy"):
                        picture_file_list.append(v["qzgy"])
                        
            if not picture_file_list:
                self._log(f"未找到目标音频: {filename}")
                return None
                
            # 3. 异常检测
            detection_results = self.algorithm_manager.detector.predict_batch(picture_file_list)
            
            # 4. 构建结果
            max_score = 0
            is_anomaly = False
            heatmap_path = None
            
            for img_path, result in zip(picture_file_list, detection_results):
                if result.anomaly_score > max_score:
                    max_score = result.anomaly_score
                    is_anomaly = result.is_anomaly
                    heatmap_path = result.metadata.get('heatmap_path') if result.metadata else None
                    
            result_dict = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': filename,
                'filepath': audio_file,
                'anomaly_score': max_score,
                'is_anomaly': is_anomaly,
                'status': '异常' if is_anomaly else '正常',
                'heatmap_path': heatmap_path,
                'processed_images': picture_file_list
            }
            
            self._log(f"处理完成: {filename} - 分数: {max_score:.4f} - {'异常' if is_anomaly else '正常'}")
            return result_dict
            
        except Exception as e:
            self._log(f"处理失败 {audio_file}: {str(e)}")
            return None
            
    def monitoring_loop(self):
        """监控主循环"""
        self._log("=" * 50)
        self._log("监控线程已启动")
        self._log(f"监控目录: {self.monitor_dir}")
        self._log(f"检测间隔: {self.interval}秒")
        self._log("=" * 50)
        
        # 加载模型
        try:
            self.algorithm_manager.detector.load_model()
            self._log("模型加载成功")
        except Exception as e:
            self._log(f"模型加载失败: {e}")
            self.is_monitoring = False
            return
            
        # 初始化时扫描已有文件
        existing_files = self.get_audio_files()
        
        # 如果设置了检测已有文件标志，先批量处理已有文件
        if self.detect_existing_on_start and existing_files:
            self._log(f"开始对 {len(existing_files)} 个已有文件进行检测...")
            batch_results = []  # 临时存储结果，一次性添加
            processed_count = 0
            
            for audio_file in existing_files:
                if not self.is_monitoring:
                    break
                result = self.process_single_file(audio_file)
                if result:
                    batch_results.append(result)
                self.processed_files.add(audio_file)
                processed_count += 1
                # 每处理5个文件更新一次日志，减少日志输出
                if processed_count % 5 == 0:
                    self._log(f"已处理 {processed_count}/{len(existing_files)} 个文件...")
                time.sleep(0.1)  # 稍微减少延迟
            
            # 一次性将所有结果添加到检测队列
            for result in batch_results:
                self.detection_results.append(result)
            
            self._log(f"✅ 已有文件检测完成，共处理 {processed_count} 个，发现 {len(batch_results)} 个有效结果")
        else:
            # 否则只是记录已有文件，不进行检测
            self.processed_files = set(existing_files)
            self._log(f"初始化完成，已记录 {len(existing_files)} 个现有文件（跳过检测）")
        
        while self.is_monitoring:
            try:
                # 检测新文件
                new_files = self.detect_new_files()
                
                if new_files:
                    self._log(f"发现 {len(new_files)} 个新文件")
                    
                    for audio_file in new_files:
                        if not self.is_monitoring:
                            break
                            
                        # 处理文件
                        result = self.process_single_file(audio_file)
                        
                        if result:
                            self.detection_results.append(result)
                            
                        # 标记为已处理
                        self.processed_files.add(audio_file)
                        
                        # 短暂休息，避免占用过多资源
                        time.sleep(0.5)
                        
                # 等待下一次检测
                time.sleep(self.interval)
                
            except Exception as e:
                self._log(f"监控循环出错: {e}")
                time.sleep(self.interval)
                
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
            return pd.DataFrame(columns=['时间', '文件名', '异常分数', '状态'])
            
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


def run_asd_btn_func(audio_files, algorithm_choice: str, progress=gr.Progress()):
    """主处理函数"""
    # 加载配置（使用绝对路径）
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "config", "algorithms.yaml")
    config = ConfigManager(config_path)
    ref_file = config.config.get('preprocessing', {}).get('ref_file', 'ref/渡口片段10s.wav')
    # 如果ref_file是相对路径，转为绝对路径
    if not os.path.isabs(ref_file):
        ref_file = os.path.join(project_root, ref_file)
    
    # 初始化组件
    p = Preprocessor(ref_file=ref_file)
    lm = LogManager()
    am = UnifiedAlgorithmManager()
    
    print("输入参数:", audio_files, algorithm_choice)
    
    if not audio_files:
        yield from lm.update_log_and_action("请上传至少一个音频文件", gr.update(interactive=True))
        return None, gr.update(visible=False), None, None, None, gr.update(interactive=True)
    
    # 加载算法
    try:
        am.update_algorithm(algorithm_choice)
        am.detector.load_model()
    except Exception as e:
        yield from lm.update_log_and_action(f"算法加载失败: {str(e)}", gr.update(interactive=True))
        return None, gr.update(visible=False), None, None, None, gr.update(interactive=True)
    
    # [1] 音频预处理
    yield from lm.update_log("执行音频预处理...")
    try:
        picture_file_dict = p.process_audio(audio_files, save_dir="slice")
    except Exception as e:
        yield from lm.update_log_and_action(f"音频预处理失败: {str(e)}", gr.update(interactive=True))
        return None, gr.update(visible=False), None, None, None, gr.update(interactive=True)
    
    if picture_file_dict is None or not picture_file_dict:
        yield from lm.update_log_and_action("音频预处理返回空结果，请检查音频文件格式", gr.update(interactive=True))
        return None, gr.update(visible=False), None, None, None, gr.update(interactive=True)
    
    yield from lm.update_log("完成目标音频搜索和切分！！")
    
    # [2] 准备图像列表
    picture_file_list = []
    for k, v in picture_file_dict.items():
        if isinstance(v, dict):
            if v.get("dk"):
                picture_file_list.append(v["dk"])
            if v.get("qzgy"):
                picture_file_list.append(v["qzgy"])
    
    if not picture_file_list:
        yield from lm.update_log_and_action("上传的素材中没有找到目标音频，请检查!", gr.update(interactive=True))
        return None, gr.update(visible=False), None, None, None, gr.update(interactive=True)
    
    # [3] 执行异常检测
    yield from lm.update_log(f"使用算法: {am.algorithm_chosen}")
    yield from lm.update_log("执行异常预测...")
    
    try:
        # 使用统一接口批量推理
        detection_results = am.detector.predict_batch(picture_file_list)
        
        # 构建结果字典和热力图路径字典
        pred_res_dict = {}
        heatmap_paths_dict = {}
        for img_path, result in zip(picture_file_list, detection_results):
            filename = os.path.basename(img_path)
            pred_res_dict[filename] = (result.anomaly_score, 1 if result.is_anomaly else 0)
            # 从metadata中获取热力图路径
            heatmap_path = result.metadata.get('heatmap_path') if result.metadata else None
            if heatmap_path:
                heatmap_paths_dict[filename] = heatmap_path
                print(f"[DEBUG] Found heatmap for {filename}: {heatmap_path}")
        
        yield from lm.update_log("检测完成!")
    except Exception as e:
        yield from lm.update_log_and_action(f"检测失败: {str(e)}", gr.update(interactive=True))
        return None, gr.update(visible=False), None, None, None, gr.update(interactive=True)
    
    # [4] 输出结果
    yield from lm.update_log("整理输出结果中...")
    
    results = []
    for k, v in pred_res_dict.items():
        results.append({"filename": k, "anomaly_score": v[0], "is_anomaly": v[1]})
    
    df_results = pd.DataFrame(results)
    df_results['状态'] = df_results['is_anomaly'].apply(lambda x: '异常' if x == 1 else '正常')
    df_results = df_results[['filename', 'anomaly_score', 'is_anomaly', '状态']]
    df_results.columns = ['文件名', '异常分数', '是否异常', '状态']
    
    temp_dir = tempfile.mkdtemp()
    excel_path = Export.create_excel_report(results, temp_dir)
    
    zip_path = os.path.join(temp_dir, f'asd_results_{am.algorithm_chosen}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')
    images_for_zip = picture_file_list
    zip_path = Export.create_zip_with_results(zip_path, excel_path, images_for_zip)
    
    # 为Gallery准备带caption的数据 (热力图路径, caption)
    gallery_data = []
    for img_path in picture_file_list:
        filename = os.path.basename(img_path)
        heatmap_path = heatmap_paths_dict.get(filename)
        
        if heatmap_path and os.path.exists(heatmap_path):
            # 使用热力图
            caption = os.path.splitext(filename)[0]
            gallery_data.append((heatmap_path, caption))
            print(f"[DEBUG] Added heatmap to gallery: {heatmap_path}")
        elif os.path.exists(img_path):
            # 回退到原图
            caption = os.path.splitext(filename)[0]
            gallery_data.append((img_path, caption))
            print(f"[DEBUG] No heatmap found, using original image: {img_path}")
        else:
            print(f"警告: 图像文件不存在: {img_path}")

    print(f"调试: gallery_data 数量: {len(gallery_data)}")
    print(f"调试: df_results 行数: {len(df_results)}")
    print(f"调试: picture_file_list 数量: {len(picture_file_list)}")
    print(f"调试: heatmap_paths_dict 数量: {len(heatmap_paths_dict)}")

    # 确保gallery_data不为空，如果为空则提供一个提示
    if not gallery_data:
        print("警告: gallery_data为空，没有图像可显示")

    lm.generate_log(f"处理完成! 共处理 {len(results)} 个文件，请查看下方结果!")

    # 释放资源
    am.detector.release()

    yield lm.log_messages, gr.update(visible=True), df_results, gallery_data, zip_path, gr.update(interactive=True)


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

am = UnifiedAlgorithmManager()

# 页面加载计数器和最后活跃时间（用于检测用户是否已离开）
_session_count = 0
_last_active_time = time.time()
_session_lock = threading.Lock()

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
        print(f"[Session] 页面离开，当前会话数: {_session_count}")
        
        # 如果没有会话了，停止监控并释放资源
        if _session_count == 0 and monitor and monitor.is_monitoring:
            print("[Session] 所有会话已离开，停止监控...")
            monitor.stop_monitoring()
            if monitor.algorithm_manager and monitor.algorithm_manager.detector:
                monitor.algorithm_manager.detector.release()
            torch.cuda.empty_cache()
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
                    choices=am.available_algorithms,
                    value=am.available_algorithms[0] if am.available_algorithms else None,
                    label="🔧 选择异常检测算法"
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

        # 绑定按钮事件
        run_button.click(
            fn=run_asd_btn_func,
            inputs=[audio_inputs, algorithm_dropdown],
            outputs=[output_text, results_section, results_table, heatmap_gallery, download_output, run_button],
            queue=True
        )

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
        monitor = DirectoryMonitor(am)
        monitor_logs = []
        
        def update_monitor_status(message: str):
            """更新监控状态日志 - message 已经包含时间戳"""
            global monitor_logs
            monitor_logs.append(message)  # message 已由 _log() 添加时间戳
            # 保留最近100条日志
            if len(monitor_logs) > 100:
                monitor_logs = monitor_logs[-100:]
            return "\n".join(monitor_logs)
        
        monitor.set_status_callback(update_monitor_status)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 监控设置
                monitor_dir_input = gr.Textbox(
                    label="📁 监控目录路径",
                    placeholder="输入要监控的目录绝对路径",
                    value=""
                )
                
                monitor_interval = gr.Slider(
                    label="⏱️ 检测间隔（秒）",
                    minimum=1,
                    maximum=60,
                    value=5,
                    step=1
                )
                
                monitor_algorithm = gr.Dropdown(
                    choices=am.available_algorithms,
                    value=am.available_algorithms[0] if am.available_algorithms else None,
                    label="🔧 检测算法"
                )
                
                with gr.Row():
                    start_monitor_btn = gr.Button("▶️ 开始监控", variant="primary")
                    stop_monitor_btn = gr.Button("⏹️ 停止监控", variant="secondary")
                    clear_monitor_btn = gr.Button("🗑️ 清空结果", variant="secondary")
                
                # 统计信息
                monitor_stats = gr.Textbox(
                    label="📊 统计信息",
                    value="总检测: 0 | 异常: 0 | 正常: 0",
                    interactive=False
                )

            with gr.Column(scale=2):
                # 监控日志
                monitor_log_box = gr.Textbox(
                    label="📝 监控日志",
                    lines=10,
                    max_lines=15,
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
                object_fit="contain"
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
        
        def start_monitoring_fn(dir_path, interval, algorithm):
            """开始监控 - 先检查是否有已有文件"""
            global monitor_logs
            
            if not dir_path or not os.path.exists(dir_path):
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 错误: 目录不存在")
                return "\n".join(monitor_logs), gr.update(), gr.update(), gr.update(visible=False), gr.update()
            
            # 检查是否已在监控中
            if monitor.is_monitoring:
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 监控已在运行中")
                return "\n".join(monitor_logs), gr.update(), gr.update(), gr.update(visible=False), gr.update()
            
            # 扫描目录下的音频文件
            existing_files = get_audio_files_in_dir(dir_path)
            
            if existing_files:
                # 有已有文件，显示确认对话框
                global pending_monitor_params
                pending_monitor_params = {
                    'dir_path': dir_path,
                    'interval': interval,
                    'algorithm': algorithm,
                    'existing_files': existing_files
                }
                count_msg = f"发现 {len(existing_files)} 个音频文件"
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {count_msg}，等待用户选择...")
                # 注意：confirm_count是Markdown，通过gr.update(value=...)更新
                return "\n".join(monitor_logs), gr.update(), gr.update(), gr.update(visible=True), gr.update(value=f"**音频文件数量: {len(existing_files)} 个**")
            
            # 没有已有文件，直接开始监控
            result = _do_start_monitoring(dir_path, interval, algorithm, detect_existing=False)
            # _do_start_monitoring 返回4个值，需要添加 confirm_count 的输出
            return result[0], result[1], result[2], result[3], gr.update()
        
        def _do_start_monitoring(dir_path, interval, algorithm, detect_existing=False):
            """执行实际开始监控的操作"""
            global monitor_logs
            
            # 设置预处理器
            project_root = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(project_root, "config", "algorithms.yaml")
            config = ConfigManager(config_path)
            ref_file = config.config.get('preprocessing', {}).get('ref_file', 'ref/渡口片段10s.wav')
            if not os.path.isabs(ref_file):
                ref_file = os.path.join(project_root, ref_file)
            
            monitor.set_preprocessor(ref_file)
            monitor.set_monitor_params(dir_path, interval)
            
            # 切换算法
            try:
                am.update_algorithm(algorithm)
            except Exception as e:
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 算法切换失败: {e}")
                return "\n".join(monitor_logs), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False)

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
                return "\n".join(monitor_logs), gr.update(interactive=False), gr.update(interactive=True), gr.update(visible=False), gr.update(interactive=False)
            else:
                return "\n".join(monitor_logs), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=True)
        
        def confirm_detect_existing_fn():
            """确认检测已有文件"""
            global monitor_logs, pending_monitor_params
            
            if not pending_monitor_params:
                return "\n".join(monitor_logs), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=True)
            
            monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 选择：检测已有文件")
            
            params = pending_monitor_params
            pending_monitor_params = {}  # 清空
            
            return _do_start_monitoring(
                params['dir_path'], 
                params['interval'], 
                params['algorithm'], 
                detect_existing=True
            )
        
        def confirm_skip_existing_fn():
            """确认跳过已有文件"""
            global monitor_logs, pending_monitor_params
            
            if not pending_monitor_params:
                return "\n".join(monitor_logs), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=True)
            
            monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 选择：跳过已有文件")
            
            params = pending_monitor_params
            pending_monitor_params = {}  # 清空
            
            return _do_start_monitoring(
                params['dir_path'], 
                params['interval'], 
                params['algorithm'], 
                detect_existing=False
            )
        
        def stop_monitoring_fn():
            """停止监控"""
            monitor.stop_monitoring()
            return "\n".join(monitor_logs), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)
        
        def clear_monitor_results_fn():
            """清空结果"""
            monitor.clear_results()
            return pd.DataFrame(columns=['时间', '文件名', '异常分数', '是否异常', '状态']), [], "总检测: 0 | 异常: 0 | 正常: 0"
        
        def refresh_monitor_status():
            """刷新监控状态"""
            global monitor_logs
            
            # 获取结果表格
            df = monitor.get_results_df()
            
            # 更新统计
            total = monitor.get_total_count()
            anomaly = monitor.get_anomaly_count()
            normal = total - anomaly
            stats = f"总检测: {total} | 异常: {anomaly} | 正常: {normal}"
            
            # 获取最近异常的热力图（按时间倒序，最新的在最前面）
            anomaly_images = []
            # 取最后20个结果并倒序，使最新的显示在最前面
            recent_results = list(monitor.detection_results)[-20:]
            for result in reversed(recent_results):  # 倒序遍历
                if result['is_anomaly'] and result.get('heatmap_path') and os.path.exists(result['heatmap_path']):
                    anomaly_images.append((result['heatmap_path'], result['filename']))
            
            return "\n".join(monitor_logs), df, anomaly_images, stats
        
        # 绑定按钮事件
        start_monitor_btn.click(
            fn=start_monitoring_fn,
            inputs=[monitor_dir_input, monitor_interval, monitor_algorithm],
            outputs=[monitor_log_box, start_monitor_btn, stop_monitor_btn, confirm_dialog, confirm_count]
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
            outputs=[monitor_log_box, start_monitor_btn, stop_monitor_btn, confirm_dialog, monitor_algorithm]
        )
        
        skip_existing_btn.click(
            fn=confirm_skip_existing_fn,
            outputs=[monitor_log_box, start_monitor_btn, stop_monitor_btn, confirm_dialog, monitor_algorithm]
        )
        
        stop_monitor_btn.click(
            fn=stop_monitoring_fn,
            outputs=[monitor_log_box, start_monitor_btn, stop_monitor_btn, monitor_algorithm]
        )
        
        clear_monitor_btn.click(
            fn=clear_monitor_results_fn,
            outputs=[monitor_results_table, recent_anomaly_gallery, monitor_stats]
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
            global monitor_logs
            if monitor.is_monitoring:
                current_algo = monitor.algorithm_manager.algorithm_chosen
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 警告: 监控运行中无法切换算法，请先停止监控")
                # 返回日志更新，并恢复算法选择器的值为当前使用的算法
                return "\n".join(monitor_logs), gr.update(value=current_algo)
            # 未监控时允许切换，不做任何阻止
            return gr.update(), gr.update()
        
        monitor_algorithm.change(
            fn=on_algorithm_change,
            inputs=[monitor_algorithm],
            outputs=[monitor_log_box, monitor_algorithm]
        )
        
        # 使用gr.Timer定期刷新状态（Gradio 4.0+支持）
        # 如果版本不支持，可以使用下面的轮询方式
        try:
            # 尝试使用定时器（Gradio 4.0+）
            timer = gr.Timer(value=2, active=True)
            timer.tick(
                fn=refresh_monitor_status,
                outputs=[monitor_log_box, monitor_results_table, recent_anomaly_gallery, monitor_stats]
            )
        except:
            # 旧版本使用按钮手动刷新
            gr.Markdown("---")
            refresh_btn = gr.Button("🔄 手动刷新状态")
            refresh_btn.click(
                fn=refresh_monitor_status,
                outputs=[monitor_log_box, monitor_results_table, recent_anomaly_gallery, monitor_stats]
            )

    # 页面加载事件绑定（用于资源管理）
    demo.load(fn=on_page_load, inputs=None, outputs=None)
    
    # 页面加载时同步监控状态到前端控件
    def get_initial_monitor_state():
        """获取当前监控状态，用于页面刷新时恢复控件状态"""
        global monitor_logs
        
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
                "\n".join(monitor_logs[-100:])  # 日志（最近100条）
            )
        else:
            # 监控未运行，使用默认值
            return (
                "",  # 监控目录
                5,   # 默认间隔
                gr.update(
                    value=am.available_algorithms[0] if am.available_algorithms else None,
                    interactive=True
                ),  # 默认算法（启用选择）
                gr.update(interactive=True),   # 开始按钮启用
                gr.update(interactive=False),  # 停止按钮禁用
                "\n".join(monitor_logs[-100:]) if monitor_logs else "准备就绪"
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
    global monitor
    if monitor and monitor.is_monitoring:
        monitor.stop_monitoring()
    if monitor and monitor.algorithm_manager and monitor.algorithm_manager.detector:
        monitor.algorithm_manager.detector.release()
    torch.cuda.empty_cache()
    gc.collect()
    print("[Cleanup] 资源已清理")

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
