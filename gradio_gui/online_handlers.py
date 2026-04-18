"""
在线模式处理函数 - 实时监控逻辑
"""
import os
import sys
from datetime import datetime
from collections import deque
from typing import List, Dict, Optional

import gradio as gr

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import ConfigManager

try:
    from .utils import get_memory_stats, cleanup_all_temp_files, generate_auto_download_html
    from .export import ExportManager
except ImportError:
    from gradio_gui.utils import get_memory_stats, cleanup_all_temp_files, generate_auto_download_html
    from gradio_gui.export import ExportManager

# 监控日志使用deque限制大小
monitor_logs = deque(maxlen=100)

# 存储临时参数（用于确认对话框）
pending_monitor_params: Dict = {}


def update_monitor_status(message: str) -> str:
    """更新监控状态日志"""
    monitor_logs.append(message)
    return "\n".join(monitor_logs)


def get_audio_files_in_dir(dir_path: str) -> List[str]:
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


def start_monitoring_fn(dir_path: str, interval: int, algorithm: str, device: str, monitor):
    """开始监控 - 先检查是否有已有文件"""
    global pending_monitor_params

    if not dir_path or not os.path.exists(dir_path):
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 错误: 目录不存在")
        return "\n".join(list(monitor_logs)), gr.update(), gr.update(), gr.update(visible=False), gr.update(), gr.update(), gr.update()

    if monitor.is_monitoring:
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 监控已在运行中")
        return "\n".join(list(monitor_logs)), gr.update(), gr.update(), gr.update(visible=False), gr.update(), gr.update(), gr.update()

    # 扫描目录下的音频文件
    existing_files = get_audio_files_in_dir(dir_path)

    if existing_files:
        # 有已有文件，显示确认对话框
        pending_monitor_params = {
            'dir_path': dir_path,
            'interval': interval,
            'algorithm': algorithm,
            'device': device,
            'existing_files': existing_files
        }
        count_msg = f"发现 {len(existing_files)} 个音频文件"
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {count_msg}，等待用户选择...")
        return (
            "\n".join(list(monitor_logs)),
            gr.update(),
            gr.update(),
            gr.update(visible=True),
            gr.update(value=f"**音频文件数量: {len(existing_files)} 个**"),
            gr.update(value=algorithm),
            gr.update()
        )

    # 没有已有文件，直接开始监控
    return _do_start_monitoring(dir_path, interval, algorithm, device, False, monitor)


def _do_start_monitoring(dir_path: str, interval: int, algorithm: str,
                        device: str, detect_existing: bool, monitor):
    """执行实际开始监控的操作"""
    global pending_monitor_params

    monitor.device = device

    # 设置预处理器
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "config.yaml")
    config = ConfigManager(config_path)

    ref_file = config.config.get('preprocessing', {}).get('ref_file', 'ref/渡口片段10s.wav')
    if not os.path.isabs(ref_file):
        ref_file = os.path.join(project_root, ref_file)

    split_method = config.config.get('preprocessing', {}).get('split_method', 'mfcc_dtw')
    shazam_config = config.config.get('preprocessing', {}).get('shazam', {})

    monitor.set_preprocessor(
        ref_file=ref_file,
        split_method=split_method,
        shazam_threshold=shazam_config.get('threshold', 10),
        shazam_auto_match=shazam_config.get('auto_match', False),
        max_workers=shazam_config.get('max_workers', 1)
    )
    monitor.set_monitor_params(dir_path, interval)

    # 切换算法
    try:
        monitor.algorithm_manager.update_algorithm(algorithm, device=device)
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 使用设备: {device}")
    except Exception as e:
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 算法切换失败: {e}")
        return "\n".join(list(monitor_logs)), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=False), gr.update(), gr.update()

    monitor.detect_existing_on_start = detect_existing

    # 如果有已有文件且选择跳过
    if not detect_existing and pending_monitor_params.get('existing_files'):
        for f in pending_monitor_params['existing_files']:
            monitor.processed_files.add(f)
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 已跳过 {len(pending_monitor_params['existing_files'])} 个已有文件")

    success = monitor.start_monitoring()

    if success:
        # 监控启动成功，禁用离线模式的算法和设备选择
        return "\n".join(list(monitor_logs)), gr.update(interactive=False), gr.update(interactive=True), gr.update(visible=False), gr.update(interactive=False), gr.update(value=algorithm, interactive=False), gr.update(interactive=False)
    else:
        return "\n".join(list(monitor_logs)), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=True), gr.update(), gr.update()


def confirm_detect_existing_fn(monitor):
    """确认检测已有文件"""
    global pending_monitor_params

    if not pending_monitor_params:
        return "\n".join(list(monitor_logs)), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=True), gr.update(), gr.update()

    monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 选择：检测已有文件")

    params = pending_monitor_params
    pending_monitor_params = {}

    result = _do_start_monitoring(
        params['dir_path'],
        params['interval'],
        params['algorithm'],
        params.get('device', 'auto'),
        True,
        monitor
    )
    return result[0], result[1], result[2], result[3], result[4], result[5], result[6]


def confirm_skip_existing_fn(monitor):
    """确认跳过已有文件"""
    global pending_monitor_params

    if not pending_monitor_params:
        return "\n".join(list(monitor_logs)), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(interactive=True), gr.update(), gr.update()

    monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 选择：跳过已有文件")

    params = pending_monitor_params
    pending_monitor_params = {}

    result = _do_start_monitoring(
        params['dir_path'],
        params['interval'],
        params['algorithm'],
        params.get('device', 'auto'),
        False,
        monitor
    )
    return result[0], result[1], result[2], result[3], result[4], result[5], result[6]


def stop_monitoring_fn(monitor):
    """停止监控并释放资源"""
    import torch

    print(f"\n{'='*60}")
    print(f"[在线模式] ⏹️ 用户请求停止监控")
    print(f"{'='*60}")

    monitor.stop_monitoring()

    if monitor.algorithm_manager and monitor.algorithm_manager.is_model_loaded():
        try:
            monitor.algorithm_manager.unload_model()
            torch.cuda.empty_cache()
            monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 模型资源已释放")
        except Exception as e:
            monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 资源释放失败: {e}")

    print(f"{'='*60}\n")
    # 停止监控后，重新启用离线模式的算法和设备选择
    return "\n".join(list(monitor_logs)), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)


def cleanup_temp_files_fn():
    """手动清理所有临时文件"""
    result = cleanup_all_temp_files()

    total_files = sum(r[0] for r in result.values())
    total_dirs = sum(r[1] for r in result.values())
    total_size = sum(r[2] for r in result.values())

    if total_files > 0 or total_dirs > 0:
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 临时文件清理完成: {total_files}个文件, {total_dirs}个目录, 共释放 {total_size/1024/1024:.1f} MB")
    else:
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ 没有需要清理的临时文件")

    return "\n".join(list(monitor_logs))


def export_monitor_zip_fn(monitor):
    """打包下载Excel和所有热力图，返回 (log_messages, file_update)"""
    import gradio as gr

    print(f"\n{'='*60}")
    print(f"[在线模式] 📦 打包下载全部结果...")

    results = list(monitor.detection_results)
    if not results:
        print(f"[在线模式] ⚠ 暂无检测结果可打包")
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 打包失败: 暂无检测结果")
        return "\n".join(list(monitor_logs)), gr.update(visible=False, value=None)

    try:
        zip_path, record_count, image_count = ExportManager.create_monitor_zip(
            results,
            monitor.algorithm_manager.algorithm_chosen,
            "exports"
        )
        print(f"[在线模式] ✓ ZIP包生成: {zip_path}")
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 打包下载成功: {record_count}条记录, {image_count}张图像")
        print(f"{'='*60}\n")

        # 返回文件路径并设置可见
        return "\n".join(list(monitor_logs)), gr.update(visible=True, value=zip_path)
    except Exception as e:
        print(f"[在线模式] ✗ 打包失败: {e}")
        import traceback
        traceback.print_exc()
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 打包失败: {str(e)}")
        return "\n".join(list(monitor_logs)), gr.update(visible=False, value=None)


def refresh_monitor_status(monitor):
    """刷新监控状态"""
    logs_str = "\n".join(list(monitor_logs))
    df = monitor.get_results_df()

    total = monitor.get_total_count()
    anomaly = monitor.get_anomaly_count()
    normal = total - anomaly
    stats = f"总检测: {total} | 异常: {anomaly} | 正常: {normal}"

    mem_stats = get_memory_stats(monitor.device if monitor.is_monitoring else "未运行")

    # 获取最近异常的热力图
    anomaly_images = []
    recent_results = list(monitor.detection_results)[-20:]
    for result in reversed(recent_results):
        if result['is_anomaly'] and result.get('heatmap_path') and os.path.exists(result['heatmap_path']):
            anomaly_images.append((result['heatmap_path'], result['filename']))

    return logs_str, df, anomaly_images, stats, mem_stats


def on_algorithm_change(algorithm: str, monitor):
    """当用户修改算法时，如果正在监控则阻止"""
    if monitor.is_monitoring:
        current_algo = monitor.algorithm_manager.algorithm_chosen
        monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 警告: 监控运行中无法切换算法")
        return "\n".join(list(monitor_logs)), gr.update(value=current_algo), gr.update()

    print(f"[算法同步] 在线模式 -> 离线模式: {algorithm}")
    return gr.update(), gr.update(value=algorithm), gr.update(value=algorithm)


def on_monitor_select_row(evt: gr.SelectData, gallery_data):
    """处理在线模式表格行选择事件"""
    if evt.index is None or not gallery_data:
        return gr.update()

    selected_filename = None
    if isinstance(evt.row_value, (list, tuple)) and len(evt.row_value) >= 2:
        selected_filename = evt.row_value[1]

    if not selected_filename:
        return gr.update()

    for idx, (img_path, caption) in enumerate(gallery_data):
        if caption == selected_filename or selected_filename in caption or caption in selected_filename:
            return gr.update(selected_index=idx)

    return gr.update()


def on_monitor_select_row_with_audio(evt: gr.SelectData):
    """处理在线模式表格行选择事件 - 加载对应的音频文件用于试听"""
    if evt.index is None:
        return gr.update(value=None), "请在上方表格中点击某行来选择要试听的音频"

    selected_filename = None
    if isinstance(evt.row_value, (list, tuple)) and len(evt.row_value) >= 2:
        selected_filename = evt.row_value[1]

    if not selected_filename:
        return gr.update(value=None), "无法获取文件名"

    import os
    import glob

    # 从文件名构建音频文件路径
    # 文件名格式: {name}.png，对应音频文件: slice/audio/{name}.wav
    base_name = os.path.splitext(selected_filename)[0]
    audio_path = os.path.join("slice", "audio", f"{base_name}.wav")

    # 获取异常分数和状态信息
    timestamp = evt.row_value[0] if len(evt.row_value) > 0 else "N/A"
    anomaly_score = evt.row_value[2] if len(evt.row_value) > 2 else "N/A"
    is_anomaly = evt.row_value[3] if len(evt.row_value) > 3 else "N/A"
    status = evt.row_value[4] if len(evt.row_value) > 4 else "N/A"

    if os.path.exists(audio_path):
        info_text = f"时间: {timestamp}\n文件名: {selected_filename}\n异常分数: {anomaly_score}\n是否异常: {is_anomaly}\n状态: {status}"
        return gr.update(value=audio_path), info_text
    else:
        # 尝试其他可能的命名格式
        audio_dir = os.path.join("slice", "audio")
        if os.path.exists(audio_dir):
            # 尝试模糊匹配
            pattern = os.path.join(audio_dir, f"*{base_name}*.wav")
            matching_files = glob.glob(pattern)
            if matching_files:
                info_text = f"时间: {timestamp}\n文件名: {selected_filename}\n异常分数: {anomaly_score}\n是否异常: {is_anomaly}\n状态: {status}"
                return gr.update(value=matching_files[0]), info_text

        return gr.update(value=None), f"未找到对应的音频文件: {audio_path}\n文件名: {selected_filename}"
