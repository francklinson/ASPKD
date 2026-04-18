"""
Gradio应用主入口 - 组装所有组件和事件绑定
"""
import os
import sys
import atexit
import threading
from datetime import datetime

import gradio as gr
import yaml

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import ConfigManager

try:
    # 作为包导入时使用相对导入
    from .managers import UnifiedAlgorithmManager
    from .monitor import DirectoryMonitor
    from .components import JS_AUTOSCROLL, create_offline_tab_components, create_online_tab_components
    from .offline_handlers import run_offline_detection, on_select_row, on_select_row_with_audio
    from .online_handlers import (
        update_monitor_status, start_monitoring_fn, stop_monitoring_fn,
        confirm_detect_existing_fn, confirm_skip_existing_fn,
        cleanup_temp_files_fn, export_monitor_zip_fn, refresh_monitor_status,
        on_algorithm_change, on_monitor_select_row, on_monitor_select_row_with_audio, monitor_logs
    )
except ImportError:
    # 直接运行时使用绝对导入
    from gradio_gui.managers import UnifiedAlgorithmManager
    from gradio_gui.monitor import DirectoryMonitor
    from gradio_gui.components import JS_AUTOSCROLL, create_offline_tab_components, create_online_tab_components
    from gradio_gui.offline_handlers import run_offline_detection, on_select_row, on_select_row_with_audio
    from gradio_gui.online_handlers import (
        update_monitor_status, start_monitoring_fn, stop_monitoring_fn,
        confirm_detect_existing_fn, confirm_skip_existing_fn,
        cleanup_temp_files_fn, export_monitor_zip_fn, refresh_monitor_status,
        on_algorithm_change, on_monitor_select_row, on_monitor_select_row_with_audio, monitor_logs
    )

# 全局实例
model_manager = None
monitor = None


def on_page_load():
    """页面加载时调用"""
    pass


def create_demo():
    """创建Gradio应用"""
    global model_manager, monitor

    # 加载配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "config.yaml")
    config = ConfigManager(config_path)

    # 设置环境变量
    dinomaly_config_path = os.path.join(project_root, "config", "config.yaml")
    if os.path.exists(dinomaly_config_path):
        with open(dinomaly_config_path, 'r', encoding='utf-8') as f:
            dinomaly_config = yaml.safe_load(f) or {}
        env_config = dinomaly_config.get('environments', {})
        for key, value in env_config.items():
            if value:
                os.environ[key] = str(value)

    # 初始化全局管理器
    model_manager = UnifiedAlgorithmManager()
    monitor = DirectoryMonitor(model_manager)
    monitor.set_status_callback(update_monitor_status)

    with gr.Blocks(title="音频异常检测工具") as demo:
        gr.Markdown("# 🎵 音频异常检测工具")
        gr.Markdown("支持离线手动检测和在线实时监测两种模式")
        gr.Markdown("---")

        # ==================== 离线模式 Tab ====================
        with gr.Tab("💻 离线模式", id=0):
            offline_comps = create_offline_tab_components(model_manager)

            # 绑定表格选择事件 - 热力图跳转
            offline_comps['results_table'].select(
                fn=on_select_row,
                inputs=[offline_comps['heatmap_gallery']],
                outputs=[offline_comps['heatmap_gallery']]
            )

            # 绑定表格选择事件 - 音频试听
            offline_comps['results_table'].select(
                fn=on_select_row_with_audio,
                outputs=[offline_comps['audio_preview'], offline_comps['audio_preview_info']]
            )

        # ==================== 在线模式 Tab ====================
        with gr.Tab("📡 在线模式", id=1):
            online_comps = create_online_tab_components(model_manager)

            # 绑定按钮事件
            online_comps['start_monitor_btn'].click(
                fn=lambda dir_path, interval, algorithm, device: start_monitoring_fn(
                    dir_path, interval, algorithm, device, monitor
                ),
                inputs=[
                    online_comps['monitor_dir_input'],
                    online_comps['monitor_interval'],
                    online_comps['monitor_algorithm'],
                    online_comps['monitor_device']
                ],
                outputs=[
                    online_comps['monitor_log_box'],
                    online_comps['start_monitor_btn'],
                    online_comps['stop_monitor_btn'],
                    online_comps['confirm_dialog'],
                    online_comps['confirm_count'],
                    offline_comps['algorithm_dropdown'],
                    offline_comps['device_dropdown']
                ]
            )

            online_comps['detect_existing_btn'].click(
                fn=lambda: confirm_detect_existing_fn(monitor),
                outputs=[
                    online_comps['monitor_log_box'],
                    online_comps['start_monitor_btn'],
                    online_comps['stop_monitor_btn'],
                    online_comps['confirm_dialog'],
                    online_comps['monitor_algorithm'],
                    offline_comps['algorithm_dropdown'],
                    offline_comps['device_dropdown']
                ]
            )

            online_comps['skip_existing_btn'].click(
                fn=lambda: confirm_skip_existing_fn(monitor),
                outputs=[
                    online_comps['monitor_log_box'],
                    online_comps['start_monitor_btn'],
                    online_comps['stop_monitor_btn'],
                    online_comps['confirm_dialog'],
                    online_comps['monitor_algorithm'],
                    offline_comps['algorithm_dropdown'],
                    offline_comps['device_dropdown']
                ]
            )

            online_comps['stop_monitor_btn'].click(
                fn=lambda: stop_monitoring_fn(monitor),
                outputs=[
                    online_comps['monitor_log_box'],
                    online_comps['start_monitor_btn'],
                    online_comps['stop_monitor_btn'],
                    online_comps['monitor_algorithm'],
                    offline_comps['algorithm_dropdown'],
                    offline_comps['device_dropdown']
                ]
            )

            online_comps['cleanup_temp_btn'].click(
                fn=cleanup_temp_files_fn,
                outputs=[online_comps['monitor_log_box']]
            )

            online_comps['export_zip_btn'].click(
                fn=lambda: export_monitor_zip_fn(monitor),
                outputs=[online_comps['monitor_log_box'], online_comps['export_download_file']]
            )

            # 检测间隔动态更新
            def on_interval_change(interval):
                if monitor.is_monitoring:
                    monitor.update_interval(interval)
                return interval

            online_comps['monitor_interval'].change(
                fn=on_interval_change,
                inputs=[online_comps['monitor_interval']],
                outputs=[online_comps['monitor_interval']]
            )

            # 表格选择事件 - 热力图跳转
            online_comps['monitor_results_table'].select(
                fn=on_monitor_select_row,
                inputs=[online_comps['recent_anomaly_gallery']],
                outputs=[online_comps['recent_anomaly_gallery']]
            )

            # 表格选择事件 - 音频试听
            online_comps['monitor_results_table'].select(
                fn=on_monitor_select_row_with_audio,
                outputs=[online_comps['monitor_audio_preview'], online_comps['monitor_audio_preview_info']]
            )

            # 算法切换
            online_comps['monitor_algorithm'].change(
                fn=lambda algo: on_algorithm_change(algo, monitor),
                inputs=[online_comps['monitor_algorithm']],
                outputs=[
                    online_comps['monitor_log_box'],
                    online_comps['monitor_algorithm'],
                    offline_comps['algorithm_dropdown']
                ]
            )

            # 定时刷新状态
            try:
                timer = gr.Timer(value=2, active=True)
                timer.tick(
                    fn=lambda: refresh_monitor_status(monitor),
                    outputs=[
                        online_comps['monitor_log_box'],
                        online_comps['monitor_results_table'],
                        online_comps['recent_anomaly_gallery'],
                        online_comps['monitor_stats'],
                        online_comps['memory_stats']
                    ]
                )
            except:
                gr.Markdown("---")
                refresh_btn = gr.Button("🔄 手动刷新状态")
                refresh_btn.click(
                    fn=lambda: refresh_monitor_status(monitor),
                    outputs=[
                        online_comps['monitor_log_box'],
                        online_comps['monitor_results_table'],
                        online_comps['recent_anomaly_gallery'],
                        online_comps['monitor_stats'],
                        online_comps['memory_stats']
                    ]
                )

        # ==================== 统一事件绑定 ====================
        # 离线模式按钮事件
        def offline_detection_wrapper(audio_files, algorithm, device):
            """包装离线检测函数，传递model_manager"""
            yield from run_offline_detection(audio_files, algorithm, device, model_manager)

        offline_comps['run_button'].click(
            fn=offline_detection_wrapper,
            inputs=[
                offline_comps['audio_inputs'],
                offline_comps['algorithm_dropdown'],
                offline_comps['device_dropdown']
            ],
            outputs=[
                offline_comps['output_text'],
                offline_comps['results_section'],
                offline_comps['results_table'],
                offline_comps['heatmap_gallery'],
                offline_comps['download_output'],
                offline_comps['run_button'],
                online_comps['monitor_algorithm']
            ]
        )

        # 算法选择同步：离线模式 -> 在线模式
        offline_comps['algorithm_dropdown'].change(
            fn=lambda algo: gr.update(value=algo),
            inputs=[offline_comps['algorithm_dropdown']],
            outputs=[online_comps['monitor_algorithm']]
        )

        # 页面加载事件
        demo.load(fn=on_page_load, inputs=None, outputs=None)

        # 页面加载时同步监控状态
        def get_initial_monitor_state():
            if monitor.is_monitoring:
                monitor_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] 页面已刷新，监控正在运行中...")
                return (
                    monitor.monitor_dir or "",
                    monitor.interval,
                    gr.update(value=monitor.algorithm_manager.algorithm_chosen, interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    "\n".join(list(monitor_logs)),
                    gr.update(value=monitor.algorithm_manager.algorithm_chosen, interactive=False),
                    gr.update(interactive=False)
                )
            else:
                return (
                    "",
                    30,
                    gr.update(
                        value=model_manager.available_algorithms[0] if model_manager.available_algorithms else None,
                        interactive=True
                    ),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    "准备就绪" if not monitor_logs else "\n".join(list(monitor_logs)),
                    gr.update(
                        value=model_manager.available_algorithms[0] if model_manager.available_algorithms else None,
                        interactive=True
                    ),
                    gr.update(interactive=True)
                )

        demo.load(
            fn=get_initial_monitor_state,
            outputs=[
                online_comps['monitor_dir_input'],
                online_comps['monitor_interval'],
                online_comps['monitor_algorithm'],
                online_comps['start_monitor_btn'],
                online_comps['stop_monitor_btn'],
                online_comps['monitor_log_box'],
                offline_comps['algorithm_dropdown'],
                offline_comps['device_dropdown']
            ]
        )

    return demo, config, JS_AUTOSCROLL


def cleanup_resources():
    """应用退出时清理资源"""
    import torch

    print(f"\n{'='*60}")
    print(f"[全局清理] 🧹 应用退出，执行全局资源清理...")
    print(f"{'='*60}")

    global monitor
    if monitor:
        monitor.cleanup()
    else:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
        import gc
        gc.collect()

    print(f"[全局清理] ✓ 全局资源清理完成")
    print(f"{'='*60}\n")


# 注册退出清理
atexit.register(cleanup_resources)
