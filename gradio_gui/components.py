"""
UI组件定义 - 共享的Gradio组件和回调函数
"""
import os
import gradio as gr

# 获取参考音频文件路径
def get_ref_audio_path():
    """获取参考音频文件路径"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "config.yaml")
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        ref_file = config.get('preprocessing', {}).get('ref_file', 'ref/渡口片段10s.wav')
        if not os.path.isabs(ref_file):
            ref_file = os.path.join(project_root, ref_file)
        return ref_file if os.path.exists(ref_file) else None
    except:
        return None


# JavaScript 用于自动滚动监控日志到底部
JS_AUTOSCROLL = """
<script>
(function() {
    function scrollMonitorLogToBottom() {
        const logBox = document.querySelector('#monitor_log_box textarea, #monitor_log_box input');
        if (logBox) {
            logBox.scrollTop = logBox.scrollHeight;
        }
    }

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

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initObserver);
    } else {
        initObserver();
    }

    setInterval(function() {
        if (!observer) {
            initObserver();
        }
        scrollMonitorLogToBottom();
    }, 1000);
})();
</script>
"""


def create_offline_tab_components(model_manager):
    """创建离线模式的UI组件"""
    from .utils import get_available_devices

    gr.Markdown("## 📤 手动上传音频检测")
    gr.Markdown("上传WAV格式音频文件进行批量异常检测")

    # 参考音频区域
    ref_audio_path = get_ref_audio_path()
    with gr.Accordion("📁 参考音频文件（点击展开）", open=False):
        if ref_audio_path:
            gr.Audio(value=ref_audio_path, label="参考音频 (渡口片段10s)", interactive=False)
            gr.File(value=ref_audio_path, label="下载参考音频")
        else:
            gr.Markdown("⚠️ 未找到参考音频文件")
        gr.Markdown("💡 目前仅适用于使用该标准音频得到的测试音频")

    # 主界面
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

            available_devices = get_available_devices()
            device_dropdown = gr.Dropdown(
                choices=[d[0] for d in available_devices],
                value="auto",
                label="🖥️ 选择运行设备",
                info=", ".join([f"{d[0]}={d[1]}" for d in available_devices]),
                allow_custom_value=True
            )

            run_button = gr.Button("🚀 开始异常检测", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="📊 处理状态",
                max_lines=9,
                interactive=True,
                autoscroll=True
            )
            download_output = gr.File(label="📥 下载结果文件", file_count="single")

    # 检测结果展示区域
    results_section = gr.Column(visible=False)
    with results_section:
        gr.Markdown("---")
        gr.Markdown("## 📊 检测结果展示")

        results_table = gr.DataFrame(
            label="检测结果明细",
            headers=["文件名", "异常分数", "是否异常", "状态"],
            interactive=False,
            wrap=True
        )

        heatmap_gallery = gr.Gallery(
            label="全部异常热力图",
            show_label=True,
            elem_id="heatmap_gallery",
            columns=4,
            rows=2,
            height="auto",
            object_fit="contain",
            preview=True
        )

    return {
        'audio_inputs': audio_inputs,
        'algorithm_dropdown': algorithm_dropdown,
        'device_dropdown': device_dropdown,
        'run_button': run_button,
        'output_text': output_text,
        'download_output': download_output,
        'results_section': results_section,
        'results_table': results_table,
        'heatmap_gallery': heatmap_gallery
    }


def create_online_tab_components(model_manager):
    """创建在线模式的UI组件"""
    from .utils import get_available_devices

    gr.Markdown("## 📡 目录实时监控")
    gr.Markdown("监控指定目录下的新增音频文件，自动进行异常检测")

    available_devices = get_available_devices()

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=350):
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

            # 下载组件，导出后显示
            export_download_file = gr.File(label="📥 点击下载结果", visible=False, interactive=True)

            monitor_stats = gr.Textbox(
                label="📊 统计信息",
                value="总检测: 0 | 异常: 0 | 正常: 0",
                interactive=False
            )

            memory_stats = gr.Textbox(
                label="🧠 内存/显存使用",
                value="CPU内存: -- | GPU显存: --",
                interactive=False
            )

        with gr.Column(scale=3, min_width=450):
            monitor_log_box = gr.Textbox(
                label="📝 监控日志",
                lines=8,
                max_lines=12,
                interactive=False,
                autoscroll=True,
                elem_id="monitor_log_box"
            )

            monitor_results_table = gr.DataFrame(
                label="实时检测结果（最近100条）",
                headers=["时间", "文件名", "异常分数", "是否异常", "状态"],
                interactive=False,
                wrap=True
            )

    # 确认对话框
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

    return {
        'monitor_dir_input': monitor_dir_input,
        'monitor_interval': monitor_interval,
        'monitor_algorithm': monitor_algorithm,
        'monitor_device': monitor_device,
        'start_monitor_btn': start_monitor_btn,
        'stop_monitor_btn': stop_monitor_btn,
        'cleanup_temp_btn': cleanup_temp_btn,
        'export_zip_btn': export_zip_btn,
        'export_download_file': export_download_file,
        'monitor_stats': monitor_stats,
        'memory_stats': memory_stats,
        'monitor_log_box': monitor_log_box,
        'monitor_results_table': monitor_results_table,
        'confirm_dialog': confirm_dialog,
        'confirm_count': confirm_count,
        'detect_existing_btn': detect_existing_btn,
        'skip_existing_btn': skip_existing_btn,
        'recent_anomaly_gallery': recent_anomaly_gallery
    }
