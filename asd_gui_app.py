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
from datetime import datetime
from typing import List, Dict

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

# Gradio界面定义
with gr.Blocks(title="音频异常检测工具") as demo:
    gr.Markdown("# 🎵 音频异常检测工具")
    gr.Markdown("上传WAV格式音频文件进行异常检测，支持批量处理")

    # 参考音频区域
    gr.Markdown("## 📁 参考音频文件")
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

    gr.Markdown("---")

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

if __name__ == "__main__":
    # 使用绝对路径加载配置
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "config", "algorithms.yaml")
    config = ConfigManager(config_path)
    
    server_config = config.config.get('server', {})
    demo.launch(
        server_name=server_config.get('server_name', '0.0.0.0'),
        server_port=server_config.get('port', 8002),
        share=server_config.get('share', False),
        inbrowser=server_config.get('inbrowser', True)
    )
