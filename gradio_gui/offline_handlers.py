"""
离线模式处理函数 - 批量音频检测逻辑
"""
import os
import sys
import time
from datetime import datetime
from typing import List, Dict

import gradio as gr
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core import ConfigManager
from preprocessing import Preprocessor

try:
    from .export import ExportManager
except ImportError:
    from gradio_gui.export import ExportManager


class LogManager:
    """日志管理模块"""

    def __init__(self, max_rows=8):
        self.log_messages = ""
        self.max_rows = max_rows

    def generate_log(self, msg: str):
        """给日志打时间戳，截断到最大行数"""
        self.log_messages += f"\n[{time.strftime('%H:%M:%S')}] {msg}"
        lines = self.log_messages.split('\n')
        if len(lines) > self.max_rows:
            lines = lines[-self.max_rows:]
        self.log_messages = '\n'.join(lines)

    def update_log(self, msg: str):
        """更新日志（生成器）"""
        self.generate_log(msg)
        yield self.log_messages, gr.update(visible=False), None, None, None, gr.update(interactive=False), gr.update()

    def update_log_and_action(self, msg: str, run_button_action):
        """更新日志并返回控件状态"""
        self.generate_log(msg)
        yield self.log_messages, gr.update(visible=False), None, None, None, run_button_action, gr.update()


def run_offline_detection(audio_files, algorithm_choice: str, device_choice: str,
                         model_manager, progress=gr.Progress()):
    """
    离线模式主处理函数 - 批量检测
    """
    print("\n" + "=" * 60)
    print("[离线模式] 开始批量异常检测")
    print(f"[离线模式] 选择设备: {device_choice}")
    print("=" * 60)

    # 加载配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config", "config.yaml")
    config = ConfigManager(config_path)

    ref_file = config.config.get('preprocessing', {}).get('ref_file', 'ref/渡口片段10s.wav')
    if not os.path.isabs(ref_file):
        ref_file = os.path.join(project_root, ref_file)

    split_method = config.config.get('preprocessing', {}).get('split_method', 'mfcc_dtw')
    shazam_config = config.config.get('preprocessing', {}).get('shazam', {})

    print(f"[离线模式] 上传文件数量: {len(audio_files) if audio_files else 0}")
    print(f"[离线模式] 选择算法: {algorithm_choice}")

    # 初始化组件
    preprocessor = Preprocessor(
        ref_file=ref_file,
        split_method=split_method,
        shazam_threshold=shazam_config.get('threshold', 10),
        shazam_auto_match=shazam_config.get('auto_match', False),
        max_workers=shazam_config.get('max_workers', 1)
    )
    log_mgr = LogManager()
    model_loaded = False

    # 保存原始文件名映射（Gradio临时文件 -> 原始文件名）
    original_names = {}

    try:
        if not audio_files:
            print("[离线模式] ✗ 错误: 未上传音频文件")
            yield from log_mgr.update_log_and_action("请上传至少一个音频文件", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()

        # 提取原始文件名
        for file_obj in audio_files:
            if hasattr(file_obj, 'name'):
                # Gradio文件对象
                temp_path = file_obj.name
                original_name = getattr(file_obj, 'orig_name', None) or os.path.basename(temp_path)
                original_names[temp_path] = original_name
                print(f"[离线模式] 文件映射: {os.path.basename(temp_path)} -> {original_name}")
            else:
                # 字符串路径（直接传入的文件路径）
                original_names[file_obj] = os.path.basename(file_obj)

        # 加载算法
        print(f"\n[离线模式] 步骤1/4: 加载算法模型...")
        try:
            model_manager.update_algorithm(algorithm_choice, device=device_choice)
            model_manager.load_model()
            model_loaded = True
            print(f"[离线模式] ✓ 算法模型加载成功")
        except Exception as e:
            print(f"[离线模式] ✗ 算法加载失败: {e}")
            yield from log_mgr.update_log_and_action(f"算法加载失败: {str(e)}", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()

        yield from log_mgr.update_log("执行音频预处理...")

        # 音频预处理
        print(f"\n[离线模式] 步骤2/4: 音频预处理...")
        try:
            picture_file_dict = preprocessor.process_audio(audio_files, save_dir="slice", original_names=original_names)
            print(f"[离线模式] ✓ 预处理完成，生成 {len(picture_file_dict)} 个音频片段")
        except Exception as e:
            print(f"[离线模式] ✗ 音频预处理失败: {e}")
            yield from log_mgr.update_log_and_action(f"音频预处理失败: {str(e)}", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()

        if not picture_file_dict:
            print("[离线模式] ✗ 预处理返回空结果")
            yield from log_mgr.update_log_and_action("音频预处理返回空结果", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()

        yield from log_mgr.update_log("完成目标音频搜索和切分！！")

        # 准备图像列表
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
            yield from log_mgr.update_log_and_action("上传的素材中没有找到目标音频!", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()

        # 执行异常检测
        print(f"\n[离线模式] 步骤4/4: 执行异常检测...")
        yield from log_mgr.update_log(f"使用算法: {model_manager.algorithm_chosen}")
        yield from log_mgr.update_log("执行异常预测...")

        try:
            detection_results = model_manager.detector.predict_batch(picture_file_list)
            print(f"[离线模式] ✓ 检测完成，返回 {len(detection_results)} 个结果")

            # 构建结果
            pred_res_dict = {}
            heatmap_paths_dict = {}
            for img_path, result in zip(picture_file_list, detection_results):
                filename = os.path.basename(img_path)
                pred_res_dict[filename] = (result.anomaly_score, 1 if result.is_anomaly else 0)
                heatmap_path = result.metadata.get('heatmap_path') if result.metadata else None
                if heatmap_path:
                    heatmap_paths_dict[filename] = heatmap_path

            yield from log_mgr.update_log("检测完成!")
        except Exception as e:
            yield from log_mgr.update_log_and_action(f"检测失败: {str(e)}", gr.update(interactive=True))
            return None, gr.update(visible=False), None, None, None, gr.update(interactive=True), gr.update()

        # 输出结果
        print(f"\n[离线模式] 整理输出结果...")
        yield from log_mgr.update_log("整理输出结果中...")

        results = [{"filename": k, "anomaly_score": v[0], "is_anomaly": v[1]} for k, v in pred_res_dict.items()]

        df_results = pd.DataFrame(results)
        df_results['状态'] = df_results['is_anomaly'].apply(lambda x: '异常' if x == 1 else '正常')
        df_results = df_results[['filename', 'anomaly_score', 'is_anomaly', '状态']]
        df_results.columns = ['文件名', '异常分数', '是否异常', '状态']

        # 导出文件
        exports_dir = "exports"
        os.makedirs(exports_dir, exist_ok=True)

        excel_path = ExportManager.create_excel_report(results, exports_dir)
        print(f"[离线模式] ✓ Excel报告生成: {excel_path}")

        zip_path = os.path.join(exports_dir, f'asd_results_{model_manager.algorithm_chosen}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')
        ExportManager.create_zip_with_results(zip_path, excel_path, picture_file_list)
        print(f"[离线模式] ✓ ZIP包生成: {zip_path}")

        # 为Gallery准备数据
        gallery_data = []
        for img_path in picture_file_list:
            filename = os.path.basename(img_path)
            heatmap_path = heatmap_paths_dict.get(filename)
            caption = os.path.splitext(filename)[0]

            if heatmap_path and os.path.exists(heatmap_path):
                gallery_data.append((heatmap_path, caption))
            elif os.path.exists(img_path):
                gallery_data.append((img_path, caption))

        print(f"\n[离线模式] >>> 处理完成! 共处理 {len(results)} 个文件")
        print("=" * 60)

        lm = log_mgr
        lm.generate_log(f"处理完成! 共处理 {len(results)} 个文件")

        yield lm.log_messages, gr.update(visible=True), df_results, gallery_data, zip_path, gr.update(interactive=True), gr.update(value=algorithm_choice)

    finally:
        if model_loaded:
            print(f"\n[离线模式] 检测流程完成，模型保持加载")


def on_select_row(evt: gr.SelectData, gallery_data):
    """处理表格行选择事件 - 跳转到对应热力图"""
    if evt.index is None or not gallery_data:
        return gr.update()

    selected_filename = evt.row_value[0] if isinstance(evt.row_value, (list, tuple)) else None
    if not selected_filename:
        return gr.update()

    for idx, (img_path, caption) in enumerate(gallery_data):
        if caption == selected_filename or selected_filename in caption or caption in selected_filename:
            return gr.update(selected_index=idx)

    return gr.update()


def on_select_row_with_audio(evt: gr.SelectData):
    """处理表格行选择事件 - 加载对应的音频文件用于试听"""
    if evt.index is None:
        return gr.update(value=None), "请在左侧表格中点击某行来选择要试听的音频"

    selected_filename = evt.row_value[0] if isinstance(evt.row_value, (list, tuple)) else None
    if not selected_filename:
        return gr.update(value=None), "无法获取文件名"

    # 从文件名构建音频文件路径
    # 文件名格式: {name}.png，对应音频文件: slice/audio/{name}.wav
    base_name = os.path.splitext(selected_filename)[0]
    audio_path = os.path.join("slice", "audio", f"{base_name}.wav")

    # 获取异常分数和状态信息
    anomaly_score = evt.row_value[1] if len(evt.row_value) > 1 else "N/A"
    is_anomaly = evt.row_value[2] if len(evt.row_value) > 2 else "N/A"
    status = evt.row_value[3] if len(evt.row_value) > 3 else "N/A"

    if os.path.exists(audio_path):
        info_text = f"文件名: {selected_filename}\n异常分数: {anomaly_score}\n是否异常: {'是' if is_anomaly == 1 else '否'}\n状态: {status}"
        return gr.update(value=audio_path), info_text
    else:
        # 尝试其他可能的命名格式
        # 有些文件名可能包含原始音频名和位置信息
        import glob
        audio_dir = os.path.join("slice", "audio")
        if os.path.exists(audio_dir):
            # 尝试模糊匹配
            pattern = os.path.join(audio_dir, f"*{base_name}*.wav")
            matching_files = glob.glob(pattern)
            if matching_files:
                info_text = f"文件名: {selected_filename}\n异常分数: {anomaly_score}\n是否异常: {'是' if is_anomaly == 1 else '否'}\n状态: {status}"
                return gr.update(value=matching_files[0]), info_text

        return gr.update(value=None), f"未找到对应的音频文件: {audio_path}\n文件名: {selected_filename}"
