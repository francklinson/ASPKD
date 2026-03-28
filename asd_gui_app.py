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

# 添加 Dinomaly 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dinomaly'))

import gradio as gr
import pandas as pd
from config.config_load import yaml_load

# 加载配置文件
param = yaml_load()

from Dinomaly.dinomaly_inference import DinomalyDinoV3Inference, DinomalyDinoV2Inference
from data_prepocessing import Preprocessor


class LogManager:
    """
    日志管理模块
    """

    def __init__(self, max_rows=8):
        self.log_messages = ""
        self.max_rows = max_rows  # 最大行数

    def generate_log(self, msg):
        """
        给日志打时间戳，截断到最大行数
        """
        self.log_messages += f"\n[{time.strftime('%H:%M:%S')}] {msg}"
        # 截断到最大行数
        lines = self.log_messages.split('\n')
        if len(lines) > self.max_rows:
            lines = lines[-self.max_rows:]
        self.log_messages = '\n'.join(lines)
        return

    def update_log(self, msg):
        """
        更新日志
        """
        self.generate_log(msg)
        yield self.log_messages, None, gr.update(interactive=False)

    def update_log_and_action(self, msg, download_output_action, run_button_action):
        """
        更新日志，同时传递GUI控件响应
        """
        self.generate_log(msg)
        yield self.log_messages, download_output_action, run_button_action


class Export:
    """
    文件导出类
    负责将结果excel文件和图像文件打包到zip压缩包中返回给用户
    """

    def __init__(self):
        pass

    @classmethod
    def create_excel_report(cls, results: List[Dict], save_dir) -> str:
        """
        创建Excel报告
        """
        df = pd.DataFrame(results)
        # 只保存文本数据到Excel，不包含图片
        df_for_excel = df[['filename', 'anomaly_score', 'is_anomaly']]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(save_dir, f"anomaly_detection_results_{timestamp}.xlsx")

        df_for_excel.to_excel(excel_path, index=False)

        return excel_path

    @classmethod
    def create_zip_with_results(cls, zip_path: str, excel_path, images: List[str]):
        """
        将Excel和图像打包成ZIP文件
        """
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(excel_path, os.path.basename(excel_path))
            print("将如下图像打包进压缩包:")
            for img in images:
                print(img)
                zipf.write(img, os.path.basename(img))
            print('*' * 50)
            print("打包完成!!!")
        return zip_path


class AlgorithmManager:
    """
    管理异常检测算法模块
    """
    algorithms = [
        "Dinomaly_DINOV3_SMALL",
        # "Dinomaly_DINOV3_BASE",
        "Dinomaly_DINOV3_LARGE",
        "Dinomaly_DINOV2_SMALL",
        # "Dinomaly_DINOV2_BASE",
        # "Dinomaly_DINOV2_LARGE"
    ]

    def __init__(self):
        self.inferencer = None
        self.algorithm_chosen = ""

    def update_algorithm(self, algorithm_choice):
        """
        更新算法
        Returns:

        """
        assert algorithm_choice in self.algorithms
        # 没有变化，不需要重复加载
        if self.inferencer is not None and algorithm_choice == self.algorithm_chosen:
            return
        else:
            # 清理原来的预测器
            if self.inferencer is not None:
                self.inferencer = None
                # 清理cuda缓存，防止爆显存
                self.__clear_cuda_cache()
            if algorithm_choice == "Dinomaly_DINOV3_SMALL":
                self.__load_asd_dinomaly_dinov3("small")
            elif algorithm_choice == "Dinomaly_DINOV3_BASE":
                self.__load_asd_dinomaly_dinov3("base")
            elif algorithm_choice == "Dinomaly_DINOV3_LARGE":
                self.__load_asd_dinomaly_dinov3("large")
            elif algorithm_choice == "Dinomaly_DINOV2_SMALL":
                self.__load_asd_dinomaly_dinov2("small")
            elif algorithm_choice == "Dinomaly_DINOV2_BASE":
                self.__load_asd_dinomaly_dinov2("base")
            elif algorithm_choice == "Dinomaly_DINOV2_LARGE":
                self.__load_asd_dinomaly_dinov2("large")

            self.algorithm_chosen = algorithm_choice
            print(f"异常检测模型成功切换到: {self.algorithm_chosen}")
            return

    def __load_asd_dinomaly_dinov2(self, model_size):
        """
        加载dinomaly dinov2模型
        Args:
            model_size:

        Returns:

        """
        self.inferencer = DinomalyDinoV2Inference(
            model_path=param["model_ckpts"]["dinomaly"]["dinov2"][model_size],
            model_size=model_size,
            threshold=param["model_threshold"]["dinomaly"]["dinov2"][model_size]
        )
        return

    def __load_asd_dinomaly_dinov3(self, model_size):
        """
        加载 dinomaly dinov3 模型
        Args:
            model_size:

        Returns:

        """
        self.inferencer = DinomalyDinoV3Inference(
            model_path=param["model_ckpts"]["dinomaly"]["dinov3"][model_size],
            model_size=model_size,
            threshold=param["model_threshold"]["dinomaly"]["dinov3"][model_size]
        )
        return

    @staticmethod
    def __clear_cuda_cache():
        """
        清除GPU缓存
        Returns:
        """
        torch.cuda.empty_cache()

        # 或者更彻底的清理
        if cuda.is_available():
            cuda.empty_cache()
            cuda.synchronize()
        # 强制垃圾回收
        gc.collect()
        return

    # 算法映射字典


def run_asd_btn_func(audio_files, algorithm_choice: str, progress=gr.Progress()):
    """
    主处理函数
    输入：待处理的音频列表、选择的算法
    返回：处理状态文本，以及返回的结果（压缩包格式）
    """
    # 音频预处理类对象
    # 为了防止多用户使用时的数据污染，放到回调函数里
    p = Preprocessor(ref_file=param["ref_file"])
    lm = LogManager()
    am = AlgorithmManager()

    print("输入参数", audio_files, algorithm_choice, progress)

    if not audio_files:
        yield from lm.update_log_and_action("请上传至少一个音频文件",
                                            None, gr.update(interactive=True))
        return None

    # 只有到实际执行的时候才更新一下算法，避免来回选择算法的时候反复加载模型
    am.update_algorithm(algorithm_choice)
    yield from lm.update_log("执行音频预处理...")
    """
    [1]预处理音频处理成图像
    """
    try:
        picture_file_dict = p.process_audio(audio_files, save_dir=param["pic_output_dir"])
    except Exception as e:
        yield from lm.update_log_and_action(f"音频预处理失败: {str(e)}",
                                            None, gr.update(interactive=True))
        return None

    # 检查返回值是否为空
    if picture_file_dict is None:
        yield from lm.update_log_and_action("音频预处理返回空结果，请检查音频文件格式",
                                            None, gr.update(interactive=True))
        return None

    # return str(picture_file_dict), None
    yield from lm.update_log("完成目标音频搜索和切分！！")
    # print("预处理后的数据:", picture_file_dict)

    """
    [2]调用模型，执行预测
    """
    yield from lm.update_log(f"使用异常检测算法: {am.algorithm_chosen}")
    yield from lm.update_log("执行异常预测...")
    # 处理每个音频文件
    # 拿到的是字典，改成list
    picture_file_list = []

    # 检查 picture_file_dict 是否为字典类型
    if not isinstance(picture_file_dict, dict):
        yield from lm.update_log_and_action(f"预处理结果格式错误，期望dict类型，实际为{type(picture_file_dict)}",
                                            None, gr.update(interactive=True))
        return None

    for k, v in picture_file_dict.items():
        # 检查 v 是否为字典类型
        if not isinstance(v, dict):
            print(f"警告: 文件 {k} 的处理结果格式错误: {type(v)}")
            continue
        if v.get("dk"):
            picture_file_list.append(v["dk"])
        if v.get("qzgy"):
            picture_file_list.append(v["qzgy"])
        if not v.get("dk") and not v.get("qzgy"):
            print(f"No match music sample found in {k}")

    # 确保输入非空
    if not picture_file_list:
        yield from lm.update_log_and_action("上传的素材中没有找到目标音频，请检查!",
                                            None, gr.update(interactive=True))

        return "", None, gr.update(interactive=True)

    # 执行预测
    pred_res_dict, save_img_path_list = am.inferencer.predict(picture_file_list)
    yield from lm.update_log("检测完成!")

    """
    [3]输出结果
    """
    yield from lm.update_log("整理输出结果中...")

    # 结果列表
    results = []

    for k, v in pred_res_dict.items():
        results.append({"filename": k, "anomaly_score": v[0], "is_anomaly": v[1]})

    temp_dir = tempfile.mkdtemp()

    # 创建Excel报告
    excel_path = Export.create_excel_report(results, temp_dir)

    # 创建临时目录和ZIP文件
    zip_path = os.path.join(temp_dir,
                            f'asd_results_{am.algorithm_chosen}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')

    # 打包输出结果
    save_img_path_list.extend(picture_file_list)
    zip_path = Export.create_zip_with_results(zip_path, excel_path, save_img_path_list)
    yield from lm.update_log_and_action(f"处理完成! 共处理 {len(results)} 个文件，请点击下方链接下载结果!", zip_path,
                                        gr.update(interactive=True))
    return "", None, gr.update(interactive=True)


# Gradio界面定义
with gr.Blocks(title="音频异常检测工具") as demo:
    gr.Markdown("# 🎵 音频异常检测工具")
    gr.Markdown("上传WAV格式音频文件进行异常检测，支持批量处理")

    # 参考音频区域
    gr.Markdown("## 📁 参考音频文件")
    gr.Markdown("点击下方链接下载示例音频文件用于测试（渡口+青藏高原片段）")
    gr.Markdown("💡 目前仅适用于使用该标准音频得到的测试音频，其余功能请静候开发!")

    example_audio_file = param["example_audio_file"]
    gr.Audio(
        label="示例音频试听",
        value=example_audio_file,
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
                choices=AlgorithmManager.algorithms,
                value=AlgorithmManager.algorithms[0],
                label="🔧 选择异常检测算法"
            )

            run_button = gr.Button("🚀 开始异常检测", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="📊 处理状态", max_lines=9, interactive=True, autoscroll=True)
            download_output = gr.File(label="📥 下载结果文件", file_count="single")

    run_button.click(
        fn=run_asd_btn_func,
        inputs=[audio_inputs, algorithm_dropdown],
        outputs=[output_text, download_output, run_button],
        queue=True
    )

    gr.Markdown("### 使用说明")
    gr.Markdown("""
    1. 点击"上传音频文件"按钮选择一个或多个WAV格式的音频文件
    2. 从下拉菜单中选择要使用的异常检测算法
    3. 点击"开始异常检测"按钮开始处理
    4. 处理完成后会显示状态信息，并提供结果文件下载
    5. 下载的ZIP文件包含Excel报告和热力图图像
    """)

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name=param["server"]["server_name"],  # 允许外部访问
        server_port=param["server"]["port"],  # 指定端口
        share=param["server"]["share"],  # 不创建公共链接
        inbrowser=param["server"]["inbrowser"]
        # 使用 Gradio 默认主题，不设置 theme 参数
    )
