"""
音频异常检测 Web 应用
使用 Gradio 构建，支持上传音频、选择特征提取器、交互式3D聚类可视化
"""
import os
import sys
import tempfile
import shutil
import numpy as np
import soundfile as sf
import librosa
import plotly.graph_objects as go
from pathlib import Path
from typing import Tuple, List, Optional

# 导入核心模块
sys.path.insert(0, str(Path(__file__).parent))
from audio_anomaly_detection import (
    Config, FeatureExtractorFactory, AnomalyDetector
)

# Gradio
import gradio as gr

# 支持的特征提取器
EXTRACTORS = {
    "MFCC (传统方法，速度快)": "mfcc",
    "Mel Spectrogram (传统方法)": "mel",
    "HuBERT (深度学习)": "hubert",
    "WavLM (深度学习)": "wavlm",
    "XLSR-Wav2Vec2 (多语言)": "xlsr-wav2vec2",
    "AST (Transformer)": "ast",
    "MERT (音乐理解)": "mert",
}

# 支持的音频格式（后缀）
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus"}


def validate_and_preprocess_audio(file_path: str, target_dir: str) -> Tuple[str, dict]:
    """
    校验音频文件格式并统一预处理为标准格式（16kHz, 单声道, WAV）。

    Args:
        file_path: 原始音频文件路径
        target_dir: 预处理后文件的存放目录

    Returns:
        (预处理后的文件路径, 转换信息字典)

    Raises:
        ValueError: 文件格式不支持或无法解码
    """
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTS:
        raise ValueError(f"不支持的格式 '{ext}'，仅支持: {', '.join(sorted(SUPPORTED_AUDIO_EXTS))}")

    info = {"original_file": Path(file_path).name, "original_ext": ext}

    try:
        # librosa.load 自动处理重采样和混音，兼容所有主流格式
        # 不在此处指定 sr，保留原始采样率信息用于日志记录
        waveform, orig_sr = librosa.load(file_path, sr=None, mono=True)
    except Exception as e:
        raise ValueError(f"无法解码音频文件: {e}")

    info["original_sr"] = orig_sr
    info["duration"] = round(len(waveform) / orig_sr, 2)
    info["channels"] = "mono (自动混音)"

    # 统一保存为 16kHz 单声道 WAV（后续各提取器的 librosa.load 会再次按需重采样）
    dst_name = Path(file_path).stem + ".wav"
    dst_path = os.path.join(target_dir, dst_name)

    # 重采样到 16kHz
    if orig_sr != 16000:
        waveform_16k = librosa.resample(waveform, orig_sr=orig_sr, target_sr=16000)
        info["resampled_to"] = 16000
    else:
        waveform_16k = waveform

    sf.write(dst_path, waveform_16k, 16000, subtype="PCM_16")
    info["output_format"] = "WAV (16kHz, mono, 16bit PCM)"
    info["output_file"] = dst_name

    return dst_path, info


def process_audio_files(
        audio_files: list,
        extractor_choice: str,
        n_clusters: int,
        anomaly_threshold: int,
        use_3d: bool,
        tsne_perplexity: int,
):
    """
    处理上传的音频文件，生成聚类可视化（生成器模式，逐步输出日志）

    Args:
        audio_files: 上传的音频文件列表
        extractor_choice: 特征提取器选项
        n_clusters: 聚类数量
        anomaly_threshold: 异常阈值百分位数
        use_3d: 是否使用3D可视化
        tsne_perplexity: t-SNE perplexity

    Yields:
        (plotly_figure, 报告文本, 日志文本) 每处理一个阶段 yield 一次
    """
    def log(msg: str):
        nonlocal logs
        import time
        logs += f"[{time.strftime('%H:%M:%S')}] {msg}\n"

    logs = ""

    if not audio_files:
        log("⚠️ 未检测到上传文件")
        yield None, "请先上传音频文件！", logs
        return

    # 创建临时目录存放上传的音频
    temp_dir = tempfile.mkdtemp(prefix="audio_anomaly_")
    preproc_dir = tempfile.mkdtemp(prefix="audio_preproc_")
    saved_paths = []
    preprocess_info_list = []
    skipped_files = []

    try:
        log(f"📁 接收到 {len(audio_files)} 个文件")
        yield None, "⏳ 正在预处理音频...", logs

        # 保存上传的音频文件，并进行格式校验和预处理
        for i, file in enumerate(audio_files):
            if hasattr(file, 'name'):
                src = file.name
            else:
                src = file

            fname = os.path.basename(src)
            try:
                preproc_path, info = validate_and_preprocess_audio(src, preproc_dir)
                saved_paths.append(preproc_path)
                preprocess_info_list.append(info)
                log(f"  ✓ [{i+1}/{len(audio_files)}] {fname} -> {info['duration']}s, {info.get('output_format', 'WAV')}")
            except ValueError as e:
                skipped_files.append(f"  - {fname}: {e}")
                log(f"  ✗ [{i+1}/{len(audio_files)}] {fname}: {e}")

            # 每处理5个文件或最后一个文件时更新一次日志
            if (i + 1) % 5 == 0 or (i + 1) == len(audio_files):
                yield None, f"⏳ 预处理中... ({i+1}/{len(audio_files)})", logs

        if not saved_paths:
            log("❌ 没有有效的音频文件")
            yield None, "没有有效的音频文件！错误详情：\n" + "\n".join(skipped_files), logs
            return

        log(f"✅ 预处理完成：{len(saved_paths)} 个有效文件" +
            (f"，跳过 {len(skipped_files)} 个" if skipped_files else ""))

        # 获取特征提取器类型
        extractor_type = EXTRACTORS.get(extractor_choice, "mfcc")
        log(f"\n🔧 特征提取器: {extractor_choice} ({extractor_type})")
        log(f"⚙️ 参数: 聚类数={n_clusters}, 异常阈值={anomaly_threshold}%, perplexity={tsne_perplexity}")
        yield None, "⏳ 正在加载特征提取模型...", logs

        # 创建配置
        config = Config(feature_extractor_type=extractor_type)
        config.n_clusters = n_clusters
        config.outlier_threshold_percentile = anomaly_threshold
        config.tsne_perplexity = tsne_perplexity

        # 创建特征提取器
        import time as _time
        t0 = _time.time()
        feature_extractor = FeatureExtractorFactory.create_extractor(config)
        log(f"  ✓ 模型加载完成 ({_time.time()-t0:.1f}s)")

        # 提取特征
        log(f"\n📊 开始提取特征...")
        yield None, "⏳ 正在提取音频特征...", logs

        all_features = []
        file_labels = []

        for i, path in enumerate(saved_paths):
            label = Path(path).stem
            try:
                t0 = _time.time()
                features = feature_extractor.extract_features(path)
                dt = _time.time() - t0
                if len(features) > 0:
                    all_features.append(features)
                    file_labels.extend([label] * len(features))
                    log(f"  ✓ [{i+1}/{len(saved_paths)}] {label}: {len(features)} 段特征, dim={features.shape[1]}, 耗时 {dt:.2f}s")
                else:
                    log(f"  ⚠ [{i+1}/{len(saved_paths)}] {label}: 未提取到特征")
            except Exception as e:
                log(f"  ✗ [{i+1}/{len(saved_paths)}] {label}: {e}")

            # 每处理5个文件或最后一个文件时更新一次日志
            if (i + 1) % 5 == 0 or (i + 1) == len(saved_paths):
                yield None, f"⏳ 特征提取中... ({i+1}/{len(saved_paths)})", logs

        if not all_features:
            log("❌ 未能从任何音频中提取特征")
            yield None, "未能从上传的音频中提取任何特征。请检查音频文件。", logs
            return

        X = np.vstack(all_features)
        log(f"\n✅ 特征提取完成: {X.shape[0]} 个特征向量, 维度={X.shape[1]}")

        # 异常检测
        from sklearn.preprocessing import StandardScaler
        t0 = _time.time()

        if n_clusters == 1:
            # 单说话人模式：使用 Isolation Forest
            log("\n🔍 单说话人模式 - 使用 Isolation Forest...")
            yield None, "⏳ 正在进行异常检测 (Isolation Forest)...", logs

            contamination = 1.0 - anomaly_threshold / 100.0
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            outlier_mask = iso_forest.fit_predict(X) == -1
            cluster_labels = np.zeros(X.shape[0], dtype=int)

            log(f"  ✓ Isolation Forest 完成 ({_time.time()-t0:.2f}s), contamination={contamination:.2f}")
            log(f"  检测到 {int(np.sum(outlier_mask))} 个异常样本")
        else:
            # 样本数不足时自动降级：聚类数不能超过样本数
            actual_n_clusters = min(n_clusters, X.shape[0])
            if actual_n_clusters < n_clusters:
                log(f"  ⚠️ 样本数 ({X.shape[0]}) 少于聚类数 ({n_clusters})，自动调整为 {actual_n_clusters}")
                n_clusters = actual_n_clusters

            # 多说话人模式：使用 K-Means + 距离阈值
            log(f"\n🔍 多说话人模式 - K-Means 聚类 (K={n_clusters})...")
            yield None, "⏳ 正在聚类分析...", logs

            config.n_clusters = n_clusters
            detector = AnomalyDetector(config)
            detector.fit(X)
            cluster_labels, outlier_mask = detector.predict(X)
            log(f"  ✓ 聚类完成 ({_time.time()-t0:.2f}s)")

            unique, counts = np.unique(cluster_labels, return_counts=True)
            for u, c in zip(unique, counts):
                log(f"    Cluster {int(u)+1}: {int(c)} 样本")
            log(f"  检测到 {int(np.sum(outlier_mask))} 个异常样本")

        # 生成可视化
        log(f"\n📈 降维可视化...")
        yield None, "⏳ 正在生成可视化...", logs

        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        t0 = _time.time()
        n_components = 3 if use_3d else 2

        if X.shape[0] <= n_components:
            # 样本数不足以运行 t-SNE/PCA，用零坐标占位
            log(f"  ⚠️ 样本数 ({X.shape[0]}) 过少，跳过降维")
            X_embedded = np.zeros((X.shape[0], n_components))
        else:
            # 使用PCA降维，再用t-SNE降维（加速计算）
            if X.shape[1] > 50:
                pca_n = min(50, X.shape[0] - 1, X.shape[1])
                pca = PCA(n_components=pca_n, random_state=42)
                X_pca = pca.fit_transform(X)
                log(f"  PCA: {X.shape[1]}D -> {pca_n}D")
            else:
                X_pca = X

            # t-SNE降维
            perplexity = min(tsne_perplexity, X.shape[0] - 1)
            perplexity = max(perplexity, 1)
            tsne = TSNE(n_components=n_components, random_state=42,
                        perplexity=perplexity,
                        max_iter=config.tsne_max_iter)
            X_embedded = tsne.fit_transform(X_pca)
        log(f"  t-SNE: -> {n_components}D ({_time.time()-t0:.1f}s)")

        # 生成plotly图形
        log("  生成交互式图表...")
        fig = create_cluster_plot(X_embedded, cluster_labels, outlier_mask,
                                  file_labels, extractor_choice, use_3d,
                                  n_clusters=n_clusters)

        # 生成报告
        report = generate_report(X, cluster_labels, outlier_mask, file_labels,
                                 extractor_choice, n_clusters, anomaly_threshold,
                                 preprocess_info_list, skipped_files)

        log(f"\n✅ 分析完成！")
        yield fig, report, logs

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(preproc_dir, ignore_errors=True)


def create_cluster_plot(
        X_embedded: np.ndarray,
        cluster_labels: np.ndarray,
        outlier_mask: np.ndarray,
        file_labels: List[str],
        extractor_choice: str,
        use_3d: bool,
        n_clusters: int = None
) -> go.Figure:
    """创建交互式聚类图"""
    if n_clusters is None:
        n_clusters = len(set(cluster_labels))

    # 颜色方案 - 高对比度
    colors = [
        '#E63946',  # 正红
        '#1D3557',  # 深蓝
        '#F4A261',  # 橙
        '#2A9D8F',  # 青绿
        '#7209B7',  # 紫
        '#06D6A0',  # 亮绿
        '#FF006E',  # 品红
        '#3A86FF',  # 亮蓝
        '#FFBE0B',  # 金黄
        '#8338EC',  # 蓝紫
    ]

    normal_mask = ~outlier_mask
    title_suffix = " (单说话人 - Isolation Forest)" if n_clusters == 1 else f" (K={n_clusters})"

    # 计算每个聚类在嵌入空间中的中心点（K>=2 时绘制）
    center_coords = None
    if n_clusters >= 2:
        _centers = []
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            _centers.append(X_embedded[mask].mean(axis=0))
        center_coords = np.array(_centers)

    if use_3d:
        fig = go.Figure()

        if n_clusters == 1:
            # 单说话人：所有正常点同色，异常点菱形
            fig.add_trace(go.Scatter3d(
                x=X_embedded[normal_mask, 0],
                y=X_embedded[normal_mask, 1],
                z=X_embedded[normal_mask, 2],
                mode='markers',
                name='Normal',
                marker=dict(size=5, color='#3A86FF', opacity=0.7,
                            line=dict(width=0.5, color='white')),
                hovertemplate='%{text}<extra>Normal</extra>',
                text=[file_labels[i] for i in range(len(normal_mask)) if normal_mask[i]]
            ))
        else:
            for cluster_id in range(n_clusters):
                mask = (cluster_labels == cluster_id) & normal_mask
                color = colors[cluster_id % len(colors)]
                fig.add_trace(go.Scatter3d(
                    x=X_embedded[mask, 0],
                    y=X_embedded[mask, 1],
                    z=X_embedded[mask, 2],
                    mode='markers',
                    name=f'Cluster {cluster_id + 1}',
                    marker=dict(size=5, color=color, opacity=0.7,
                                line=dict(width=0.5, color='white')),
                    hovertemplate='%{text}<extra>Cluster %{data.name}</extra>',
                    text=[file_labels[i] for i in range(len(mask)) if mask[i]]
                ))

        # 绘制异常点
        if np.any(outlier_mask):
            fig.add_trace(go.Scatter3d(
                x=X_embedded[outlier_mask, 0],
                y=X_embedded[outlier_mask, 1],
                z=X_embedded[outlier_mask, 2],
                mode='markers',
                name='Anomalies',
                marker=dict(size=8, color='black', symbol='diamond',
                            line=dict(width=2, color='red')),
                hovertemplate='%{text}<extra>Anomaly!</extra>',
                text=[file_labels[i] for i in range(len(outlier_mask)) if outlier_mask[i]]
            ))

        # 绘制聚类中心（仅 K>=2）
        if center_coords is not None:
            fig.add_trace(go.Scatter3d(
                x=center_coords[:, 0], y=center_coords[:, 1], z=center_coords[:, 2],
                mode='markers+text', name='Cluster Centers',
                marker=dict(size=15, color='#FFD700', symbol='cross',
                            line=dict(width=2, color='#B8860B')),
                text=[f'C{i+1}' for i in range(n_clusters)],
                textposition='top center', textfont=dict(size=12, color='#B8860B'),
                hovertemplate='%{text} (Center)<extra>Cluster Center</extra>'
            ))

        fig.update_layout(
            scene=dict(xaxis_title='t-SNE 1', yaxis_title='t-SNE 2',
                       zaxis_title='t-SNE 3', bgcolor='white'),
            title=f'Audio Clustering Analysis - {extractor_choice}{title_suffix}',
            template='plotly_white', height=850, hovermode='closest'
        )
    else:
        fig = go.Figure()

        if n_clusters == 1:
            fig.add_trace(go.Scatter(
                x=X_embedded[normal_mask, 0], y=X_embedded[normal_mask, 1],
                mode='markers', name='Normal',
                marker=dict(size=8, color='#3A86FF', opacity=0.7,
                            line=dict(width=0.5, color='white')),
                hovertemplate='%{text}<extra>Normal</extra>',
                text=[file_labels[i] for i in range(len(normal_mask)) if normal_mask[i]]
            ))
        else:
            for cluster_id in range(n_clusters):
                mask = (cluster_labels == cluster_id) & normal_mask
                color = colors[cluster_id % len(colors)]
                fig.add_trace(go.Scatter(
                    x=X_embedded[mask, 0], y=X_embedded[mask, 1],
                    mode='markers', name=f'Cluster {cluster_id + 1}',
                    marker=dict(size=8, color=color, opacity=0.7,
                                line=dict(width=0.5, color='white')),
                    hovertemplate='%{text}<extra>Cluster %{data.name}</extra>',
                    text=[file_labels[i] for i in range(len(mask)) if mask[i]]
                ))

        # 绘制异常点
        if np.any(outlier_mask):
            fig.add_trace(go.Scatter(
                x=X_embedded[outlier_mask, 0], y=X_embedded[outlier_mask, 1],
                mode='markers', name='Anomalies',
                marker=dict(size=12, color='black', symbol='diamond',
                            line=dict(width=2, color='red')),
                hovertemplate='%{text}<extra>Anomaly!</extra>',
                text=[file_labels[i] for i in range(len(outlier_mask)) if outlier_mask[i]]
            ))

        # 绘制聚类中心（仅 K>=2）
        if center_coords is not None:
            fig.add_trace(go.Scatter(
                x=center_coords[:, 0], y=center_coords[:, 1],
                mode='markers+text', name='Cluster Centers',
                marker=dict(size=20, color='#FFD700', symbol='star',
                            line=dict(width=2, color='#B8860B')),
                text=[f'C{i+1}' for i in range(n_clusters)],
                textposition='top center', textfont=dict(size=12, color='#B8860B'),
                hovertemplate='%{text} (Center)<extra>Cluster Center</extra>'
            ))

        fig.update_layout(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            title=f'Audio Clustering Analysis - {extractor_choice}',
            template='plotly_white',
            height=750,
            hovermode='closest'
        )

    return fig


def generate_report(
        X: np.ndarray,
        cluster_labels: np.ndarray,
        outlier_mask: np.ndarray,
        file_labels: List[str],
        extractor_name: str,
        n_clusters: int,
        threshold: int,
        preprocess_info: List[dict] = None,
        skipped_files: List[str] = None,
) -> str:
    """生成分析报告"""
    n_samples = len(X)
    n_outliers = int(np.sum(outlier_mask))
    n_normal = n_samples - n_outliers

    # 检测方法
    method = "Isolation Forest (单说话人模式)" if n_clusters == 1 else f"K-Means (K={n_clusters})"

    # 聚类分布
    if n_clusters == 1:
        cluster_dist = f"  - 单一类: {n_normal} normal samples"
    else:
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_dist = "\n".join([f"  - Cluster {int(u) + 1}: {int(c)} samples" for u, c in zip(unique, counts)])

    # 异常文件
    anomaly_files = np.array(file_labels)[outlier_mask]
    unique_anomalies, anomaly_counts = np.unique(anomaly_files, return_counts=True)
    anomaly_list = "\n".join([f"  - {f}: {c} segments" for f, c in zip(unique_anomalies, anomaly_counts)])

    # 音频预处理摘要
    preproc_section = ""
    if preprocess_info:
        total_duration = sum(p["duration"] for p in preprocess_info)
        resampled_count = sum(1 for p in preprocess_info if "resampled_to" in p)
        converted_count = sum(1 for p in preprocess_info if p["original_ext"] != ".wav")
        preproc_section = f"""
## 音频预处理
- **有效文件**: {len(preprocess_info)} 个
- **总时长**: {total_duration:.1f} 秒
- **需重采样**: {resampled_count} 个
- **需格式转换**: {converted_count} 个
"""
        if converted_count > 0:
            converted_files = [p["original_file"] for p in preprocess_info if p["original_ext"] != ".wav"]
            if len(converted_files) <= 5:
                preproc_section += "- **转换详情**: " + ", ".join(converted_files) + "\n"
            else:
                preproc_section += f"- **转换详情**: {converted_files[0]} 等 {len(converted_files)} 个文件\n"

    # 跳过的文件
    skip_section = ""
    if skipped_files:
        skip_section = f"\n## 跳过的文件\n" + "\n".join(skipped_files) + "\n"

    report = f"""# 音频异常检测报告

## 基本信息
- **特征提取器**: {extractor_name}
- **检测方法**: {method}
- **总样本数**: {n_samples}
- **特征维度**: {X.shape[1]}
- **聚类数量**: {n_clusters}
- **异常阈值**: {threshold}% 百分位
{preproc_section}
## 检测结果
- **正常样本**: {n_normal} ({n_normal / n_samples * 100:.1f}%)
- **异常样本**: {n_outliers} ({n_outliers / n_samples * 100:.1f}%)

## 聚类分布
{cluster_dist}

## 检测到的异常
{anomaly_list if anomaly_list else "  无异常检测"}
{skip_section}"""
    return report


# ==================== Gradio 界面 ====================

def create_ui():
    """创建Gradio界面"""
    with gr.Blocks(title="音频异常检测") as demo:
        gr.Markdown("""
        # 🎵 音频异常检测与聚类可视化

        上传音频文件，选择特征提取方法，自动进行聚类分析和异常检测。
        支持 MP3、WAV、FLAC、OGG、AAC、M4A、OPUS 格式。上传后自动统一采样率与声道。
        """)

        with gr.Row():
            # 左：上传音频 + 按钮
            with gr.Column(scale=1):
                audio_input = gr.File(
                    label="📁 上传音频（支持多选）",
                    file_count="multiple",
                    file_types=["audio"],
                    height=180,
                )
                run_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")

            # 右：参数设置
            with gr.Column(scale=1):
                extractor_dropdown = gr.Dropdown(
                    choices=list(EXTRACTORS.keys()),
                    value="MFCC (传统方法，速度快)",
                    label="特征提取方法"
                )
                with gr.Row():
                    n_clusters_slider = gr.Slider(
                        minimum=1, maximum=10, value=3, step=1,
                        label="聚类数量 (1=单说话人模式)"
                    )
                    threshold_slider = gr.Slider(
                        minimum=80, maximum=99, value=90, step=1,
                        label="异常阈值 (百分位数)"
                    )
                with gr.Row():
                    use_3d_checkbox = gr.Checkbox(
                        value=True,
                        label="使用3D可视化"
                    )
                    perplexity_slider = gr.Slider(
                        minimum=5, maximum=50, value=30, step=1,
                        label="t-SNE Perplexity"
                    )

        # 第二行：可视化 + 报告 + 日志
        with gr.Row():
            with gr.Column(scale=3):
                plot_output = gr.Plot(label="📊 聚类可视化")
            with gr.Column(scale=1):
                report_output = gr.Markdown()

        # 第三行：运行日志
        with gr.Row():
            log_output = gr.Textbox(
                label="📝 运行日志",
                lines=12,
                max_lines=20,
                interactive=False,
            )

        # 事件绑定
        run_btn.click(
            fn=process_audio_files,
            inputs=[
                audio_input,
                extractor_dropdown,
                n_clusters_slider,
                threshold_slider,
                use_3d_checkbox,
                perplexity_slider,
            ],
            outputs=[plot_output, report_output, log_output]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
