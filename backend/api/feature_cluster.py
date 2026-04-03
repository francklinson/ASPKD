"""
特征聚类分析 API
基于 AudioFeatureCluster 实现音频特征提取和聚类可视化
"""
import os
import sys
import time
import shutil
import tempfile
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel

# 添加 AudioFeatureCluster 到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "algorithms", "AudioFeatureCluster"))

router = APIRouter()

# 支持的特征提取器
EXTRACTORS = {
    "mfcc": "MFCC (传统方法，速度快)",
    "mel": "Mel Spectrogram (传统方法)",
    "hubert": "HuBERT (深度学习)",
    "wavlm": "WavLM (深度学习)",
    "xlsr-wav2vec2": "XLSR-Wav2Vec2 (多语言)",
    "ast": "AST (Transformer)",
    "mert": "MERT (音乐理解)",
}


class ClusterRequest(BaseModel):
    """聚类分析请求"""
    extractor_type: str = "mfcc"
    n_clusters: int = 3
    anomaly_threshold: int = 90
    use_3d: bool = True
    tsne_perplexity: int = 30


class ClusterResponse(BaseModel):
    """聚类分析响应"""
    task_id: str
    status: str
    message: str


class ClusterResult(BaseModel):
    """聚类结果"""
    task_id: str
    status: str
    progress: float
    result_image: Optional[str] = None
    report: Optional[str] = None
    error: Optional[str] = None


def log_operation(operation: str, details: str = "", status: str = "INFO"):
    """记录操作日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [FeatureCluster] [{status}] {operation} | {details}")


@router.get("/extractors")
async def get_available_extractors():
    """获取可用的特征提取器列表"""
    return {
        "extractors": [
            {"id": key, "name": value}
            for key, value in EXTRACTORS.items()
        ]
    }


@router.post("/analyze", response_model=ClusterResponse)
async def analyze_audio_cluster(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    extractor_type: str = Form("mfcc"),
    n_clusters: int = Form(3),
    anomaly_threshold: int = Form(90),
    use_3d: bool = Form(True),
    tsne_perplexity: int = Form(30)
):
    """
    上传音频文件并进行特征聚类分析

    - **files**: 音频文件列表 (支持 wav, mp3, flac 等格式)
    - **extractor_type**: 特征提取器类型
    - **n_clusters**: 聚类数量
    - **anomaly_threshold**: 异常阈值百分位数
    - **use_3d**: 是否使用3D可视化
    - **tsne_perplexity**: t-SNE perplexity参数
    """
    import uuid

    start_time = time.time()
    task_id = str(uuid.uuid4())

    log_operation(
        "ANALYZE_START",
        f"任务ID={task_id}, 提取器={extractor_type}, 文件数={len(files)}"
    )

    # 验证提取器类型
    if extractor_type not in EXTRACTORS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的提取器类型: {extractor_type}"
        )

    # 验证文件
    if not files or len(files) < 2:
        raise HTTPException(
            status_code=400,
            detail="请至少上传2个音频文件进行分析"
        )

    allowed_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    valid_files = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in allowed_extensions:
            valid_files.append(file)

    if len(valid_files) < 2:
        raise HTTPException(
            status_code=400,
            detail="请至少上传2个有效的音频文件"
        )

    # 创建任务目录
    task_dir = os.path.join("uploads", "cluster", task_id)
    os.makedirs(task_dir, exist_ok=True)

    # 保存上传的文件
    saved_files = []
    try:
        for file in valid_files:
            file_path = os.path.join(task_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_files.append(file_path)

        log_operation(
            "FILES_SAVED",
            f"任务ID={task_id}, 保存了 {len(saved_files)} 个文件"
        )

        # 启动后台分析任务
        background_tasks.add_task(
            run_cluster_analysis,
            task_id=task_id,
            file_paths=saved_files,
            extractor_type=extractor_type,
            n_clusters=n_clusters,
            anomaly_threshold=anomaly_threshold,
            use_3d=use_3d,
            tsne_perplexity=tsne_perplexity
        )

        elapsed_time = (time.time() - start_time) * 1000
        log_operation(
            "ANALYZE_QUEUED",
            f"任务ID={task_id}, 耗时={elapsed_time:.2f}ms"
        )

        return ClusterResponse(
            task_id=task_id,
            status="queued",
            message=f"分析任务已创建，共 {len(saved_files)} 个文件"
        )

    except Exception as e:
        # 清理临时文件
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)
        log_operation(
            "ANALYZE_ERROR",
            f"任务ID={task_id}, 错误={str(e)}",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"创建分析任务失败: {str(e)}")


def run_cluster_analysis(
    task_id: str,
    file_paths: List[str],
    extractor_type: str,
    n_clusters: int,
    anomaly_threshold: int,
    use_3d: bool,
    tsne_perplexity: int
):
    """运行聚类分析（后台任务）"""
    from audio_anomaly_detection import Config, FeatureExtractorFactory, AnomalyDetector

    start_time = time.time()
    log_operation("CLUSTER_START", f"任务ID={task_id}, 开始聚类分析")
    log_operation("PARAMS", f"任务ID={task_id}, 提取器={extractor_type}, 聚类数={n_clusters}, "
                            f"异常阈值={anomaly_threshold}, 3D={use_3d}, t-SNE perplexity={tsne_perplexity}")

    try:
        # 创建配置
        log_operation("CONFIG", f"任务ID={task_id}, 创建配置...")
        config = Config(feature_extractor_type=extractor_type)
        config.n_clusters = n_clusters
        config.outlier_threshold_percentile = anomaly_threshold
        config.tsne_perplexity = tsne_perplexity
        log_operation("CONFIG", f"任务ID={task_id}, 配置完成: n_clusters={config.n_clusters}, "
                                f"outlier_threshold={config.outlier_threshold_percentile}, "
                                f"tsne_perplexity={config.tsne_perplexity}, max_iter={config.tsne_max_iter}")

        # 设置输出路径
        result_dir = os.path.join("visualize", "cluster", task_id)
        os.makedirs(result_dir, exist_ok=True)
        config.output_image = os.path.join(result_dir, f"cluster_result_{extractor_type}.png")
        log_operation("OUTPUT", f"任务ID={task_id}, 输出路径: {config.output_image}")

        # 创建特征提取器
        log_operation("EXTRACTOR_LOAD", f"任务ID={task_id}, 正在加载特征提取器: {extractor_type}...")
        feature_extractor = FeatureExtractorFactory.create_extractor(config)
        log_operation("EXTRACTOR_LOAD", f"任务ID={task_id}, 特征提取器加载完成")

        # 提取特征
        log_operation("FEATURE_EXTRACT", f"任务ID={task_id}, 开始提取特征, 共 {len(file_paths)} 个文件")
        features_list = []
        file_labels = []

        for i, file_path in enumerate(file_paths):
            try:
                log_operation("FILE_PROCESSING", f"任务ID={task_id}, [{i+1}/{len(file_paths)}] 处理中: {os.path.basename(file_path)}")
                features = feature_extractor.extract_features(file_path)
                if len(features) > 0:
                    features_list.append(features[0])  # 取第一个片段
                    file_labels.append(os.path.basename(file_path))
                    log_operation("FILE_SUCCESS", f"任务ID={task_id}, [{i+1}/{len(file_paths)}] 成功提取特征, "
                                                  f"特征维度: {features[0].shape}")
                else:
                    log_operation("FILE_WARNING", f"任务ID={task_id}, [{i+1}/{len(file_paths)}] 未提取到特征", "WARNING")
            except Exception as e:
                log_operation("FILE_ERROR", f"任务ID={task_id}, [{i+1}/{len(file_paths)}] 文件处理失败: {os.path.basename(file_path)}, 错误: {str(e)}", "ERROR")
                import traceback
                log_operation("FILE_ERROR_DETAIL", f"任务ID={task_id}, 错误详情: {traceback.format_exc()}", "ERROR")

        if len(features_list) < 2:
            raise ValueError(f"成功处理的文件不足2个，无法进行聚类分析 (成功: {len(features_list)} 个)")

        X = np.array(features_list)
        log_operation("FEATURE_DONE", f"任务ID={task_id}, 特征提取完成: 样本数={X.shape[0]}, 特征维度={X.shape[1]}")

        # 执行聚类和异常检测
        log_operation("CLUSTERING", f"任务ID={task_id}, 开始执行聚类和异常检测...")
        detector = AnomalyDetector(config)
        log_operation("CLUSTERING", f"任务ID={task_id}, 正在拟合模型 (fit)...")
        detector.fit(X)
        log_operation("CLUSTERING", f"任务ID={task_id}, 模型拟合完成, 正在进行预测...")
        labels, is_outlier = detector.predict(X)
        log_operation("CLUSTERING", f"任务ID={task_id}, 预测完成: 聚类标签={set(labels)}, 异常数={np.sum(is_outlier)}")

        # 计算异常分数 (距离)
        log_operation("SCORING", f"任务ID={task_id}, 计算异常分数...")
        X_scaled = detector.scaler.transform(X)
        centers = detector.kmeans.cluster_centers_
        distances = np.linalg.norm(X_scaled - centers[labels], axis=1)
        outlier_scores = distances / detector.threshold if detector.threshold else distances
        log_operation("SCORING", f"任务ID={task_id}, 异常分数计算完成: min={np.min(outlier_scores):.4f}, "
                                 f"max={np.max(outlier_scores):.4f}, mean={np.mean(outlier_scores):.4f}")

        # 生成可视化
        interactive_html_path = None
        if use_3d:
            # 只生成交互式 3D 可视化
            log_operation("VISUALIZATION", f"任务ID={task_id}, 生成交互式 3D 可视化...")
            interactive_html_path = detector.visualize_interactive(X, is_outlier, file_labels)
            if interactive_html_path:
                log_operation("VISUALIZATION", f"任务ID={task_id}, 交互式可视化已保存到: {interactive_html_path}")
        else:
            # 2D 模式下生成静态图
            log_operation("VISUALIZATION", f"任务ID={task_id}, 生成 2D 静态可视化...")
            detector.visualize(X, is_outlier, file_labels, use_3d=False)
            log_operation("VISUALIZATION", f"任务ID={task_id}, 静态可视化已保存到: {config.output_image}")

        # 生成报告
        report = generate_report(file_labels, labels, is_outlier, outlier_scores, extractor_type)

        # 保存结果
        result_data = {
            "task_id": task_id,
            "status": "completed",
            "progress": 100.0,
            "result_image": f"visualize/cluster/{task_id}/cluster_result_{extractor_type}.png",
            "report": report,
            "extractor_type": extractor_type,
            "n_files": len(file_paths),
            "n_processed": len(features_list),
            "n_clusters": n_clusters,
            "n_outliers": int(np.sum(is_outlier)),
            "use_3d": use_3d
        }
        
        # 添加交互式 HTML 路径
        if interactive_html_path:
            # 将绝对路径转换为相对路径（相对于项目根目录）
            rel_html_path = interactive_html_path.replace(os.path.sep, '/')
            # 移除绝对路径前缀，只保留 visualize/... 部分
            if '/visualize/' in rel_html_path:
                rel_html_path = rel_html_path[rel_html_path.find('visualize/'):]
            elif rel_html_path.startswith('./'):
                rel_html_path = rel_html_path[2:]
            result_data["interactive_html"] = rel_html_path
            log_operation("VISUALIZATION", f"任务ID={task_id}, 交互式HTML相对路径: {rel_html_path}")

        # 保存结果到文件
        import json
        result_file = os.path.join(result_dir, "result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        elapsed_time = (time.time() - start_time)
        log_operation(
            "CLUSTER_SUCCESS",
            f"任务ID={task_id}, 耗时={elapsed_time:.2f}s, 异常数={result_data['n_outliers']}"
        )

    except Exception as e:
        log_operation(
            "CLUSTER_ERROR",
            f"任务ID={task_id}, 错误={str(e)}",
            "ERROR"
        )
        # 保存错误信息
        result_data = {
            "task_id": task_id,
            "status": "failed",
            "progress": 0,
            "error": str(e)
        }
        result_dir = os.path.join("visualize", "cluster", task_id)
        os.makedirs(result_dir, exist_ok=True)
        import json
        result_file = os.path.join(result_dir, "result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)


def generate_report(file_labels, labels, is_outlier, outlier_scores, extractor_type):
    """生成分析报告"""
    report_lines = [f"# 音频特征聚类分析报告 ({extractor_type.upper()})", ""]
    report_lines.append(f"## 分析概况")
    report_lines.append(f"- 分析文件数: {len(file_labels)}")
    report_lines.append(f"- 聚类数量: {len(set(labels))}")
    report_lines.append(f"- 检测到的异常: {int(np.sum(is_outlier))} 个")
    report_lines.append("")

    report_lines.append(f"## 聚类分布")
    for cluster_id in sorted(set(labels)):
        count = np.sum(labels == cluster_id)
        report_lines.append(f"- 聚类 {cluster_id}: {count} 个样本")
    report_lines.append("")

    report_lines.append(f"## 异常样本")
    if np.sum(is_outlier) > 0:
        report_lines.append("| 文件名 | 异常分数 |")
        report_lines.append("|--------|----------|")
        for i, (label, is_out, score) in enumerate(zip(file_labels, is_outlier, outlier_scores)):
            if is_out:
                report_lines.append(f"| {label} | {score:.4f} |")
    else:
        report_lines.append("未检测到异常样本")
    report_lines.append("")

    report_lines.append(f"## 所有样本")
    report_lines.append("| 文件名 | 所属聚类 | 是否异常 | 异常分数 |")
    report_lines.append("|--------|----------|----------|----------|")
    for label, cluster, is_out, score in zip(file_labels, labels, is_outlier, outlier_scores):
        out_str = "是" if is_out else "否"
        report_lines.append(f"| {label} | {cluster} | {out_str} | {score:.4f} |")

    return "\n".join(report_lines)


@router.get("/result/{task_id}")
async def get_cluster_result(task_id: str):
    """获取聚类分析结果"""
    result_dir = os.path.join("visualize", "cluster", task_id)
    result_file = os.path.join(result_dir, "result.json")

    if not os.path.exists(result_file):
        # 检查任务是否还在进行中
        task_dir = os.path.join("uploads", "cluster", task_id)
        if os.path.exists(task_dir):
            return {
                "task_id": task_id,
                "status": "processing",
                "progress": 50.0,
                "message": "分析正在进行中..."
            }
        else:
            raise HTTPException(status_code=404, detail="任务不存在")

    import json
    with open(result_file, "r", encoding="utf-8") as f:
        result = json.load(f)

    return result
