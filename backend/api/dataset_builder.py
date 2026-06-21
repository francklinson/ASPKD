"""
数据集构建 API V1（遗留模块）

⚠️ 此模块大部分功能已被 V2（dataset_builder_v2.py）替代，
    前端上传/标注/会话管理已切换到 V2。

仍被使用的部分（保留）：
  - get_url_path() / get_file_path() — 被 V2 导入用作路径映射工具
  - GET /references                 — 前端 loadReferences() 使用
  - GET /stats                      — 前端 refreshStats() 使用

已过时（前端不再调用，仅做兼容保留）：
  - POST /upload-manual             — 被 V2 from-manual(skip_slicing=true) 替代
  - POST /upload-and-split          — 被 V2 from-manual(skip_slicing=false) 替代
  - POST /split-manual              — V2 会话内直接处理
  - GET /segments                   — V2 会话管理片段
  - GET /segments/{category}        — 同上
  - POST /annotate                  — V2 会话管理标注
  - POST /annotate/batch            — 同上
  - GET /annotation-stats           — 无引用
"""
import os
import shutil
import json
import time
import random
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel, Field
import librosa
import soundfile as sf

router = APIRouter()

# 数据集根目录
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# 数据集根目录
DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "spk")
os.makedirs(DATASET_ROOT, exist_ok=True)

# 临时上传目录（默认，用于未登录场景）
UPLOAD_TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "uploads", "dataset_temp")
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)

# 切分后音频临时存储目录（默认，用于未登录场景）
SLICE_TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "uploads", "dataset_slices")
os.makedirs(SLICE_TEMP_DIR, exist_ok=True)


def get_user_temp_dirs(username: Optional[str] = None) -> tuple:
    """获取共享临时目录（确保目录存在）"""
    os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)
    os.makedirs(SLICE_TEMP_DIR, exist_ok=True)
    return UPLOAD_TEMP_DIR, SLICE_TEMP_DIR


def _static_mount_url(disk_relative_dir: str, url_prefix: str,
                      normalized_path: str, normalized_project_root: str) -> Optional[str]:
    """
    如果路径在已知的静态挂载目录下，返回对应的 URL 路径。

    Args:
        disk_relative_dir: 磁盘上相对于项目根目录的路径，如 "data/spk"
        url_prefix: URL 前缀，如 "/data/spk/"
        normalized_path: 规范化的文件系统路径
        normalized_project_root: 规范化的项目根目录

    Returns:
        URL 路径或 None
    """
    prefix = os.path.join(normalized_project_root, disk_relative_dir).replace("\\", "/") + "/"
    if normalized_path.startswith(prefix):
        rel = normalized_path[len(prefix):]
        return f"{url_prefix}{rel}"
    return None


def get_file_path(url_path: str) -> str:
    """
    将 URL 路径转换为文件系统绝对路径

    仍在被 V2 (dataset_builder_v2.py) 导入使用。

    Args:
        url_path: URL 路径 (如 /data/spk/... 或 /uploads/...)

    Returns:
        文件系统绝对路径
    """
    # 静态挂载点映射: (URL 前缀, 磁盘相对路径)
    mount_map = [
        ("/data/spk/", "data/spk"),
        ("/data/dataset-builder/", "data/dataset_builder"),
        ("/uploads/", "data/uploads"),
        ("/output/", "data/output"),
        ("/visualize/", "data/output/vis"),
    ]
    for url_prefix, disk_rel in mount_map:
        if url_path.startswith(url_prefix):
            rel = url_path[len(url_prefix):].lstrip("/")
            return os.path.join(PROJECT_ROOT, disk_rel, rel)

    # 后备：其他以 / 开头的路径，尝试作为项目根目录下的路径
    if url_path.startswith('/'):
        return os.path.join(PROJECT_ROOT, url_path.lstrip('/'))

    return url_path


def get_url_path(file_path: str) -> str:
    """
    将文件系统路径转换为 HTTP 可访问的 URL 路径

    已知静态挂载点:
      data/spk/          → /data/spk/
      data/uploads/      → /uploads/
      data/output/       → /output/
      data/output/vis/   → /visualize/
      data/dataset_builder/ → /data/dataset-builder/

    Args:
        file_path: 文件系统绝对路径

    Returns:
        URL 路径
    """
    if not file_path:
        return file_path

    # 如果已经是 URL 路径格式，直接返回
    for prefix in ("/data/spk/", "/uploads/", "/output/", "/visualize/", "/data/dataset-builder/"):
        if file_path.startswith(prefix):
            return file_path

    # 规范化路径
    normalized_path = os.path.normpath(file_path).replace("\\", "/")
    npr = os.path.normpath(PROJECT_ROOT).replace("\\", "/")

    # 按优先级检查已知的静态挂载目录
    # (磁盘相对路径, URL 前缀)
    known_mounts = [
        ("data/spk", "/data/spk/"),
        ("data/dataset_builder", "/data/dataset-builder/"),
        ("data/uploads", "/uploads/"),
        ("data/output", "/output/"),
    ]
    for disk_rel, url_prefix in known_mounts:
        result = _static_mount_url(disk_rel, url_prefix, normalized_path, npr)
        if result:
            return result

    # 通用处理：项目根目录下的其他文件
    if normalized_path.startswith(npr + "/") or normalized_path == npr:
        rel = normalized_path[len(npr):].lstrip("/")
        return f"/{rel}"

    # 后备：从路径中推断已知前缀
    for marker, url_prefix in [
        ("/data/spk/", "/data/spk/"),
        ("/data/uploads/", "/uploads/"),
        ("/data/output/", "/output/"),
        ("/uploads/", "/uploads/"),
    ]:
        idx = normalized_path.find(marker)
        if idx != -1:
            return f"{url_prefix}{normalized_path[idx + len(marker):]}"

    return file_path

# 数据集划分随机种子（确保可复现）
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def log_operation(operation: str, details: str = "", status: str = "INFO"):
    """记录操作日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [DatasetBuilder] [{status}] {operation} | {details}")


# ========== 数据模型 ==========

class AudioSegmentInfo(BaseModel):
    """音频片段信息"""
    segment_id: str
    original_filename: str
    segment_filename: str
    duration: float
    sample_rate: int
    file_path: str
    spectrogram_path: Optional[str] = None  # 时频图路径
    reference_audio: str
    start_time: float
    end_time: float
    label: Optional[str] = None  # "normal", "anomaly", None表示未标注
    annotated_at: Optional[str] = None


class DatasetCategoryInfo(BaseModel):
    """数据集类别信息"""
    category_name: str
    train_normal_count: int
    test_normal_count: int
    test_anomaly_count: int
    total_count: int


class DatasetStats(BaseModel):
    """数据集统计信息"""
    total_categories: int
    total_audio_files: int
    train_total: int
    test_total: int
    train_normal: int
    test_normal: int
    test_anomaly: int
    categories: List[DatasetCategoryInfo]


class UploadAndSplitResponse(BaseModel):
    """上传并切分响应"""
    success: bool
    message: str
    segments: List[AudioSegmentInfo]
    reference_audio: str


class AnnotationRequest(BaseModel):
    """标注请求"""
    segment_id: str
    label: str  # "normal" 或 "anomaly"
    category: str  # 参考音频名称（类别名）


class BatchAnnotationRequest(BaseModel):
    """批量标注请求"""
    annotations: List[Dict[str, str]]  # 每个元素包含 segment_id, label, category, segment_path


class BatchAnnotationResponse(BaseModel):
    """批量标注响应"""
    success: bool
    message: str
    total: int
    success_count: int
    failed_count: int
    failed_items: List[Dict[str, str]]


class AnnotationResponse(BaseModel):
    """标注响应"""
    success: bool
    message: str
    segment: Optional[AudioSegmentInfo] = None


class AnnotationStats(BaseModel):
    """标注统计信息"""
    total_annotated: int
    normal_count: int
    anomaly_count: int
    unlabeled_count: int
    by_category: Dict[str, Dict[str, int]]
    recent_annotations: List[Dict[str, Any]]


class DatasetSplitLog(BaseModel):
    """数据集划分日志"""
    timestamp: str
    category: str
    operation: str
    file_name: str
    split_type: str  # "train" 或 "test"
    label: str  # "normal" 或 "anomaly"


# ========== 辅助函数 ==========

def get_reference_audios() -> List[Dict[str, Any]]:
    """获取所有可用的参考音频列表"""
    try:
        from backend.core.shazam import AudioFingerprinter
        with AudioFingerprinter() as fp:
            references = fp.get_all_references()
            return references
    except Exception as e:
        log_operation("GET_REFERENCES_ERROR", str(e), "ERROR")
        return []


def split_audio_auto_match_v2(
    audio_path: str,
    output_dir: str,
    segment_duration: float = 10.0,
    overlap_ratio: float = 0.5,
    min_segment_duration: float = 3.0,
    auto_annotate: bool = False
) -> List[Dict[str, Any]]:
    """
    增强版音频切分函数
    支持更多参数控制和自动标注

    Args:
        audio_path: 原始音频文件路径
        output_dir: 切分后音频输出目录
        segment_duration: 每个片段的时长（秒）
        overlap_ratio: 重叠比例（0-1）
        min_segment_duration: 最小片段时长（秒）
        auto_annotate: 是否启用自动标注

    Returns:
        切分后的音频片段信息列表
    """
    segments = []

    try:
        # 导入长音频分析器
        from backend.core.long_audio_analyzer import LongAudioAnalyzer, AnalyzerConfig
        from backend.core.shazam.database.in_memory import InMemoryConnector

        # 创建数据库连接（使用进程内内存数据库）
        db_connector = InMemoryConnector()

        # 创建分析器配置（使用传入的参数）
        config = AnalyzerConfig(
            window_size=segment_duration,
            step_size=segment_duration * (1 - overlap_ratio),
            match_threshold=10,
            min_match_ratio=0.05,
            time_tolerance=2.0,
            min_segment_duration=min_segment_duration,
            use_parallel=True,
            max_workers=4,
            skip_silence=True
        )

        # 创建长音频分析器
        analyzer = LongAudioAnalyzer(config=config, db_connector=db_connector)

        log_operation("ANALYZE_START", f"开始分析音频: {os.path.basename(audio_path)}")

        # 分析长音频
        result = analyzer.analyze(audio_path)

        # 关闭数据库连接
        try:
            db_connector.cursor.close()
            db_connector.conn.close()
        except Exception:
            pass

        if not result.segment_matches:
            log_operation("NO_SEGMENTS_FOUND", f"音频 {audio_path} 中未找到匹配片段", "WARNING")
            return []

        log_operation("ANALYZE_SUCCESS", f"找到 {len(result.segment_matches)} 个匹配片段")

        # 加载原始音频
        y, sr = librosa.load(audio_path, sr=22050)
        audio_total_duration = librosa.get_duration(y=y, sr=sr)

        # 根据分析结果切分音频
        for idx, segment_match in enumerate(result.segment_matches):
            # 计算切分位置
            start_time = segment_match.start_time
            end_time = segment_match.end_time

            # 确保不超出音频边界
            start_time = max(0, start_time)
            end_time = min(audio_total_duration, end_time)

            # 如果片段太短，跳过
            if end_time - start_time < min_segment_duration:
                continue

            # 转换为采样点
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # 切分音频
            segment_audio = y[start_sample:end_sample]

            # 自动标注（基于音频质量）
            auto_label = None
            if auto_annotate:
                auto_label = _auto_detect_audio_quality(segment_audio, sr)

            # 生成文件名
            original_name = os.path.splitext(os.path.basename(audio_path))[0]
            matched_name = segment_match.music_name
            segment_filename = f"{original_name}_seg{idx:03d}_{matched_name}_{int(start_time)}s.wav"
            segment_path = os.path.join(output_dir, segment_filename)

            # 保存音频片段
            sf.write(segment_path, segment_audio, sr)

            # 生成时频图
            spectrogram_path = segment_path.replace('.wav', '.png')
            generate_spectrogram_image(segment_path, spectrogram_path)

            # 计算唯一ID
            file_hash = hashlib.md5(f"{audio_path}_{start_time}_{end_time}".encode()).hexdigest()[:12]

            segment_info = {
                "segment_id": f"seg_{file_hash}",
                "original_filename": os.path.basename(audio_path),
                "segment_filename": segment_filename,
                "duration": round(end_time - start_time, 2),
                "sample_rate": sr,
                "file_path": get_url_path(segment_path),
                "spectrogram_path": get_url_path(spectrogram_path),
                "reference_audio": matched_name,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "confidence": segment_match.confidence,
                "match_ratio": segment_match.match_ratio,
                "is_reliable": segment_match.is_reliable,
                "label": auto_label,
                "annotated_at": datetime.now().isoformat() if auto_label else None,
                "auto_annotated": auto_annotate
            }
            segments.append(segment_info)

            log_operation("SEGMENT_SAVED",
                         f"保存片段: {segment_filename} ({segment_info['duration']}s), "
                         f"自动标注: {auto_label}")

        if segments:
            log_operation("SPLIT_SUCCESS",
                         f"音频 {os.path.basename(audio_path)} 切分为 {len(segments)} 个片段")
        else:
            log_operation("NO_SEGMENTS_FOUND",
                         f"音频 {audio_path} 中未找到有效片段（所有片段都太短）", "WARNING")

    except Exception as e:
        log_operation("SPLIT_ERROR", f"切分音频失败: {str(e)}", "ERROR")
        raise HTTPException(status_code=500, detail=f"音频切分失败: {str(e)}")

    return segments


def _auto_detect_audio_quality(audio: np.ndarray, sr: int) -> Optional[str]:
    """
    自动检测音频质量并给出标注建议

    Returns:
        "normal" 或 "anomaly" 或 None
    """
    try:
        # 计算音频特征
        # 1. 信噪比估算
        rms = librosa.feature.rms(y=audio)[0]
        signal_power = np.mean(rms ** 2)

        # 2. 计算过零率（检测噪声）
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)

        # 3. 计算频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_mean = np.mean(spectral_centroids)

        # 简单的规则判断
        # 如果过零率过高或频谱质心异常，可能是异常音频
        if zcr_mean > 0.15 or centroid_mean > 4000:
            return "anomaly"
        return "normal"
    except Exception:
        return None


def split_audio_auto_match(
    audio_path: str,
    output_dir: str,
    segment_duration: float = 10.0
) -> List[Dict[str, Any]]:
    """
    使用长音频分析器自动匹配参考音频并切分音频文件
    支持提取长音频中所有匹配的片段

    Args:
        audio_path: 原始音频文件路径
        output_dir: 切分后音频输出目录
        segment_duration: 每个片段的时长（秒）

    Returns:
        切分后的音频片段信息列表，每个片段包含自动匹配到的参考音频名称
    """
    segments = []

    try:
        # 导入长音频分析器
        from backend.core.long_audio_analyzer import LongAudioAnalyzer, AnalyzerConfig
        from backend.core.shazam.database.in_memory import InMemoryConnector
        from backend.core.shazam.utils.hparam import hp

        # 创建数据库连接（使用进程内内存数据库）
        db_connector = InMemoryConnector()

        # 创建分析器配置
        config = AnalyzerConfig(
            window_size=segment_duration,      # 窗口大小（秒）
            step_size=segment_duration / 2,    # 步长（50%重叠）
            match_threshold=10,                 # 匹配阈值
            min_match_ratio=0.05,              # 最小匹配比例
            time_tolerance=2.0,                # 时间容差（秒）
            min_segment_duration=3.0,          # 最小片段时长
            use_parallel=True,                 # 使用并行处理
            max_workers=4,                     # 最大线程数
            skip_silence=True                  # 跳过静音区域
        )

        # 创建长音频分析器
        analyzer = LongAudioAnalyzer(config=config, db_connector=db_connector)

        log_operation("ANALYZE_START", f"开始分析音频: {os.path.basename(audio_path)}")

        # 分析长音频
        result = analyzer.analyze(audio_path)

        # 关闭数据库连接
        try:
            db_connector.cursor.close()
            db_connector.conn.close()
        except Exception:
            pass

        if not result.segment_matches:
            log_operation("NO_SEGMENTS_FOUND", f"音频 {audio_path} 中未找到匹配片段", "WARNING")
            return []

        log_operation("ANALYZE_SUCCESS", f"找到 {len(result.segment_matches)} 个匹配片段")

        # 加载原始音频
        y, sr = librosa.load(audio_path, sr=22050)
        audio_total_duration = librosa.get_duration(y=y, sr=sr)

        # 根据分析结果切分音频
        for idx, segment_match in enumerate(result.segment_matches):
            # 计算切分位置
            start_time = segment_match.start_time
            end_time = segment_match.end_time

            # 确保不超出音频边界
            start_time = max(0, start_time)
            end_time = min(audio_total_duration, end_time)

            # 如果片段太短，跳过
            if end_time - start_time < 3.0:
                continue

            # 转换为采样点
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # 切分音频
            segment_audio = y[start_sample:end_sample]

            # 生成文件名
            original_name = os.path.splitext(os.path.basename(audio_path))[0]
            matched_name = segment_match.music_name
            segment_filename = f"{original_name}_seg{idx:03d}_{matched_name}_{int(start_time)}s.wav"
            segment_path = os.path.join(output_dir, segment_filename)

            # 保存音频片段
            sf.write(segment_path, segment_audio, sr)

            # 生成时频图
            spectrogram_path = segment_path.replace('.wav', '.png')
            generate_spectrogram_image(segment_path, spectrogram_path)

            # 计算唯一ID
            file_hash = hashlib.md5(f"{audio_path}_{start_time}_{end_time}".encode()).hexdigest()[:12]

            segment_info = {
                "segment_id": f"seg_{file_hash}",
                "original_filename": os.path.basename(audio_path),
                "segment_filename": segment_filename,
                "duration": round(end_time - start_time, 2),
                "sample_rate": sr,
                "file_path": get_url_path(segment_path),
                "spectrogram_path": get_url_path(spectrogram_path),
                "reference_audio": matched_name,  # 自动匹配到的参考音频名称
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "confidence": segment_match.confidence,
                "match_ratio": segment_match.match_ratio,
                "is_reliable": segment_match.is_reliable,
                "label": None,
                "annotated_at": None
            }
            segments.append(segment_info)

            log_operation("SEGMENT_SAVED", f"保存片段: {segment_filename} ({segment_info['duration']}s), 时频图: {os.path.basename(spectrogram_path)}")

        if segments:
            log_operation("SPLIT_SUCCESS", f"音频 {os.path.basename(audio_path)} 切分为 {len(segments)} 个片段")
        else:
            log_operation("NO_SEGMENTS_FOUND", f"音频 {audio_path} 中未找到有效片段（所有片段都太短）", "WARNING")

    except Exception as e:
        log_operation("SPLIT_ERROR", f"切分音频失败: {str(e)}", "ERROR")
        raise HTTPException(status_code=500, detail=f"音频切分失败: {str(e)}")

    return segments


def get_category_path(category: str) -> str:
    """获取类别数据目录路径"""
    category_path = os.path.join(DATASET_ROOT, category)
    return category_path


def ensure_category_structure(category: str):
    """确保类别目录结构存在"""
    category_path = get_category_path(category)

    # 创建训练集目录（仅正常数据）
    train_good_path = os.path.join(category_path, "train", "good")
    os.makedirs(train_good_path, exist_ok=True)

    # 创建测试集目录（正常和异常数据）
    test_good_path = os.path.join(category_path, "test", "good")
    test_anomaly_path = os.path.join(category_path, "test", "anomaly")
    os.makedirs(test_good_path, exist_ok=True)
    os.makedirs(test_anomaly_path, exist_ok=True)

    return {
        "train_good": train_good_path,
        "test_good": test_good_path,
        "test_anomaly": test_anomaly_path,
        "category_path": category_path
    }


def split_train_test(existing_train_count: int, existing_test_count: int) -> str:
    """
    决定新数据应该放入训练集还是测试集
    按照训练集:测试集 = 10:1 的比例划分

    Args:
        existing_train_count: 现有训练集数量
        existing_test_count: 现有测试集数量

    Returns:
        "train" 或 "test"
    """
    # 计算当前比例
    total = existing_train_count + existing_test_count

    if total == 0:
        # 第一个数据放入训练集
        return "train"

    current_train_ratio = existing_train_count / total
    target_train_ratio = 10 / 11  # 约 0.909

    # 如果训练集比例低于目标比例，优先放入训练集
    if current_train_ratio < target_train_ratio:
        return "train"
    else:
        return "test"


def log_dataset_split(category: str, operation: str, file_name: str, split_type: str, label: str):
    """记录数据集划分日志"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "operation": operation,
        "file_name": file_name,
        "split_type": split_type,
        "label": label
    }

    # 日志文件路径
    log_file = os.path.join(DATASET_ROOT, "split_log.jsonl")

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log_operation("LOG_WRITE_ERROR", str(e), "ERROR")


def generate_spectrogram_image(audio_path: str, output_path: str, offset: float = 0.0, duration: float = None) -> bool:
    """
    生成音频的时频图
    
    使用与 preprocessing.py plot_spectrogram 相同的算法：
    - 采样率 22050 Hz
    - 幅值归一化
    - STFT 计算
    - 转换为分贝标度
    - 6.3x6.3 英寸图像，无白边
    
    Args:
        audio_path: 输入音频文件路径
        output_path: 输出图像保存路径
        offset: 开始时间偏移量（秒）
        duration: 持续时长（秒），None表示整个音频
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt

        # 加载音频（与 preprocessing.py 一致）
        y, sr = librosa.load(audio_path, offset=offset, duration=duration, sr=22050)

        # 幅值归一化（与 preprocessing.py 一致）
        y = librosa.util.normalize(y)

        # 计算STFT（与 preprocessing.py 一致）
        D = librosa.stft(y)
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # 绘制时频图（与 preprocessing.py 一致）
        # 图形大小 6.3 x 6.3 英寸
        plt.figure(figsize=(6.3, 6.3))
        librosa.display.specshow(DB, sr=sr)
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图像，没有白边（与 preprocessing.py 一致）
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=False)
        plt.close()

        log_operation("SPECTROGRAM_GENERATED", f"时频图已生成: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        log_operation("SPECTROGRAM_ERROR", f"生成时频图失败: {str(e)}", "ERROR")
        return False


# ========== API 端点 ==========
# ⚠️ 以下端点大部分为 V1 遗留，前端已切换到 V2 会话系统。
# 仍在使用的: GET /references, GET /stats
# 已过时的:   upload / upload-and-split / split-manual / annotate / segments
#             （保留以兼容外部调用，但前端 dataset.html 不再直接使用）

@router.get("/references", response_model=List[Dict[str, Any]])
# ✅ 仍被前端 loadReferences() 使用
async def get_available_references():
    """
    获取所有可用的参考音频列表
    用于数据集构建时选择参考音频类别
    """
    log_operation("GET_REFERENCES", "获取参考音频列表")
    references = get_reference_audios()
    return references


@router.post("/upload-and-split")
# ⚠️ V1 遗留 — 前端已切换到 V2 from-manual(skip_slicing=false)
async def upload_and_split_audio(
    files: List[UploadFile] = File(..., description="支持批量上传音频文件"),
    username: Optional[str] = Form(None),
    segment_duration: float = Form(10.0, description="切分片段时长(秒)"),
    overlap_ratio: float = Form(0.5, description="重叠比例(0-1)", ge=0.0, le=0.9),
    min_segment_duration: float = Form(3.0, description="最小片段时长(秒)"),
    auto_annotate: bool = Form(False, description="是否启用自动标注(基于音频质量检测)")
):
    """
    批量上传音频文件并使用Shazam自动匹配参考音频进行切分

    - **files**: 音频文件列表 (支持 wav, mp3, flac 等格式)
    - **username**: 可选，用于多用户隔离
    - **segment_duration**: 切分片段时长(秒)，默认10秒
    - **overlap_ratio**: 重叠比例(0-1)，默认0.5
    - **min_segment_duration**: 最小片段时长(秒)，默认3秒
    - **auto_annotate**: 是否启用自动标注(基于音频质量检测)

    系统会自动从参考音频库中匹配最合适的参考音频，并使用Shazam音频指纹算法进行切分。
    """
    start_time = time.time()
    log_operation("UPLOAD_SPLIT_START", f"批量上传 {len(files)} 个文件, 用户: {username or '匿名'}")

    # 获取用户隔离的临时目录
    user_upload_dir, user_slice_dir = get_user_temp_dirs(username)

    # 验证文件格式
    allowed_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}

    all_segments = []
    processed_files = []
    failed_files = []
    reference_stats = {}

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            failed_files.append({"file": file.filename, "reason": f"不支持的格式: {ext}"})
            continue

        temp_file_path = None
        try:
            # 保存上传的文件到用户隔离的临时目录
            temp_filename = f"{int(time.time() * 1000)}_{file.filename}"
            temp_file_path = os.path.join(user_upload_dir, temp_filename)

            with open(temp_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            log_operation("UPLOAD_SAVED", f"临时文件: {temp_file_path}")

            # 使用Shazam自动匹配并切分音频（使用新的参数）
            segments = split_audio_auto_match_v2(
                audio_path=temp_file_path,
                output_dir=user_slice_dir,
                segment_duration=segment_duration,
                overlap_ratio=overlap_ratio,
                min_segment_duration=min_segment_duration,
                auto_annotate=auto_annotate
            )

            if segments:
                all_segments.extend(segments)
                matched_ref = segments[0]["reference_audio"] if segments else "unknown"
                reference_stats[matched_ref] = reference_stats.get(matched_ref, 0) + len(segments)
                processed_files.append({"file": file.filename, "segments": len(segments)})
            else:
                failed_files.append({"file": file.filename, "reason": "未找到匹配的参考片段"})

        except Exception as e:
            log_operation("UPLOAD_SPLIT_ERROR", f"处理 {file.filename} 失败: {str(e)}", "ERROR")
            failed_files.append({"file": file.filename, "reason": str(e)})

        finally:
            # 清理临时上传文件
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass

    elapsed_time = (time.time() - start_time) * 1000

    # 获取匹配到的主要参考音频
    main_reference = max(reference_stats.keys(), key=lambda x: reference_stats[x]) if reference_stats else ""

    log_operation("UPLOAD_SPLIT_SUCCESS",
                  f"处理完成: {len(processed_files)}/{len(files)} 个文件, "
                  f"生成 {len(all_segments)} 个片段, 主参考音频: {main_reference}, "
                  f"耗时 {elapsed_time:.2f}ms")

    return {
        "success": len(all_segments) > 0,
        "message": f"批量处理完成：成功 {len(processed_files)} 个文件，失败 {len(failed_files)} 个，共生成 {len(all_segments)} 个片段",
        "segments": all_segments,
        "reference_audio": main_reference,
        "stats": {
            "total_files": len(files),
            "processed": len(processed_files),
            "failed": len(failed_files),
            "total_segments": len(all_segments),
            "reference_distribution": reference_stats
        },
        "processed_files": processed_files,
        "failed_files": failed_files
    }


@router.post("/upload-manual", response_model=UploadAndSplitResponse)
# ⚠️ V1 遗留 — 前端已切换到 V2 from-manual(skip_slicing=true)
async def upload_manual_audio(
    files: List[UploadFile] = File(...),
    username: Optional[str] = Form(None)
):
    """
    批量手动上传音频文件（不自动切分）
    音频将被保存到用户临时目录，用户可后续手动选择参考音频进行切分

    - **files**: 音频文件列表 (支持 wav, mp3, flac 等格式)
    - **username**: 可选，用于多用户隔离
    """
    start_time = time.time()
    log_operation("MANUAL_UPLOAD_START", f"批量上传 {len(files)} 个文件, 用户: {username or '匿名'}")

    allowed_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    user_upload_dir, _ = get_user_temp_dirs(username)

    all_segment_infos = []
    success_count = 0
    failed_files = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            failed_files.append({"file": file.filename, "reason": f"不支持的格式: {ext}"})
            continue

        try:
            temp_filename = f"{int(time.time())}_{file.filename}"
            temp_file_path = os.path.join(user_upload_dir, temp_filename)

            with open(temp_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # 加载音频获取基本信息
            try:
                y, sr = librosa.load(temp_file_path, sr=22050)
                duration = round(librosa.get_duration(y=y, sr=sr), 2)
            except Exception:
                duration = 0.0

            segment_info = {
                "segment_id": f"manual_{hashlib.md5(temp_file_path.encode()).hexdigest()[:12]}",
                "original_filename": file.filename,
                "segment_filename": temp_filename,
                "duration": duration,
                "sample_rate": 22050,
                "file_path": get_url_path(temp_file_path),
                "spectrogram_path": None,
                "reference_audio": "",
                "start_time": 0.0,
                "end_time": duration,
                "label": None,
                "annotated_at": None
            }

            all_segment_infos.append(segment_info)
            success_count += 1
            log_operation("MANUAL_UPLOAD_FILE", f"文件: {temp_filename}", "INFO")

        except Exception as e:
            log_operation("MANUAL_UPLOAD_ERROR", f"文件 {file.filename}: {str(e)}", "ERROR")
            failed_files.append({"file": file.filename, "reason": str(e)})

    elapsed_time = (time.time() - start_time) * 1000

    detail_parts = []
    if success_count:
        detail_parts.append(f"成功 {success_count} 个")
    if failed_files:
        detail_parts.append(f"失败 {len(failed_files)} 个")

    log_operation("MANUAL_UPLOAD_COMPLETE",
                  f"{', '.join(detail_parts)}, 耗时 {elapsed_time:.2f}ms")

    return UploadAndSplitResponse(
        success=success_count > 0,
        message=f"上传完成：{'；'.join(detail_parts)}，请在数据集构建页面选择参考音频进行切分"
                + (f"（失败: {failed_files[0]['file']}: {failed_files[0]['reason']}" if len(failed_files) == 1
                   else f"（{len(failed_files)} 个文件上传失败）" if failed_files else ""),
        segments=all_segment_infos,
        reference_audio=""
    )


@router.post("/split-manual")
# ⚠️ V1 遗留 — 前端不再调用，V2 会话内直接处理切分
async def split_manual_audio(
    file_path: str = Form(...),
    reference_audio: str = Form(...),
    username: Optional[str] = Form(None)
):
    """
    对已手动上传的音频使用指定参考音频进行切分

    - **file_path**: 已上传的音频文件路径
    - **reference_audio**: 用于匹配的参考音频名称
    - **username**: 可选，用于多用户隔离
    """
    start_time = time.time()
    log_operation("MANUAL_SPLIT_START", f"文件: {file_path}, 参考音频: {reference_audio}")

    # 将URL路径转换为文件系统路径
    fs_path = get_file_path(file_path)
    if not os.path.exists(fs_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {fs_path}")

    _, user_slice_dir = get_user_temp_dirs(username)

    try:
        from backend.core.long_audio_analyzer import LongAudioAnalyzer, AnalyzerConfig
        from backend.core.shazam.database.in_memory import InMemoryConnector

        db_connector = InMemoryConnector()
        config = AnalyzerConfig(
            window_size=10.0,
            step_size=5.0,
            match_threshold=10,
            min_match_ratio=0.05,
            time_tolerance=2.0,
            min_segment_duration=3.0,
            use_parallel=True,
            max_workers=4,
            skip_silence=True
        )
        analyzer = LongAudioAnalyzer(config=config, db_connector=db_connector)

        result = analyzer.analyze(fs_path, target_music_name=reference_audio)
        try:
            db_connector.cursor.close()
            db_connector.conn.close()
        except Exception:
            pass

        if not result.segment_matches:
            return UploadAndSplitResponse(
                success=False,
                message=f"未找到与参考音频 '{reference_audio}' 匹配的片段",
                segments=[],
                reference_audio=reference_audio
            )

        y, sr = librosa.load(fs_path, sr=22050)
        total_duration = librosa.get_duration(y=y, sr=sr)
        segments = []

        for idx, segment_match in enumerate(result.segment_matches):
            start_t = max(0, segment_match.start_time)
            end_t = min(total_duration, segment_match.end_time)
            if end_t - start_t < 3.0:
                continue

            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            segment_audio = y[start_sample:end_sample]

            original_name = os.path.splitext(os.path.basename(fs_path))[0]
            seg_filename = f"{original_name}_seg{idx:03d}_{reference_audio}_{int(start_t)}s.wav"
            seg_path = os.path.join(user_slice_dir, seg_filename)
            sf.write(seg_path, segment_audio, sr)

            spec_path = seg_path.replace('.wav', '.png')
            generate_spectrogram_image(seg_path, spec_path)

            file_hash = hashlib.md5(f"{fs_path}_{start_t}_{end_t}".encode()).hexdigest()[:12]
            segments.append({
                "segment_id": f"seg_{file_hash}",
                "original_filename": os.path.basename(fs_path),
                "segment_filename": seg_filename,
                "duration": round(end_t - start_t, 2),
                "sample_rate": sr,
                "file_path": get_url_path(seg_path),
                "spectrogram_path": get_url_path(spec_path),
                "reference_audio": reference_audio,
                "start_time": round(start_t, 2),
                "end_time": round(end_t, 2),
                "label": None,
                "annotated_at": None
            })

        elapsed_time = (time.time() - start_time) * 1000
        return UploadAndSplitResponse(
            success=True,
            message=f"切分完成，共生成 {len(segments)} 个片段",
            segments=segments,
            reference_audio=reference_audio
        )

    except Exception as e:
        log_operation("MANUAL_SPLIT_ERROR", str(e), "ERROR")
        raise HTTPException(status_code=500, detail=f"切分失败: {str(e)}")


@router.get("/segments/{reference_audio}", response_model=List[AudioSegmentInfo])
# ⚠️ V1 遗留 — 前端已切换到 V2 会话管理，不再调用此端点
async def get_segments_by_reference(
    reference_audio: str,
    label_filter: Optional[str] = Query(None, description="按标签过滤: normal, anomaly, unlabeled")
):
    """
    获取指定参考音频下的所有音频片段
    包括已标注到数据集的片段和临时切分片段

    - **reference_audio**: 参考音频名称
    - **label_filter**: 可选，按标签过滤 (normal, anomaly, unlabeled)
    """
    log_operation("GET_SEGMENTS", f"参考音频: {reference_audio}, 过滤: {label_filter}")

    segments = []

    # 1. 获取已标注到数据集的片段
    category_paths = ensure_category_structure(reference_audio)

    # 扫描训练集
    train_good_dir = category_paths["train_good"]
    if os.path.exists(train_good_dir):
        for filename in os.listdir(train_good_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(train_good_dir, filename)
                if label_filter in (None, "normal"):
                    # 检查时频图是否存在
                    spectrogram_path = file_path.replace('.wav', '.png')
                    segments.append(AudioSegmentInfo(
                        segment_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                        original_filename=filename,
                        segment_filename=filename,
                        duration=round(librosa.get_duration(path=file_path), 2),
                        sample_rate=22050,
                        file_path=get_url_path(file_path),
                        spectrogram_path=get_url_path(spectrogram_path) if os.path.exists(spectrogram_path) else None,
                        reference_audio=reference_audio,
                        start_time=0.0,
                        end_time=0.0,
                        label="normal",
                        annotated_at=datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    ))

    # 扫描测试集 - 正常
    test_good_dir = category_paths["test_good"]
    if os.path.exists(test_good_dir):
        for filename in os.listdir(test_good_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(test_good_dir, filename)
                if label_filter in (None, "normal"):
                    # 检查时频图是否存在
                    spectrogram_path = file_path.replace('.wav', '.png')
                    segments.append(AudioSegmentInfo(
                        segment_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                        original_filename=filename,
                        segment_filename=filename,
                        duration=round(librosa.get_duration(path=file_path), 2),
                        sample_rate=22050,
                        file_path=get_url_path(file_path),
                        spectrogram_path=get_url_path(spectrogram_path) if os.path.exists(spectrogram_path) else None,
                        reference_audio=reference_audio,
                        start_time=0.0,
                        end_time=0.0,
                        label="normal",
                        annotated_at=datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    ))

    # 扫描测试集 - 异常 (anomaly 目录)
    test_anomaly_dir = category_paths["test_anomaly"]
    if os.path.exists(test_anomaly_dir):
        for filename in os.listdir(test_anomaly_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(test_anomaly_dir, filename)
                if label_filter in (None, "anomaly"):
                    # 检查时频图是否存在
                    spectrogram_path = file_path.replace('.wav', '.png')
                    segments.append(AudioSegmentInfo(
                        segment_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                        original_filename=filename,
                        segment_filename=filename,
                        duration=round(librosa.get_duration(path=file_path), 2),
                        sample_rate=22050,
                        file_path=get_url_path(file_path),
                        spectrogram_path=get_url_path(spectrogram_path) if os.path.exists(spectrogram_path) else None,
                        reference_audio=reference_audio,
                        start_time=0.0,
                        end_time=0.0,
                        label="anomaly",
                        annotated_at=datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    ))

    # 扫描测试集 - 异常 (bad 目录，兼容旧数据)
    test_bad_dir = os.path.join(category_paths["category_path"], "test", "bad")
    if os.path.exists(test_bad_dir):
        for filename in os.listdir(test_bad_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(test_bad_dir, filename)
                if label_filter in (None, "anomaly"):
                    # 检查时频图是否存在
                    spectrogram_path = file_path.replace('.wav', '.png')
                    segments.append(AudioSegmentInfo(
                        segment_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                        original_filename=filename,
                        segment_filename=filename,
                        duration=round(librosa.get_duration(path=file_path), 2),
                        sample_rate=22050,
                        file_path=get_url_path(file_path),
                        spectrogram_path=get_url_path(spectrogram_path) if os.path.exists(spectrogram_path) else None,
                        reference_audio=reference_audio,
                        start_time=0.0,
                        end_time=0.0,
                        label="anomaly",
                        annotated_at=datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    ))

    # 2. 获取临时切分但未标注的片段（只获取与当前参考音频相关的）
    if os.path.exists(SLICE_TEMP_DIR) and label_filter in (None, "unlabeled"):
        for filename in os.listdir(SLICE_TEMP_DIR):
            if filename.endswith('.wav'):
                # 只获取与当前参考音频相关的片段（文件名中包含参考音频名称）
                if reference_audio not in filename:
                    continue
                file_path = os.path.join(SLICE_TEMP_DIR, filename)
                # 检查是否已经在数据集中
                already_in_dataset = any(s.file_path == get_url_path(file_path) for s in segments)
                if not already_in_dataset:
                    try:
                        duration = round(librosa.get_duration(path=file_path), 2)
                        # 检查时频图是否存在
                        spectrogram_path = file_path.replace('.wav', '.png')
                        segments.append(AudioSegmentInfo(
                            segment_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                            original_filename=filename,
                            segment_filename=filename,
                            duration=duration,
                            sample_rate=22050,
                            file_path=get_url_path(file_path),
                            spectrogram_path=get_url_path(spectrogram_path) if os.path.exists(spectrogram_path) else None,
                            reference_audio=reference_audio,
                            start_time=0.0,
                            end_time=0.0,
                            label=None,
                            annotated_at=None
                        ))
                    except Exception as e:
                        log_operation("SEGMENT_LOAD_ERROR", f"加载片段失败 {filename}: {e}", "ERROR")

    return segments


def _get_segments_for_category(category: str, label_filter: Optional[str] = None) -> List[AudioSegmentInfo]:
    """
    同步获取指定类别下的所有音频片段
    用于内部调用
    """
    segments = []

    # 获取已标注到数据集的片段
    category_paths = ensure_category_structure(category)

    # 扫描训练集
    train_good_dir = category_paths["train_good"]
    if os.path.exists(train_good_dir):
        for filename in os.listdir(train_good_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(train_good_dir, filename)
                if label_filter in (None, "normal"):
                    try:
                        spectrogram_path = file_path.replace('.wav', '.png')
                        segments.append(AudioSegmentInfo(
                            segment_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                            original_filename=filename,
                            segment_filename=filename,
                            duration=round(librosa.get_duration(path=file_path), 2),
                            sample_rate=22050,
                            file_path=get_url_path(file_path),
                            spectrogram_path=get_url_path(spectrogram_path) if os.path.exists(spectrogram_path) else None,
                            reference_audio=category,
                            start_time=0.0,
                            end_time=0.0,
                            label="normal",
                            annotated_at=datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                        ))
                    except Exception:
                        pass

    # 扫描测试集 - 正常
    test_good_dir = category_paths["test_good"]
    if os.path.exists(test_good_dir):
        for filename in os.listdir(test_good_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(test_good_dir, filename)
                if label_filter in (None, "normal"):
                    try:
                        spectrogram_path = file_path.replace('.wav', '.png')
                        segments.append(AudioSegmentInfo(
                            segment_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                            original_filename=filename,
                            segment_filename=filename,
                            duration=round(librosa.get_duration(path=file_path), 2),
                            sample_rate=22050,
                            file_path=get_url_path(file_path),
                            spectrogram_path=get_url_path(spectrogram_path) if os.path.exists(spectrogram_path) else None,
                            reference_audio=category,
                            start_time=0.0,
                            end_time=0.0,
                            label="normal",
                            annotated_at=datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                        ))
                    except Exception:
                        pass

    # 扫描测试集 - 异常 (anomaly 目录)
    test_anomaly_dir = category_paths["test_anomaly"]
    if os.path.exists(test_anomaly_dir):
        for filename in os.listdir(test_anomaly_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(test_anomaly_dir, filename)
                if label_filter in (None, "anomaly"):
                    try:
                        spectrogram_path = file_path.replace('.wav', '.png')
                        segments.append(AudioSegmentInfo(
                            segment_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                            original_filename=filename,
                            segment_filename=filename,
                            duration=round(librosa.get_duration(path=file_path), 2),
                            sample_rate=22050,
                            file_path=get_url_path(file_path),
                            spectrogram_path=get_url_path(spectrogram_path) if os.path.exists(spectrogram_path) else None,
                            reference_audio=category,
                            start_time=0.0,
                            end_time=0.0,
                            label="anomaly",
                            annotated_at=datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                        ))
                    except Exception:
                        pass

    # 扫描测试集 - 异常 (bad 目录，兼容旧数据)
    test_bad_dir = os.path.join(category_paths["category_path"], "test", "bad")
    if os.path.exists(test_bad_dir):
        for filename in os.listdir(test_bad_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(test_bad_dir, filename)
                if label_filter in (None, "anomaly"):
                    try:
                        spectrogram_path = file_path.replace('.wav', '.png')
                        segments.append(AudioSegmentInfo(
                            segment_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                            original_filename=filename,
                            segment_filename=filename,
                            duration=round(librosa.get_duration(path=file_path), 2),
                            sample_rate=22050,
                            file_path=get_url_path(file_path),
                            spectrogram_path=get_url_path(spectrogram_path) if os.path.exists(spectrogram_path) else None,
                            reference_audio=category,
                            start_time=0.0,
                            end_time=0.0,
                            label="anomaly",
                            annotated_at=datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                        ))
                    except Exception:
                        pass

    return segments


@router.get("/segments", response_model=List[AudioSegmentInfo])
# ⚠️ V1 遗留 — 前端已切换到 V2 会话管理，不再调用此端点
async def get_all_segments(
    label_filter: Optional[str] = Query(None, description="按标签过滤: normal, anomaly, unlabeled")
):
    """
    获取所有音频片段（包括所有类别）
    用于前端显示全部片段列表

    - **label_filter**: 可选，按标签过滤 (normal, anomaly, unlabeled)
    """
    log_operation("GET_ALL_SEGMENTS", f"过滤: {label_filter}")

    all_segments = []

    # 1. 从数据集目录获取所有已标注的片段
    if os.path.exists(DATASET_ROOT):
        for category in os.listdir(DATASET_ROOT):
            category_path = os.path.join(DATASET_ROOT, category)
            if not os.path.isdir(category_path) or category == "split_log.jsonl":
                continue

            try:
                category_segments = _get_segments_for_category(category, label_filter)
                all_segments.extend(category_segments)
            except Exception as e:
                log_operation("GET_CATEGORY_SEGMENTS_ERROR", f"类别 {category}: {e}", "ERROR")

    # 2. 获取临时目录中的未标注片段
    if os.path.exists(SLICE_TEMP_DIR) and label_filter in (None, "unlabeled"):
        for filename in os.listdir(SLICE_TEMP_DIR):
            if filename.endswith('.wav'):
                file_path = os.path.join(SLICE_TEMP_DIR, filename)
                try:
                    # 从文件名中提取参考音频名称
                    # 格式: {original}_seg{idx}_{matched_name}_{start_time}s.wav
                    parts = filename.rsplit('_', 2)
                    if len(parts) >= 3:
                        reference_audio = parts[-2]  # 倒数第二部分是参考音频名称
                    else:
                        reference_audio = "未知"

                    duration = round(librosa.get_duration(path=file_path), 2)
                    # 检查时频图是否存在
                    spectrogram_path = file_path.replace('.wav', '.png')
                    all_segments.append(AudioSegmentInfo(
                        segment_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                        original_filename=filename,
                        segment_filename=filename,
                        duration=duration,
                        sample_rate=22050,
                        file_path=get_url_path(file_path),
                        spectrogram_path=get_url_path(spectrogram_path) if os.path.exists(spectrogram_path) else None,
                        reference_audio=reference_audio,
                        start_time=0.0,
                        end_time=0.0,
                        label=None,
                        annotated_at=None
                    ))
                except Exception as e:
                    log_operation("TEMP_SEGMENT_LOAD_ERROR", f"加载临时片段失败 {filename}: {e}", "ERROR")

    log_operation("GET_ALL_SEGMENTS_SUCCESS", f"共 {len(all_segments)} 个片段")
    return all_segments


@router.post("/annotate/batch")
# ⚠️ V1 遗留 — 前端已切换到 V2 会话管理标注
async def batch_annotate_segments(request: BatchAnnotationRequest):
    """
    批量标注音频片段

    - **annotations**: 标注列表，每个元素包含:
        - segment_id: 片段ID
        - label: 标签 (normal 或 anomaly)
        - category: 类别名称
        - segment_path: 音频片段路径
    """
    start_time = time.time()
    log_operation("BATCH_ANNOTATE_START", f"批量标注 {len(request.annotations)} 个片段")

    success_count = 0
    failed_items = []

    for item in request.annotations:
        try:
            segment_id = item.get("segment_id")
            label = item.get("label")
            category = item.get("category")
            segment_path = item.get("segment_path")

            # 调用单条标注逻辑
            result = await _annotate_single_segment(
                segment_id=segment_id,
                label=label,
                category=category,
                segment_path=segment_path
            )

            if result["success"]:
                success_count += 1
            else:
                failed_items.append({"segment_id": segment_id, "reason": result.get("message", "未知错误")})

        except Exception as e:
            failed_items.append({"segment_id": item.get("segment_id", "unknown"), "reason": str(e)})

    elapsed_time = (time.time() - start_time) * 1000
    log_operation("BATCH_ANNOTATE_COMPLETE",
                  f"成功: {success_count}/{len(request.annotations)}, 耗时: {elapsed_time:.2f}ms")

    return BatchAnnotationResponse(
        success=success_count == len(request.annotations),
        message=f"批量标注完成：成功 {success_count} 个，失败 {len(failed_items)} 个",
        total=len(request.annotations),
        success_count=success_count,
        failed_count=len(failed_items),
        failed_items=failed_items
    )


async def _annotate_single_segment(
    segment_id: str,
    label: str,
    category: str,
    segment_path: str
) -> Dict[str, Any]:
    """单条标注的内部实现"""
    try:
        # 验证标签
        if label not in ("normal", "anomaly"):
            return {"success": False, "message": "标签必须是 'normal' 或 'anomaly'"}

        # 将 URL 路径转换为文件系统路径
        file_path = get_file_path(segment_path)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {"success": False, "message": f"音频片段不存在: {file_path}"}

        # 确保目录结构存在
        paths = ensure_category_structure(category)

        # 获取当前数据集统计
        train_normal_count = len([f for f in os.listdir(paths["train_good"]) if f.endswith('.wav')])
        test_normal_count = len([f for f in os.listdir(paths["test_good"]) if f.endswith('.wav')])

        # 决定放入训练集还是测试集
        if label == "anomaly":
            split_type = "test"
            target_dir = paths["test_anomaly"]
        else:
            split_type = split_train_test(train_normal_count, test_normal_count)
            target_dir = paths["train_good"] if split_type == "train" else paths["test_good"]

        # 生成目标文件名
        filename = os.path.basename(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{filename}"
        target_path = os.path.join(target_dir, new_filename)

        # 复制文件到目标目录
        shutil.copy2(file_path, target_path)

        # 处理时频图
        source_spectrogram = file_path.replace('.wav', '.png')
        target_spectrogram = target_path.replace('.wav', '.png')
        if os.path.exists(source_spectrogram):
            shutil.copy2(source_spectrogram, target_spectrogram)
        else:
            generate_spectrogram_image(target_path, target_spectrogram)

        # 记录划分日志
        log_dataset_split(category, "add", new_filename, split_type, label)

        # 删除临时文件
        if file_path.startswith(SLICE_TEMP_DIR) and os.path.exists(file_path):
            os.remove(file_path)

        return {
            "success": True,
            "message": f"标注成功！已添加到 {split_type} 集的 {label} 类别",
            "segment": {
                "segment_id": segment_id,
                "segment_filename": new_filename,
                "file_path": get_url_path(target_path),
                "label": label
            }
        }

    except Exception as e:
        return {"success": False, "message": f"标注失败: {str(e)}"}


@router.get("/annotation-stats", response_model=AnnotationStats)
# ⚠️ V1 遗留 — 前端不再调用
async def get_annotation_stats():
    """获取标注统计信息"""
    log_operation("GET_ANNOTATION_STATS", "获取标注统计")

    try:
        total_normal = 0
        total_anomaly = 0
        total_unlabeled = 0
        by_category = {}
        recent_annotations = []

        # 统计已标注数据
        if os.path.exists(DATASET_ROOT):
            for category in os.listdir(DATASET_ROOT):
                category_path = os.path.join(DATASET_ROOT, category)
                if not os.path.isdir(category_path) or category == "split_log.jsonl":
                    continue

                # 初始化类别统计
                by_category[category] = {"normal": 0, "anomaly": 0, "unlabeled": 0}

                # 统计训练集正常数据
                train_good_dir = os.path.join(category_path, "train", "good")
                if os.path.exists(train_good_dir):
                    count = len([f for f in os.listdir(train_good_dir) if f.endswith('.wav')])
                    by_category[category]["normal"] += count
                    total_normal += count

                # 统计测试集正常数据
                test_good_dir = os.path.join(category_path, "test", "good")
                if os.path.exists(test_good_dir):
                    count = len([f for f in os.listdir(test_good_dir) if f.endswith('.wav')])
                    by_category[category]["normal"] += count
                    total_normal += count

                # 统计测试集异常数据
                test_anomaly_dir = os.path.join(category_path, "test", "anomaly")
                if os.path.exists(test_anomaly_dir):
                    count = len([f for f in os.listdir(test_anomaly_dir) if f.endswith('.wav')])
                    by_category[category]["anomaly"] += count
                    total_anomaly += count

        # 统计未标注数据（临时目录）
        if os.path.exists(SLICE_TEMP_DIR):
            for filename in os.listdir(SLICE_TEMP_DIR):
                if filename.endswith('.wav'):
                    total_unlabeled += 1
                    # 尝试从文件名提取类别
                    parts = filename.rsplit('_', 2)
                    if len(parts) >= 3:
                        category = parts[-2]
                        if category not in by_category:
                            by_category[category] = {"normal": 0, "anomaly": 0, "unlabeled": 0}
                        by_category[category]["unlabeled"] += 1

        # 读取最近的标注日志
        log_file = os.path.join(DATASET_ROOT, "split_log.jsonl")
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # 取最近20条
                    for line in reversed(lines[-20:]):
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("operation") == "add":
                                recent_annotations.append({
                                    "timestamp": entry.get("timestamp"),
                                    "category": entry.get("category"),
                                    "filename": entry.get("file_name"),
                                    "split_type": entry.get("split_type"),
                                    "label": entry.get("label")
                                })
                        except:
                            continue
            except Exception:
                pass

        return AnnotationStats(
            total_annotated=total_normal + total_anomaly,
            normal_count=total_normal,
            anomaly_count=total_anomaly,
            unlabeled_count=total_unlabeled,
            by_category=by_category,
            recent_annotations=recent_annotations[:10]  # 最近10条
        )

    except Exception as e:
        log_operation("GET_ANNOTATION_STATS_ERROR", str(e), "ERROR")
        raise HTTPException(status_code=500, detail=f"获取标注统计失败: {str(e)}")


@router.post("/annotate", response_model=AnnotationResponse)
# ⚠️ V1 遗留 — 前端已切换到 V2 会话管理标注
async def annotate_segment(
    segment_id: str = Form(...),
    label: str = Form(...),
    category: str = Form(...),
    segment_path: str = Form(...)
):
    """
    为音频片段添加标注，并根据划分规则存入训练集或测试集

    - **segment_id**: 片段ID
    - **label**: 标签 (normal 或 anomaly)
    - **category**: 类别名称（参考音频名称）
    - **segment_path**: 音频片段文件路径
    """
    start_time = time.time()
    log_operation("ANNOTATE_START", f"片段: {segment_id}, 标签: {label}, 类别: {category}")

    # 验证标签
    if label not in ("normal", "anomaly"):
        raise HTTPException(status_code=400, detail="标签必须是 'normal' 或 'anomaly'")

    # 将 URL 路径转换为文件系统路径
    file_path = get_file_path(segment_path)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"音频片段不存在: {file_path} (原始路径: {segment_path})")

    # 使用文件系统路径进行后续操作
    segment_path = file_path

    try:
        # 确保目录结构存在
        paths = ensure_category_structure(category)

        # 获取当前数据集统计
        train_normal_count = len([f for f in os.listdir(paths["train_good"]) if f.endswith('.wav')])
        test_normal_count = len([f for f in os.listdir(paths["test_good"]) if f.endswith('.wav')])
        test_anomaly_count = len([f for f in os.listdir(paths["test_anomaly"]) if f.endswith('.wav')])

        # 决定放入训练集还是测试集
        # 异常数据只能放入测试集
        if label == "anomaly":
            split_type = "test"
            target_dir = paths["test_anomaly"]
        else:
            # 正常数据按照 10:1 比例划分
            split_type = split_train_test(train_normal_count, test_normal_count)
            target_dir = paths["train_good"] if split_type == "train" else paths["test_good"]

        # 生成目标文件名
        filename = os.path.basename(segment_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{filename}"
        target_path = os.path.join(target_dir, new_filename)

        # 复制文件到目标目录
        shutil.copy2(segment_path, target_path)

        # 处理时频图
        source_spectrogram = segment_path.replace('.wav', '.png')
        target_spectrogram = target_path.replace('.wav', '.png')
        if os.path.exists(source_spectrogram):
            # 如果源文件有时频图，复制它
            shutil.copy2(source_spectrogram, target_spectrogram)
            log_operation("SPECTROGRAM_COPIED", f"复制时频图: {os.path.basename(target_spectrogram)}")
        else:
            # 否则生成新的时频图
            generate_spectrogram_image(target_path, target_spectrogram)

        # 记录划分日志
        log_dataset_split(category, "add", new_filename, split_type, label)

        elapsed_time = (time.time() - start_time) * 1000
        log_operation("ANNOTATE_SUCCESS", f"文件: {new_filename}, 划分: {split_type}, 耗时 {elapsed_time:.2f}ms")

        # 删除临时文件
        if segment_path.startswith(SLICE_TEMP_DIR) and os.path.exists(segment_path):
            os.remove(segment_path)

        return AnnotationResponse(
            success=True,
            message=f"标注成功！音频已添加到 {split_type} 集的 {label} 类别",
            segment=AudioSegmentInfo(
                segment_id=segment_id,
                original_filename=filename,
                segment_filename=new_filename,
                duration=round(librosa.get_duration(path=target_path), 2),
                sample_rate=22050,
                file_path=get_url_path(target_path),
                spectrogram_path=get_url_path(target_spectrogram),
                reference_audio=category,
                start_time=0.0,
                end_time=0.0,
                label=label,
                annotated_at=datetime.now().isoformat()
            )
        )

    except Exception as e:
        log_operation("ANNOTATE_ERROR", str(e), "ERROR")
        raise HTTPException(status_code=500, detail=f"标注失败: {str(e)}")


@router.delete("/segment/{category}/{split_type}/{filename}")
async def delete_segment(
    category: str,
    split_type: str,  # train 或 test
    filename: str,
    label: Optional[str] = None  # normal 或 anomaly，仅test集需要
):
    """
    删除数据集中的音频片段

    - **category**: 类别名称
    - **split_type**: train 或 test
    - **filename**: 文件名
    - **label**: 标签（仅test集需要，normal 或 anomaly）
    """
    log_operation("DELETE_START", f"类别: {category}, 划分: {split_type}, 文件: {filename}")

    try:
        paths = ensure_category_structure(category)

        if split_type == "train":
            file_path = os.path.join(paths["train_good"], filename)
        else:
            if not label:
                raise HTTPException(status_code=400, detail="删除测试集文件需要提供label参数")
            if label == "normal":
                file_path = os.path.join(paths["test_good"], filename)
            else:
                file_path = os.path.join(paths["test_anomaly"], filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")

        # 删除音频文件
        os.remove(file_path)

        # 删除对应的时频图
        spectrogram_path = file_path.replace('.wav', '.png')
        if os.path.exists(spectrogram_path):
            os.remove(spectrogram_path)

        # 记录日志
        log_dataset_split(category, "delete", filename, split_type, label or "normal")

        log_operation("DELETE_SUCCESS", f"已删除: {filename}")

        return {"success": True, "message": f"文件 {filename} 已删除"}

    except HTTPException:
        raise
    except Exception as e:
        log_operation("DELETE_ERROR", str(e), "ERROR")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.get("/stats", response_model=DatasetStats)
# ✅ 仍被前端 refreshStats() 使用
async def get_dataset_stats():
    """
    获取数据集统计信息
    包括总文件数、各类别分布、训练/测试集比例等
    """
    log_operation("GET_STATS", "获取数据集统计信息")

    try:
        categories = []
        total_train = 0
        total_test = 0
        total_train_normal = 0
        total_test_normal = 0
        total_test_anomaly = 0

        # 遍历所有类别
        if os.path.exists(DATASET_ROOT):
            for category in os.listdir(DATASET_ROOT):
                category_path = os.path.join(DATASET_ROOT, category)
                if not os.path.isdir(category_path) or category == "split_log.jsonl":
                    continue

                # 统计各类别数量
                train_good_dir = os.path.join(category_path, "train", "good")
                test_good_dir = os.path.join(category_path, "test", "good")
                test_anomaly_dir = os.path.join(category_path, "test", "anomaly")
                test_bad_dir = os.path.join(category_path, "test", "bad")  # 兼容旧数据

                train_normal_count = len([f for f in os.listdir(train_good_dir) if f.endswith('.wav')]) if os.path.exists(train_good_dir) else 0
                test_normal_count = len([f for f in os.listdir(test_good_dir) if f.endswith('.wav')]) if os.path.exists(test_good_dir) else 0
                test_anomaly_count = len([f for f in os.listdir(test_anomaly_dir) if f.endswith('.wav')]) if os.path.exists(test_anomaly_dir) else 0
                test_bad_count = len([f for f in os.listdir(test_bad_dir) if f.endswith('.wav')]) if os.path.exists(test_bad_dir) else 0
                # 将 bad 目录的统计合并到 anomaly 中
                test_anomaly_count += test_bad_count

                category_info = DatasetCategoryInfo(
                    category_name=category,
                    train_normal_count=train_normal_count,
                    test_normal_count=test_normal_count,
                    test_anomaly_count=test_anomaly_count,
                    total_count=train_normal_count + test_normal_count + test_anomaly_count
                )
                categories.append(category_info)

                total_train += train_normal_count
                total_test += test_normal_count + test_anomaly_count
                total_train_normal += train_normal_count
                total_test_normal += test_normal_count
                total_test_anomaly += test_anomaly_count

        # 按类别名称排序
        categories.sort(key=lambda x: x.category_name)

        stats = DatasetStats(
            total_categories=len(categories),
            total_audio_files=total_train + total_test,
            train_total=total_train,
            test_total=total_test,
            train_normal=total_train_normal,
            test_normal=total_test_normal,
            test_anomaly=total_test_anomaly,
            categories=categories
        )

        log_operation("GET_STATS_SUCCESS", f"类别数: {len(categories)}, 总文件数: {stats.total_audio_files}")

        return stats

    except Exception as e:
        log_operation("GET_STATS_ERROR", str(e), "ERROR")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/stats/{category}", response_model=DatasetCategoryInfo)
async def get_category_stats(category: str):
    """
    获取指定类别的统计信息

    - **category**: 类别名称
    """
    log_operation("GET_CATEGORY_STATS", f"类别: {category}")

    try:
        paths = ensure_category_structure(category)

        train_normal_count = len([f for f in os.listdir(paths["train_good"]) if f.endswith('.wav')])
        test_normal_count = len([f for f in os.listdir(paths["test_good"]) if f.endswith('.wav')])
        test_anomaly_count = len([f for f in os.listdir(paths["test_anomaly"]) if f.endswith('.wav')])

        # 兼容旧数据：统计 bad 目录
        test_bad_dir = os.path.join(paths["category_path"], "test", "bad")
        if os.path.exists(test_bad_dir):
            test_bad_count = len([f for f in os.listdir(test_bad_dir) if f.endswith('.wav')])
            test_anomaly_count += test_bad_count

        return DatasetCategoryInfo(
            category_name=category,
            train_normal_count=train_normal_count,
            test_normal_count=test_normal_count,
            test_anomaly_count=test_anomaly_count,
            total_count=train_normal_count + test_normal_count + test_anomaly_count
        )

    except Exception as e:
        log_operation("GET_CATEGORY_STATS_ERROR", str(e), "ERROR")
        raise HTTPException(status_code=500, detail=f"获取类别统计失败: {str(e)}")


@router.get("/split-log", response_model=List[DatasetSplitLog])
async def get_split_logs(
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    获取数据集划分日志

    - **category**: 可选，按类别过滤
    - **limit**: 返回的最大日志条数，默认100
    """
    log_file = os.path.join(DATASET_ROOT, "split_log.jsonl")

    if not os.path.exists(log_file):
        return []

    logs = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if category is None or entry.get("category") == category:
                        logs.append(DatasetSplitLog(**entry))
                except:
                    continue

        # 按时间倒序排列，限制数量
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return logs[:limit]

    except Exception as e:
        log_operation("GET_LOGS_ERROR", str(e), "ERROR")
        return []


@router.post("/cleanup-temp")
async def cleanup_temp_files():
    """清理临时切分文件"""
    log_operation("CLEANUP_TEMP", "清理临时文件")

    try:
        count = 0
        if os.path.exists(SLICE_TEMP_DIR):
            for filename in os.listdir(SLICE_TEMP_DIR):
                file_path = os.path.join(SLICE_TEMP_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        count += 1
                except:
                    pass

        log_operation("CLEANUP_TEMP_SUCCESS", f"清理了 {count} 个临时文件")
        return {"success": True, "message": f"已清理 {count} 个临时文件"}

    except Exception as e:
        log_operation("CLEANUP_TEMP_ERROR", str(e), "ERROR")
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")
