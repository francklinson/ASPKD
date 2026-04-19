"""
数据集构建 API
用于构建和管理音频异常检测数据集
支持音频上传、切分、标注、数据集划分和统计
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
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "spk")
os.makedirs(DATASET_ROOT, exist_ok=True)

# 临时上传目录
UPLOAD_TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads", "dataset_temp")
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)

# 切分后音频临时存储目录
SLICE_TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads", "dataset_slices")
os.makedirs(SLICE_TEMP_DIR, exist_ok=True)

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


class AnnotationResponse(BaseModel):
    """标注响应"""
    success: bool
    message: str
    segment: Optional[AudioSegmentInfo] = None


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
        from core.shazam import AudioFingerprinter
        with AudioFingerprinter() as fp:
            references = fp.get_all_references()
            return references
    except Exception as e:
        log_operation("GET_REFERENCES_ERROR", str(e), "ERROR")
        return []


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
        from core.long_audio_analyzer import LongAudioAnalyzer, AnalyzerConfig
        from core.shazam.database.connector import MySQLConnector
        from core.shazam.utils.hparam import hp

        # 创建数据库连接（MySQLConnector 从全局 hp 配置读取数据库信息）
        db_connector = MySQLConnector()

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
                "file_path": segment_path,
                "spectrogram_path": spectrogram_path,
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


def generate_spectrogram_image(audio_path: str, output_path: str) -> bool:
    """生成音频的时频图"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt

        # 加载音频
        y, sr = librosa.load(audio_path, sr=22050)

        # 归一化
        y = librosa.util.normalize(y)

        # 计算STFT
        D = librosa.stft(y)
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # 绘制时频图
        plt.figure(figsize=(6.3, 6.3))
        librosa.display.specshow(DB, sr=sr)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=False)
        plt.close()

        return True
    except Exception as e:
        log_operation("SPECTROGRAM_ERROR", f"生成时频图失败: {str(e)}", "ERROR")
        return False


# ========== API 端点 ==========

@router.get("/references", response_model=List[Dict[str, Any]])
async def get_available_references():
    """
    获取所有可用的参考音频列表
    用于数据集构建时选择参考音频类别
    """
    log_operation("GET_REFERENCES", "获取参考音频列表")
    references = get_reference_audios()
    return references


@router.post("/upload-and-split", response_model=UploadAndSplitResponse)
async def upload_and_split_audio(
    file: UploadFile = File(...)
):
    """
    上传音频文件并使用Shazam自动匹配参考音频进行切分

    - **file**: 音频文件 (支持 wav, mp3, flac 等格式)
    
    系统会自动从参考音频库中匹配最合适的参考音频，并使用Shazam音频指纹算法进行切分。
    """
    start_time = time.time()
    log_operation("UPLOAD_SPLIT_START", f"文件: {file.filename}, 使用Shazam自动匹配")

    # 验证文件格式
    allowed_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式，仅支持 {allowed_extensions}"
        )

    temp_file_path = None

    try:
        # 保存上传的文件到临时目录
        temp_filename = f"{int(time.time())}_{file.filename}"
        temp_file_path = os.path.join(UPLOAD_TEMP_DIR, temp_filename)

        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        log_operation("UPLOAD_SAVED", f"临时文件: {temp_file_path}")

        # 使用Shazam自动匹配并切分音频
        segments = split_audio_auto_match(
            audio_path=temp_file_path,
            output_dir=SLICE_TEMP_DIR
        )

        if not segments:
            return UploadAndSplitResponse(
                success=False,
                message="未能在音频中找到匹配的参考片段，请检查音频内容或确认参考音频库中已添加相应的参考音频",
                segments=[],
                reference_audio=""
            )

        # 获取匹配到的参考音频名称（所有片段应该都是同一个参考音频）
        matched_reference = segments[0]["reference_audio"] if segments else ""

        elapsed_time = (time.time() - start_time) * 1000
        log_operation("UPLOAD_SPLIT_SUCCESS", f"生成 {len(segments)} 个片段, 匹配参考音频: {matched_reference}, 耗时 {elapsed_time:.2f}ms")

        return UploadAndSplitResponse(
            success=True,
            message=f"音频上传并切分成功，共生成 {len(segments)} 个片段，自动匹配参考音频: {matched_reference}",
            segments=segments,
            reference_audio=matched_reference
        )

    except HTTPException:
        raise
    except Exception as e:
        log_operation("UPLOAD_SPLIT_ERROR", str(e), "ERROR")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    finally:
        # 清理临时上传文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


@router.get("/segments/{reference_audio}", response_model=List[AudioSegmentInfo])
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
                        file_path=file_path,
                        spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
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
                        file_path=file_path,
                        spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
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
                        file_path=file_path,
                        spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
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
                        file_path=file_path,
                        spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
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
                already_in_dataset = any(s.file_path == file_path for s in segments)
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
                            file_path=file_path,
                            spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
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
                            file_path=file_path,
                            spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
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
                            file_path=file_path,
                            spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
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
                            file_path=file_path,
                            spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
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
                            file_path=file_path,
                            spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
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
                        file_path=file_path,
                        spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
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


@router.post("/annotate", response_model=AnnotationResponse)
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

    # 检查文件是否存在
    if not os.path.exists(segment_path):
        raise HTTPException(status_code=404, detail=f"音频片段不存在: {segment_path}")

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
                file_path=target_path,
                spectrogram_path=target_spectrogram,
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
