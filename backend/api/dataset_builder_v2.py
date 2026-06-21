"""
数据集构建 API V2
支持两种数据来源：
1. 在线检测生成的数据 - 保留模型预测值作为预标注
2. 用户手动上传的数据 - 可调用模型预测或保留原样

所有数据都经过检测算法切分后进入数据集
"""
import os
import shutil
import json
import time
import uuid
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import librosa
import soundfile as sf

router = APIRouter()

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# 数据集根目录
DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "spk")
os.makedirs(DATASET_ROOT, exist_ok=True)

# 数据集构建工作区
DATASET_BUILDER_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_builder")
os.makedirs(DATASET_BUILDER_DIR, exist_ok=True)

# 检测任务结果关联目录
DETECTION_RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "output", "results")


def log_operation(operation: str, details: str = "", status: str = "INFO"):
    """记录操作日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [DatasetBuilderV2] [{status}] {operation} | {details}")


# ========== 数据模型 ==========

class DataSourceType(str):
    """数据来源类型"""
    DETECTION = "detection"  # 在线检测生成
    MANUAL = "manual"        # 手动上传


class DatasetItemStatus(str):
    """数据集条目状态"""
    PENDING = "pending"           # 待处理
    PREPROCESSING = "preprocessing"  # 预处理中
    SLICED = "sliced"             # 已切分
    PREDICTED = "predicted"       # 已预测（有预标注）
    ANNOTATED = "annotated"       # 已人工标注
    IN_DATASET = "in_dataset"     # 已在数据集中


class AudioSegmentInfo(BaseModel):
    """音频片段信息"""
    segment_id: str
    source_type: str  # detection 或 manual
    original_filename: str
    segment_filename: str
    duration: float
    sample_rate: int = 22050
    file_path: str
    spectrogram_path: Optional[str] = None
    reference_audio: str
    start_time: float
    end_time: float
    
    # 预标注信息（来自模型预测）
    predicted_label: Optional[str] = None  # "normal" 或 "anomaly"
    predicted_score: Optional[float] = None
    predicted_confidence: Optional[float] = None
    
    # 人工标注信息
    manual_label: Optional[str] = None
    annotated_by: Optional[str] = None
    annotated_at: Optional[str] = None
    
    # 状态
    status: str = DatasetItemStatus.PENDING
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # 关联信息
    detection_task_id: Optional[str] = None  # 如果是检测数据，关联的检测任务ID
    source_file_path: Optional[str] = None   # 原始文件路径


class DatasetBuildSession(BaseModel):
    """数据集构建会话"""
    session_id: str
    source_type: str  # detection 或 manual
    status: str  # pending, processing, completed, failed
    created_at: str
    completed_at: Optional[str] = None

    # 源数据信息
    source_files: List[str] = []
    detection_task_id: Optional[str] = None

    # 详细来源记录（支持多数据源合并）
    source_details: List[Dict[str, Any]] = []
    # 导入历史（用于去重，记录已导入的 task_id / result_id）
    import_history: List[str] = []

    # 处理结果
    segments: List[AudioSegmentInfo] = []
    total_segments: int = 0
    normal_count: int = 0
    anomaly_count: int = 0
    unlabeled_count: int = 0

    # 处理配置
    auto_predict: bool = False  # 是否自动预测（手动上传时）
    algorithm: Optional[str] = None  # 用于预测的算法

    # 统计信息
    processing_stats: Dict[str, Any] = {}


class CreateFromDetectionRequest(BaseModel):
    """从检测结果创建数据集请求"""
    detection_task_id: str
    auto_annotate_threshold: float = 0.8  # 自动标注置信度阈值
    include_normal: bool = True
    include_anomaly: bool = True


class CreateFromManualRequest(BaseModel):
    """从手动上传创建数据集请求"""
    session_id: str
    auto_predict: bool = False
    algorithm: Optional[str] = "dinomaly_dinov3_small"


class CreateFromClientDetectionRequest(BaseModel):
    """从客户端（在线）检测结果创建数据集请求"""
    result_ids: List[int]
    auto_annotate_threshold: float = 0.8


class BatchAnnotateRequest(BaseModel):
    """批量标注请求"""
    session_id: str
    annotations: List[Dict[str, Any]]  # segment_id, label, annotator


class DatasetBuildResponse(BaseModel):
    """数据集构建响应"""
    success: bool
    message: str
    session_id: Optional[str] = None
    segments: List[AudioSegmentInfo] = []
    stats: Dict[str, Any] = {}


class DatasetPreviewResponse(BaseModel):
    """数据集预览响应"""
    session_id: str
    segments: List[AudioSegmentInfo]
    total_count: int
    normal_count: int
    anomaly_count: int
    unlabeled_count: int
    by_reference: Dict[str, int]


class ConfirmToDatasetRequest(BaseModel):
    """确认导入数据集请求"""
    session_id: str
    segment_ids: Optional[List[str]] = None  # None表示全部导入
    target_category: Optional[str] = None  # 指定目标类别
    reference_audios: Optional[List[str]] = None  # 指定参考音频列表（None表示全部）
    train_ratio: int = 10  # 训练集:测试集比例，默认 10:1


class ConfirmAllAnnotatedRequest(BaseModel):
    """一键确认所有已标注片段导入数据集请求"""
    reference_audios: Optional[List[str]] = None  # None 表示全部
    train_ratio: int = 10  # 训练集:测试集比例，默认 10:1


class ImportToSessionRequest(BaseModel):
    """追加导入到已有会话请求"""
    session_id: str
    source_type: str  # "detection" 或 "client_detection"
    # 当 source_type 为 "detection" 时
    detection_task_id: Optional[str] = None
    # 当 source_type 为 "client_detection" 时
    result_ids: Optional[List[int]] = None
    auto_annotate_threshold: float = 0.8
    include_normal: bool = True
    include_anomaly: bool = True


class SplitPreviewRequest(BaseModel):
    """划分预览请求"""
    session_id: str
    reference_audios: Optional[List[str]] = None  # None表示全部
    train_ratio: int = 10


class SplitImportRequest(BaseModel):
    """划分导入请求"""
    session_id: str
    reference_audios: Optional[List[str]] = None  # None表示全部
    train_ratio: int = 10
    target_category: Optional[str] = None


# ========== 会话管理 ==========

# 内存中的会话存储
BUILD_SESSIONS: Dict[str, DatasetBuildSession] = {}


def create_session(source_type: str, detection_task_id: Optional[str] = None) -> DatasetBuildSession:
    """创建新的构建会话"""
    session_id = f"dbs_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    session = DatasetBuildSession(
        session_id=session_id,
        source_type=source_type,
        status="pending",
        created_at=datetime.now().isoformat(),
        detection_task_id=detection_task_id
    )
    BUILD_SESSIONS[session_id] = session
    return session


def get_session(session_id: str) -> Optional[DatasetBuildSession]:
    """获取会话"""
    return BUILD_SESSIONS.get(session_id)


def save_session(session: DatasetBuildSession):
    """保存会话到文件（持久化）"""
    session_file = os.path.join(DATASET_BUILDER_DIR, f"{session.session_id}.json")
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session.model_dump(), f, ensure_ascii=False, indent=2)


def _load_sessions_from_disk():
    """从磁盘加载已有会话（服务重启后恢复）"""
    if not os.path.exists(DATASET_BUILDER_DIR):
        return
    loaded_count = 0
    for fname in os.listdir(DATASET_BUILDER_DIR):
        if fname.endswith('.json') and fname.startswith('dbs_'):
            fpath = os.path.join(DATASET_BUILDER_DIR, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                session = DatasetBuildSession(**data)
                BUILD_SESSIONS[session.session_id] = session
                loaded_count += 1
            except Exception as e:
                log_operation("LOAD_SESSION_ERROR", f"加载 {fname}: {e}", "WARNING")
    if loaded_count:
        log_operation("LOAD_SESSIONS", f"从磁盘恢复 {loaded_count} 个会话")


# ========== 核心功能 ==========

async def process_detection_results(
    session: DatasetBuildSession,
    request: CreateFromDetectionRequest
) -> DatasetBuildResponse:
    """
    处理在线检测结果，提取音频片段并保留预测值作为预标注
    """
    log_operation("PROCESS_DETECTION_START", f"Task: {request.detection_task_id}")

    # 去重检查
    if request.detection_task_id in session.import_history:
        return DatasetBuildResponse(
            success=False,
            message=f"检测任务 {request.detection_task_id} 已导入过，请勿重复导入"
        )

    # 获取检测结果
    from backend.core.task_manager import task_manager
    task_result = task_manager.get_task_result(request.detection_task_id)

    if not task_result:
        return DatasetBuildResponse(
            success=False,
            message=f"检测任务不存在: {request.detection_task_id}"
        )

    if task_result["status"] != "completed":
        return DatasetBuildResponse(
            success=False,
            message=f"检测任务尚未完成，当前状态: {task_result['status']}"
        )

    results = task_result.get("results", [])
    if not results:
        return DatasetBuildResponse(
            success=False,
            message="检测任务没有结果"
        )

    session.status = "processing"
    session.source_files = [r.get("filepath") for r in results if r.get("filepath")]

    # 记录导入历史
    session.import_history.append(request.detection_task_id)
    session.source_details.append({
        "type": "detection",
        "task_id": request.detection_task_id,
        "source": "离线检测",
        "file_count": len(results),
        "imported_at": datetime.now().isoformat(),
        "auto_annotate_threshold": request.auto_annotate_threshold,
        "include_normal": request.include_normal,
        "include_anomaly": request.include_anomaly
    })
    
    segments = []
    processed_count = 0
    
    for result in results:
        # 检查是否应该包含
        is_anomaly = result.get("is_anomaly", False)
        if is_anomaly and not request.include_anomaly:
            continue
        if not is_anomaly and not request.include_normal:
            continue
        
        # 获取音频切片路径
        audio_slice_path = result.get("audio_slice_path")
        if not audio_slice_path:
            continue
        
        full_audio_path = os.path.join(PROJECT_ROOT, audio_slice_path)
        if not os.path.exists(full_audio_path):
            log_operation("AUDIO_NOT_FOUND", f"Path: {full_audio_path}", "WARNING")
            continue
        
        # 创建会话专属目录
        session_dir = os.path.join(DATASET_BUILDER_DIR, session.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # 复制音频到会话目录
        segment_filename = os.path.basename(audio_slice_path)
        target_path = os.path.join(session_dir, segment_filename)
        shutil.copy2(full_audio_path, target_path)
        
        # 生成时频图
        spectrogram_path = target_path.replace('.wav', '.png')
        generate_spectrogram(target_path, spectrogram_path)
        
        # 获取音频信息
        try:
            duration = librosa.get_duration(path=target_path)
        except:
            duration = 0.0
        
        # 创建片段信息
        segment_id = f"seg_{hashlib.md5(target_path.encode()).hexdigest()[:12]}"
        
        # 确定预标注
        predicted_label = "anomaly" if is_anomaly else "normal"
        predicted_score = result.get("anomaly_score", 0.0)
        predicted_confidence = predicted_score if is_anomaly else (1 - predicted_score)
        
        # 如果置信度足够高，自动标注
        manual_label = None
        status = DatasetItemStatus.PREDICTED
        if predicted_confidence >= request.auto_annotate_threshold:
            manual_label = predicted_label
            status = DatasetItemStatus.ANNOTATED
        
        segment = AudioSegmentInfo(
            segment_id=segment_id,
            source_type=DataSourceType.DETECTION,
            original_filename=result.get("filename", "unknown"),
            segment_filename=segment_filename,
            duration=round(duration, 2),
            file_path=target_path,
            spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
            reference_audio=result.get("music_name", "unknown"),
            start_time=0.0,
            end_time=round(duration, 2),
            predicted_label=predicted_label,
            predicted_score=predicted_score,
            predicted_confidence=predicted_confidence,
            manual_label=manual_label,
            status=status,
            detection_task_id=request.detection_task_id,
            source_file_path=result.get("filepath")
        )
        
        segments.append(segment)
        processed_count += 1
    
    # 更新会话
    session.segments = segments
    session.total_segments = len(segments)
    session.normal_count = sum(1 for s in segments if s.predicted_label == "normal")
    session.anomaly_count = sum(1 for s in segments if s.predicted_label == "anomaly")
    session.unlabeled_count = sum(1 for s in segments if s.status == DatasetItemStatus.PREDICTED)

    # 更新来源信息中的段数
    if session.source_details:
        session.source_details[-1]["segment_count"] = len(segments)

    session.status = "completed"
    session.completed_at = datetime.now().isoformat()

    save_session(session)

    log_operation("PROCESS_DETECTION_COMPLETE",
                  f"Processed: {processed_count}, Total segments: {len(segments)}")

    return DatasetBuildResponse(
        success=True,
        message=f"成功处理 {processed_count} 个检测文件，生成 {len(segments)} 个片段",
        session_id=session.session_id,
        segments=segments,
        stats={
            "total": len(segments),
            "normal": session.normal_count,
            "anomaly": session.anomaly_count,
            "unlabeled": session.unlabeled_count,
            "auto_annotated": sum(1 for s in segments if s.status == DatasetItemStatus.ANNOTATED)
        }
    )


def _find_audio_slice_for_client_result(result: dict) -> Optional[str]:
    """
    从客户端检测结果中找到音频切片路径
    尝试多种推断方式：
    1. 直接取 audio_slice_path
    2. 从 overlay_path 反向推断
    """
    # 方式1：直接取
    audio_path = result.get("audio_slice_path")
    if audio_path:
        full_path = os.path.join(PROJECT_ROOT, audio_path) if not os.path.isabs(audio_path) else audio_path
        if os.path.exists(full_path):
            return full_path
        # 尝试音频路径已经存在
        if os.path.exists(audio_path):
            return audio_path

    # 方式2：从 overlay_path 推断
    overlay = result.get("overlay_path")
    if overlay:
        # overlay 可能是相对路径（如 visualize/dinomaly_xxx/xxx_overlay.png）
        # 也可能是绝对路径
        if not os.path.isabs(overlay):
            full_overlay = os.path.join(PROJECT_ROOT, overlay.replace("visualize/", "data/output/vis/"))
        else:
            full_overlay = overlay
        base_name = os.path.splitext(os.path.basename(full_overlay))[0]
        if base_name.endswith('_overlay'):
            base_name = base_name[:-8]
        # 尝试几个可能的切片目录
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "uploads", "clients", "segments", f"{base_name}.wav"),
            os.path.join(PROJECT_ROOT, "data", "output", "slices", "audio", f"{base_name}.wav"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
    return None


async def process_client_detection_results(
    session: DatasetBuildSession,
    request: CreateFromClientDetectionRequest
) -> DatasetBuildResponse:
    """
    处理客户端（在线）检测结果，提取音频片段并保留预测值作为预标注
    """
    log_operation("PROCESS_CLIENT_START", f"Result IDs: {request.result_ids}")

    from backend.core.client_monitor_service import client_detection_service

    # 去重检查：过滤已导入的结果 ID
    new_ids = [rid for rid in request.result_ids if str(rid) not in session.import_history]
    if not new_ids:
        return DatasetBuildResponse(
            success=False,
            message=f"所有选中的检测结果均已导入过，请选择其他结果"
        )
    skipped = len(request.result_ids) - len(new_ids)
    if skipped > 0:
        log_operation("DEDUP_SKIP", f"{skipped} 条结果已存在，跳过", "WARNING")

    # 按 result_id 查找结果
    all_results = list(client_detection_service.detection_results)
    matched_results = [r for r in all_results if r.get("result_id") in new_ids]

    if not matched_results:
        return DatasetBuildResponse(
            success=False,
            message=f"未找到指定的检测结果（IDs: {request.result_ids}）"
        )

    session.status = "processing"
    session.source_files = [r.get("filepath") for r in matched_results if r.get("filepath")]

    session_dir = os.path.join(DATASET_BUILDER_DIR, session.session_id)
    os.makedirs(session_dir, exist_ok=True)

    segments = []
    for result in matched_results:
        # 查找音频切片路径
        full_audio_path = _find_audio_slice_for_client_result(result)
        if not full_audio_path or not os.path.exists(full_audio_path):
            log_operation("AUDIO_NOT_FOUND", f"Result {result.get('result_id')}, file: {result.get('filename')}", "WARNING")
            continue

        # 复制音频到会话目录
        segment_filename = os.path.basename(full_audio_path)
        target_path = os.path.join(session_dir, segment_filename)
        # 如果目标已存在（多个结果指向同一切片），加后缀
        if os.path.exists(target_path):
            name, ext = os.path.splitext(segment_filename)
            segment_filename = f"{name}_{result.get('result_id')}{ext}"
            target_path = os.path.join(session_dir, segment_filename)
        shutil.copy2(full_audio_path, target_path)

        # 生成时频图
        spectrogram_path = target_path.replace('.wav', '.png')
        generate_spectrogram(target_path, spectrogram_path)

        try:
            duration = librosa.get_duration(path=target_path)
        except Exception:
            duration = 0.0

        # 预标注信息
        is_anomaly = result.get("is_anomaly", False)
        anomaly_score = result.get("anomaly_score", 0.0)
        predicted_label = "anomaly" if is_anomaly else "normal"
        predicted_confidence = anomaly_score if is_anomaly else (1 - anomaly_score)

        # 从 segment_info 获取参考音频名称
        seg_info = result.get("segment_info") or {}
        music_name = seg_info.get("music_name") or result.get("music_name", "unknown")

        # 如果置信度足够高，自动标注
        manual_label = None
        status = DatasetItemStatus.PREDICTED
        if predicted_confidence >= request.auto_annotate_threshold:
            manual_label = predicted_label
            status = DatasetItemStatus.ANNOTATED

        segment_id = f"seg_{hashlib.md5(target_path.encode()).hexdigest()[:12]}"
        segment = AudioSegmentInfo(
            segment_id=segment_id,
            source_type="detection",
            original_filename=result.get("filename", "unknown"),
            segment_filename=segment_filename,
            duration=round(duration, 2),
            file_path=target_path,
            spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
            reference_audio=music_name,
            start_time=seg_info.get("start_time", 0.0),
            end_time=seg_info.get("end_time", round(duration, 2)),
            predicted_label=predicted_label,
            predicted_score=anomaly_score,
            predicted_confidence=predicted_confidence,
            manual_label=manual_label,
            status=status,
            source_file_path=result.get("filepath")
        )
        segments.append(segment)

    # 记录导入历史
    for rid in request.result_ids:
        if str(rid) not in session.import_history:
            session.import_history.append(str(rid))

    # 更新会话
    session.segments = segments
    session.total_segments = len(segments)
    session.normal_count = sum(1 for s in segments if s.predicted_label == "normal")
    session.anomaly_count = sum(1 for s in segments if s.predicted_label == "anomaly")
    session.unlabeled_count = sum(1 for s in segments if s.status == DatasetItemStatus.PREDICTED)

    # 记录来源信息
    matched_filenames = list(set(r.get("filename", "unknown") for r in matched_results))
    session.source_details.append({
        "type": "client_detection",
        "result_ids": request.result_ids,
        "source": "在线检测",
        "file_count": len(matched_results),
        "filenames": matched_filenames[:10],  # 最多记录 10 个文件名
        "segment_count": len(segments),
        "imported_at": datetime.now().isoformat(),
        "auto_annotate_threshold": request.auto_annotate_threshold
    })

    session.status = "completed"
    session.completed_at = datetime.now().isoformat()

    save_session(session)

    log_operation("PROCESS_CLIENT_COMPLETE",
                  f"Results: {len(matched_results)}, Segments: {len(segments)}")

    return DatasetBuildResponse(
        success=True,
        message=f"成功处理 {len(matched_results)} 个检测结果，生成 {len(segments)} 个片段",
        session_id=session.session_id,
        segments=segments,
        stats={
            "total": len(segments),
            "normal": session.normal_count,
            "anomaly": session.anomaly_count,
            "unlabeled": session.unlabeled_count,
            "auto_annotated": sum(1 for s in segments if s.status == DatasetItemStatus.ANNOTATED)
        }
    )


async def process_manual_upload(
    session: DatasetBuildSession,
    files: List[UploadFile],
    auto_predict: bool = False,
    algorithm: Optional[str] = None
) -> DatasetBuildResponse:
    """
    处理手动上传的文件
    1. 保存文件
    2. 使用检测算法切分
    3. 可选：调用模型进行预测
    """
    log_operation("PROCESS_MANUAL_START", f"Files: {len(files)}, Auto predict: {auto_predict}")
    
    session.status = "processing"
    session.auto_predict = auto_predict
    session.algorithm = algorithm
    
    # 创建会话目录
    session_dir = os.path.join(DATASET_BUILDER_DIR, session.session_id)
    upload_dir = os.path.join(session_dir, "uploads")
    slice_dir = os.path.join(session_dir, "slices")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(slice_dir, exist_ok=True)
    
    # 保存上传的文件
    saved_files = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}:
            continue
        
        file_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}{ext}")
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        saved_files.append({
            "original_name": file.filename,
            "saved_path": file_path
        })
    
    session.source_files = [f["saved_path"] for f in saved_files]

    # 记录来源信息
    session.source_details.append({
        "type": "manual",
        "source": "手动上传",
        "file_count": len(saved_files),
        "filenames": [f["original_name"] for f in saved_files],
        "imported_at": datetime.now().isoformat(),
        "auto_predict": auto_predict,
        "algorithm": algorithm
    })

    # 切分音频
    all_segments = []
    for file_info in saved_files:
        segments = await slice_audio_file(
            file_path=file_info["saved_path"],
            output_dir=slice_dir,
            session=session,
            original_filename=file_info["original_name"]
        )
        all_segments.extend(segments)
    
    # 如果需要自动预测
    if auto_predict and algorithm:
        all_segments = await predict_segments(all_segments, algorithm, session)
    
    # 更新会话
    session.segments = all_segments
    session.total_segments = len(all_segments)
    session.normal_count = sum(1 for s in all_segments if s.predicted_label == "normal")
    session.anomaly_count = sum(1 for s in all_segments if s.predicted_label == "anomaly")
    session.unlabeled_count = sum(1 for s in all_segments if s.status == DatasetItemStatus.SLICED)
    session.status = "completed"
    session.completed_at = datetime.now().isoformat()
    
    save_session(session)
    
    log_operation("PROCESS_MANUAL_COMPLETE", f"Total segments: {len(all_segments)}")
    
    return DatasetBuildResponse(
        success=True,
        message=f"成功处理 {len(saved_files)} 个文件，生成 {len(all_segments)} 个片段",
        session_id=session.session_id,
        segments=all_segments,
        stats={
            "total": len(all_segments),
            "normal": session.normal_count,
            "anomaly": session.anomaly_count,
            "unlabeled": session.unlabeled_count
        }
    )


async def slice_audio_file(
    file_path: str,
    output_dir: str,
    session: DatasetBuildSession,
    original_filename: str
) -> List[AudioSegmentInfo]:
    """
    使用长音频分析器切分音频文件
    """
    segments = []
    
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
        result = analyzer.analyze(file_path)
        
        try:
            db_connector.cursor.close()
            db_connector.conn.close()
        except:
            pass
        
        if not result.segment_matches:
            log_operation("NO_SEGMENTS", f"File: {original_filename}", "WARNING")
            return []
        
        # 加载音频
        y, sr = librosa.load(file_path, sr=22050)
        audio_total_duration = librosa.get_duration(y=y, sr=sr)
        
        for idx, segment_match in enumerate(result.segment_matches):
            start_time = max(0, segment_match.start_time)
            end_time = min(audio_total_duration, segment_match.end_time)
            
            if end_time - start_time < 3.0:
                continue
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]
            
            # 生成文件名
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            matched_name = segment_match.music_name
            segment_filename = f"{base_name}_seg{idx:03d}_{matched_name}_{int(start_time)}s.wav"
            segment_path = os.path.join(output_dir, segment_filename)
            
            # 保存音频
            sf.write(segment_path, segment_audio, sr)
            
            # 生成时频图
            spectrogram_path = segment_path.replace('.wav', '.png')
            generate_spectrogram(segment_path, spectrogram_path)
            
            # 创建片段信息
            segment_id = f"seg_{hashlib.md5(segment_path.encode()).hexdigest()[:12]}"
            
            segment = AudioSegmentInfo(
                segment_id=segment_id,
                source_type=DataSourceType.MANUAL,
                original_filename=original_filename,
                segment_filename=segment_filename,
                duration=round(end_time - start_time, 2),
                file_path=segment_path,
                spectrogram_path=spectrogram_path if os.path.exists(spectrogram_path) else None,
                reference_audio=matched_name,
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                status=DatasetItemStatus.SLICED,
                source_file_path=file_path
            )
            
            segments.append(segment)
    
    except Exception as e:
        log_operation("SLICE_ERROR", f"File: {original_filename}, Error: {str(e)}", "ERROR")
    
    return segments


async def predict_segments(
    segments: List[AudioSegmentInfo],
    algorithm: str,
    session: DatasetBuildSession
) -> List[AudioSegmentInfo]:
    """
    对音频片段进行预测
    """
    if not segments:
        return segments
    
    log_operation("PREDICT_START", f"Segments: {len(segments)}, Algorithm: {algorithm}")
    
    try:
        from backend.core.task_manager import task_manager
        from algorithms import create_detector
        
        # 创建检测器
        detector = create_detector(
            algorithm_name=algorithm,
            config_manager=task_manager.config,
            device="auto"
        )
        detector.load_model()
        
        # 准备图片路径（时频图）
        image_paths = []
        segment_map = {}
        
        for segment in segments:
            if segment.spectrogram_path and os.path.exists(segment.spectrogram_path):
                image_paths.append(segment.spectrogram_path)
                segment_map[len(image_paths) - 1] = segment
        
        if not image_paths:
            log_operation("NO_IMAGES", "No spectrograms found", "WARNING")
            return segments
        
        # 批量预测
        results = detector.predict_batch(image_paths)
        
        # 更新片段信息
        for idx, result in enumerate(results):
            if idx in segment_map:
                segment = segment_map[idx]
                segment.predicted_label = "anomaly" if result.is_anomaly else "normal"
                segment.predicted_score = result.anomaly_score
                segment.predicted_confidence = result.anomaly_score if result.is_anomaly else (1 - result.anomaly_score)
                segment.status = DatasetItemStatus.PREDICTED
        
        # 释放资源
        detector.release()
        import torch
        torch.cuda.empty_cache()
        
        log_operation("PREDICT_COMPLETE", f"Predicted: {len(results)}")
        
    except Exception as e:
        log_operation("PREDICT_ERROR", str(e), "ERROR")
    
    return segments


def generate_spectrogram(audio_path: str, output_path: str) -> bool:
    """生成时频图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        y, sr = librosa.load(audio_path, sr=22050)
        y = librosa.util.normalize(y)
        D = librosa.stft(y)
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        plt.figure(figsize=(6.3, 6.3))
        librosa.display.specshow(DB, sr=sr)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return True
    except Exception as e:
        log_operation("SPECTROGRAM_ERROR", str(e), "ERROR")
        return False


def _import_single_segment(
    segment: "AudioSegmentInfo",
    train_ratio: int = 10,
    target_category: Optional[str] = None
) -> Tuple[bool, dict]:
    """
    拷贝一个已标注片段到正式数据集目录 data/spk/{category}/{split}/{label}/。

    Returns:
        (success, info_dict) 其中 info_dict = {"category", "split_type", "label"}
    """
    try:
        category = target_category or segment.reference_audio
        label = segment.manual_label or segment.predicted_label or "normal"

        category_path = os.path.join(DATASET_ROOT, category)
        train_good = os.path.join(category_path, "train", "good")
        test_good = os.path.join(category_path, "test", "good")
        test_anomaly = os.path.join(category_path, "test", "anomaly")

        os.makedirs(train_good, exist_ok=True)
        os.makedirs(test_good, exist_ok=True)
        os.makedirs(test_anomaly, exist_ok=True)

        train_count = len([f for f in os.listdir(train_good) if f.endswith('.wav')])
        test_count = len([f for f in os.listdir(test_good) if f.endswith('.wav')])

        if label == "anomaly":
            target_dir = test_anomaly
            split_type = "test"
        else:
            total = train_count + test_count
            target_ratio = train_ratio / (train_ratio + 1)
            if total == 0 or train_count / (total + 1) < target_ratio:
                target_dir = train_good
                split_type = "train"
            else:
                target_dir = test_good
                split_type = "test"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{segment.segment_filename}"
        target_path = os.path.join(target_dir, new_filename)

        shutil.copy2(segment.file_path, target_path)

        if segment.spectrogram_path and os.path.exists(segment.spectrogram_path):
            target_spec = target_path.replace('.wav', '.png')
            shutil.copy2(segment.spectrogram_path, target_spec)

        segment.status = DatasetItemStatus.IN_DATASET

        return True, {"category": category, "split_type": split_type, "label": label}

    except Exception as e:
        log_operation("IMPORT_SEGMENT_ERROR",
                      f"Segment: {segment.segment_id}, Path: {segment.file_path}, Error: {str(e)}", "ERROR")
        return False, {}


async def confirm_to_dataset(
    session: DatasetBuildSession,
    segment_ids: Optional[List[str]] = None,
    target_category: Optional[str] = None,
    reference_audios: Optional[List[str]] = None,
    train_ratio: int = 10
) -> DatasetBuildResponse:
    """
    确认将片段导入正式数据集
    支持按参考音频过滤和自定义训练:测试比例
    """
    ref_filter_desc = f", 参考音频: {reference_audios}" if reference_audios else ", 全部参考音频"
    log_operation("CONFIRM_TO_DATASET",
                  f"Session: {session.session_id}, Segments: {len(segment_ids) if segment_ids else 'all'}{ref_filter_desc}, 比例: {train_ratio}:1")

    segments_to_import = []

    if segment_ids:
        segments_to_import = [s for s in session.segments if s.segment_id in segment_ids]
    else:
        segments_to_import = session.segments

    # 按参考音频过滤
    if reference_audios:
        segments_to_import = [s for s in segments_to_import if s.reference_audio in reference_audios]
        log_operation("FILTER_REFERENCE",
                      f"按参考音频过滤后: {len(segments_to_import)} 个片段 (条件: {reference_audios})")

    if not segments_to_import:
        return DatasetBuildResponse(
            success=False,
            message="没有符合条件的片段可以导入",
            session_id=session.session_id,
            stats={"imported": 0, "failed": 0}
        )

    imported_count = 0
    failed_count = 0
    split_stats = {}  # 按参考音频统计划分结果

    for segment in segments_to_import:
        success, info = _import_single_segment(
            segment=segment,
            train_ratio=train_ratio,
            target_category=target_category
        )
        if success:
            imported_count += 1
            cat = info["category"]
            st = info["split_type"]
            lb = info["label"]
            if cat not in split_stats:
                split_stats[cat] = {"train": 0, "test_normal": 0, "test_anomaly": 0}
            if st == "train":
                split_stats[cat]["train"] += 1
            elif lb == "anomaly":
                split_stats[cat]["test_anomaly"] += 1
            else:
                split_stats[cat]["test_normal"] += 1
        else:
            failed_count += 1

    # 输出详细的划分日志
    for cat, stats in split_stats.items():
        log_operation("SPLIT_RESULT",
                      f"类别 '{cat}': 训练集={stats['train']}, 测试集(正常)={stats['test_normal']}, 测试集(异常)={stats['test_anomaly']}")

    # 保存会话更新
    save_session(session)

    log_operation("CONFIRM_COMPLETE",
                  f"成功导入 {imported_count} 个片段到 {len(split_stats)} 个类别，失败 {failed_count} 个")

    return DatasetBuildResponse(
        success=failed_count == 0 or imported_count > 0,
        message=f"成功导入 {imported_count} 个片段，失败 {failed_count} 个",
        session_id=session.session_id,
        stats={
            "imported": imported_count,
            "failed": failed_count,
            "split_details": split_stats
        }
    )


async def confirm_all_annotated_to_dataset(
    reference_audios: Optional[List[str]] = None,
    train_ratio: int = 10
) -> DatasetBuildResponse:
    """
    一键将所有会话中已人工标注的片段导入正式数据集。
    遍历 BUILD_SESSIONS，收集 manual_label 非空且状态为 annotated 的片段，
    导入后设置状态为 in_dataset。
    """
    log_operation("CONFIRM_ALL_ANNOTATED",
                  f"跨会话收集已标注片段, 比例 {train_ratio}:1, 参考音频过滤: {reference_audios or '全部'}")

    # 收集所有会话中已人工标注的片段
    all_annotated = []
    affected_sessions = []

    for session in BUILD_SESSIONS.values():
        annotated = [
            s for s in session.segments
            if s.manual_label is not None and s.status == DatasetItemStatus.ANNOTATED
        ]
        if annotated:
            all_annotated.extend(annotated)
            affected_sessions.append(session)

    if not all_annotated:
        return DatasetBuildResponse(
            success=False,
            message="没有找到已标注的片段。请先在各个会话中完成标注后再导入。",
            stats={"imported": 0, "failed": 0}
        )

    # 按参考音频过滤
    if reference_audios:
        filtered = [s for s in all_annotated if s.reference_audio in reference_audios]
        log_operation("FILTER_REFERENCE",
                      f"按参考音频过滤前: {len(all_annotated)}, 过滤后: {len(filtered)}")
        if not filtered:
            return DatasetBuildResponse(
                success=False,
                message=f"在已标注片段中未找到参考音频: {reference_audios}",
                stats={"imported": 0, "failed": 0}
            )
        all_annotated = filtered

    imported_count = 0
    failed_count = 0
    split_stats = {}

    for segment in all_annotated:
        success, info = _import_single_segment(segment=segment, train_ratio=train_ratio)
        if success:
            imported_count += 1
            cat = info["category"]
            st = info["split_type"]
            lb = info["label"]
            if cat not in split_stats:
                split_stats[cat] = {"train": 0, "test_normal": 0, "test_anomaly": 0}
            if st == "train":
                split_stats[cat]["train"] += 1
            elif lb == "anomaly":
                split_stats[cat]["test_anomaly"] += 1
            else:
                split_stats[cat]["test_normal"] += 1
        else:
            failed_count += 1

    # 保存所有受影响会话
    for session in affected_sessions:
        save_session(session)

    # 输出划分日志
    for cat, stats in split_stats.items():
        log_operation("SPLIT_RESULT",
                      f"类别 '{cat}': 训练集={stats['train']}, 测试集(正常)={stats['test_normal']}, 测试集(异常)={stats['test_anomaly']}")

    log_operation("CONFIRM_ALL_COMPLETE",
                  f"从 {len(affected_sessions)} 个会话导入 {imported_count} 个片段, 失败 {failed_count}")

    return DatasetBuildResponse(
        success=(failed_count == 0 or imported_count > 0),
        message=f"成功导入 {imported_count} 个片段（来自 {len(affected_sessions)} 个会话）"
                + (f"，失败 {failed_count} 个" if failed_count else ""),
        stats={
            "imported": imported_count,
            "failed": failed_count,
            "session_count": len(affected_sessions),
            "split_details": split_stats
        }
    )


# ========== API 端点 ==========

@router.post("/from-detection", response_model=DatasetBuildResponse)
async def create_from_detection(request: CreateFromDetectionRequest):
    """
    从在线检测结果创建数据集构建会话
    保留模型预测值作为预标注
    """
    session = create_session(
        source_type=DataSourceType.DETECTION,
        detection_task_id=request.detection_task_id
    )
    
    result = await process_detection_results(session, request)
    return result


@router.post("/from-manual", response_model=DatasetBuildResponse)
async def create_from_manual(
    files: List[UploadFile] = File(...),
    auto_predict: bool = Form(False),
    algorithm: Optional[str] = Form("dinomaly_dinov3_small")
):
    """
    从手动上传的文件创建数据集构建会话
    
    - **files**: 音频文件列表
    - **auto_predict**: 是否自动调用模型预测
    - **algorithm**: 用于预测的算法（auto_predict为true时有效）
    """
    session = create_session(source_type=DataSourceType.MANUAL)
    
    result = await process_manual_upload(
        session=session,
        files=files,
        auto_predict=auto_predict,
        algorithm=algorithm
    )
    return result


@router.post("/from-client-detection", response_model=DatasetBuildResponse)
async def create_from_client_detection(request: CreateFromClientDetectionRequest):
    """
    从客户端（在线）检测结果创建数据集构建会话
    保留模型的预测分数作为预标注

    - **result_ids**: 客户端检测结果的 ID 列表（从 /api/client/results 获取）
    - **auto_annotate_threshold**: 自动标注置信度阈值（0-1），默认 0.8
    """
    session = create_session(
        source_type=DataSourceType.DETECTION,
        detection_task_id=None
    )

    result = await process_client_detection_results(session, request)
    return result


@router.get("/preview/{session_id}", response_model=DatasetPreviewResponse)
async def preview_session(session_id: str):
    """
    预览会话中的数据
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 按参考音频分组统计
    by_reference = {}
    for segment in session.segments:
        ref = segment.reference_audio
        by_reference[ref] = by_reference.get(ref, 0) + 1
    
    return DatasetPreviewResponse(
        session_id=session_id,
        segments=session.segments,
        total_count=session.total_segments,
        normal_count=session.normal_count,
        anomaly_count=session.anomaly_count,
        unlabeled_count=session.unlabeled_count,
        by_reference=by_reference
    )


@router.post("/annotate/{session_id}")
async def annotate_segment(
    session_id: str,
    segment_id: str = Form(...),
    label: str = Form(...),  # normal 或 anomaly
    annotator: Optional[str] = Form(None)
):
    """
    为单个片段添加人工标注
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 查找片段
    segment = None
    for s in session.segments:
        if s.segment_id == segment_id:
            segment = s
            break
    
    if not segment:
        raise HTTPException(status_code=404, detail="片段不存在")
    
    if label not in ("normal", "anomaly"):
        raise HTTPException(status_code=400, detail="标签必须是 normal 或 anomaly")
    
    # 更新标注
    segment.manual_label = label
    segment.annotated_by = annotator
    segment.annotated_at = datetime.now().isoformat()
    segment.status = DatasetItemStatus.ANNOTATED
    
    # 更新统计
    session.normal_count = sum(1 for s in session.segments if s.manual_label == "normal" or 
                               (s.manual_label is None and s.predicted_label == "normal"))
    session.anomaly_count = sum(1 for s in session.segments if s.manual_label == "anomaly" or 
                                (s.manual_label is None and s.predicted_label == "anomaly"))
    session.unlabeled_count = sum(1 for s in session.segments if s.manual_label is None and s.predicted_label is None)
    
    save_session(session)
    
    return {
        "success": True,
        "message": f"标注成功：{segment_id} -> {label}",
        "segment": segment
    }


@router.post("/annotate-batch/{session_id}")
async def annotate_batch(session_id: str, request: BatchAnnotateRequest):
    """
    批量标注片段
    """
    if session_id != request.session_id:
        raise HTTPException(status_code=400, detail="会话ID不匹配")
    
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    success_count = 0
    failed_items = []
    
    for ann in request.annotations:
        segment_id = ann.get("segment_id")
        label = ann.get("label")
        annotator = ann.get("annotator")
        
        # 查找并更新片段
        for segment in session.segments:
            if segment.segment_id == segment_id:
                if label in ("normal", "anomaly"):
                    segment.manual_label = label
                    segment.annotated_by = annotator
                    segment.annotated_at = datetime.now().isoformat()
                    segment.status = DatasetItemStatus.ANNOTATED
                    success_count += 1
                else:
                    failed_items.append({"segment_id": segment_id, "reason": "无效的标签"})
                break
        else:
            failed_items.append({"segment_id": segment_id, "reason": "片段不存在"})
    
    # 更新统计
    session.normal_count = sum(1 for s in session.segments if s.manual_label == "normal" or 
                               (s.manual_label is None and s.predicted_label == "normal"))
    session.anomaly_count = sum(1 for s in session.segments if s.manual_label == "anomaly" or 
                                (s.manual_label is None and s.predicted_label == "anomaly"))
    session.unlabeled_count = sum(1 for s in session.segments if s.manual_label is None and s.predicted_label is None)
    
    save_session(session)
    
    return {
        "success": True,
        "message": f"批量标注完成：成功 {success_count} 个，失败 {len(failed_items)} 个",
        "success_count": success_count,
        "failed_count": len(failed_items),
        "failed_items": failed_items
    }


@router.post("/confirm-to-dataset")
async def confirm_to_dataset_endpoint(request: ConfirmToDatasetRequest):
    """
    确认将会话中的数据导入正式数据集
    支持指定参考音频和自定义训练:测试比例
    """
    session = get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    result = await confirm_to_dataset(
        session=session,
        segment_ids=request.segment_ids,
        target_category=request.target_category,
        reference_audios=request.reference_audios,
        train_ratio=request.train_ratio
    )

    return result


@router.post("/confirm-all-annotated", response_model=DatasetBuildResponse)
async def confirm_all_annotated_endpoint(request: ConfirmAllAnnotatedRequest):
    """
    一键将所有会话中已人工标注的片段导入正式数据集。
    无需指定 session_id，自动扫描所有会话中的已标注片段。
    """
    result = await confirm_all_annotated_to_dataset(
        reference_audios=request.reference_audios,
        train_ratio=request.train_ratio
    )
    return result


@router.post("/import-to-session")
async def import_to_session(request: ImportToSessionRequest):
    """
    将检测结果追加导入到已有会话中（支持合并多个数据源到一个会话）
    """
    session = get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    if session.status == "pending":
        session.status = "processing"

    if request.source_type == "detection":
        # 复用检测结果处理逻辑
        from backend.core.task_manager import task_manager

        # 去重检查
        if request.detection_task_id and request.detection_task_id in session.import_history:
            raise HTTPException(status_code=400, detail=f"检测任务 {request.detection_task_id} 已导入过")

        # 模拟 CreateFromDetectionRequest
        det_req = CreateFromDetectionRequest(
            detection_task_id=request.detection_task_id or "",
            auto_annotate_threshold=request.auto_annotate_threshold,
            include_normal=request.include_normal,
            include_anomaly=request.include_anomaly
        )
        result = await process_detection_results(session, det_req)
        return result

    elif request.source_type == "client_detection":
        # 复用客户端检测结果处理逻辑
        from backend.core.client_monitor_service import client_detection_service

        # 去重检查
        new_ids = [rid for rid in (request.result_ids or []) if str(rid) not in session.import_history]
        if not new_ids:
            raise HTTPException(status_code=400, detail="所有选中的检测结果均已导入过")
        skipped_req = list(request.result_ids or [])
        for rid in skipped_req:
            if rid not in new_ids:
                log_operation("DEDUP_SKIP", f"结果 {rid} 已存在，跳过", "WARNING")

        # 模拟 CreateFromClientDetectionRequest
        client_req = CreateFromClientDetectionRequest(
            result_ids=new_ids,
            auto_annotate_threshold=request.auto_annotate_threshold
        )
        result = await process_client_detection_results(session, client_req)
        return result

    else:
        raise HTTPException(status_code=400, detail=f"不支持的数据源类型: {request.source_type}")


@router.post("/session/{session_id}/split-preview")
async def preview_session_split(session_id: str, request: SplitPreviewRequest):
    """
    预览会话中的片段按参考音频和比例划分的结果
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    segments = session.segments
    # 按参考音频过滤
    if request.reference_audios:
        segments = [s for s in segments if s.reference_audio in request.reference_audios]

    if not segments:
        return {"session_id": session_id, "total": 0, "by_reference": {}}

    # 统计各参考音频在不同标签下的数量
    by_reference = {}
    for seg in segments:
        ref = seg.reference_audio
        if ref not in by_reference:
            by_reference[ref] = {
                "total": 0,
                "normal": 0,
                "anomaly": 0,
                "unlabeled": 0,
                "predicted_train": 0,
                "predicted_test": 0
            }
        by_reference[ref]["total"] += 1

        label = seg.manual_label or seg.predicted_label
        if label == "normal":
            by_reference[ref]["normal"] += 1
        elif label == "anomaly":
            by_reference[ref]["anomaly"] += 1
        else:
            by_reference[ref]["unlabeled"] += 1

        # 预估划分（正常数据按 train_ratio:1）
        if label == "anomaly":
            by_reference[ref]["predicted_test"] += 1
        elif label == "normal":
            # 使用简单比例预估
            ratio = request.train_ratio / (request.train_ratio + 1)
            by_reference[ref]["predicted_train"] += ratio
            by_reference[ref]["predicted_test"] += (1 - ratio)

    # 转换为整数
    for ref in by_reference:
        by_reference[ref]["predicted_train"] = round(by_reference[ref]["predicted_train"])
        by_reference[ref]["predicted_test"] = round(by_reference[ref]["predicted_test"])
        # 修正舍入误差
        total_normal = by_reference[ref]["normal"]
        if by_reference[ref]["predicted_train"] + by_reference[ref]["predicted_test"] != total_normal:
            by_reference[ref]["predicted_train"] = total_normal - by_reference[ref]["predicted_test"]

    return {
        "session_id": session_id,
        "total": len(segments),
        "train_ratio": request.train_ratio,
        "by_reference": by_reference
    }


@router.post("/session/{session_id}/split-and-import")
async def split_and_import(request: SplitImportRequest):
    """
    按参考音频和比例划分后导入正式数据集
    """
    session = get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 检查是否有未标注的片段
    unlabeled = [s for s in session.segments if s.status == DatasetItemStatus.PREDICTED or s.status == DatasetItemStatus.SLICED]
    if unlabeled:
        log_operation("SPLIT_WARNING",
                      f"会话中有 {len(unlabeled)} 个未人工标注的片段，将使用预标注或默认标签",
                      "WARNING")

    result = await confirm_to_dataset(
        session=session,
        segment_ids=None,
        target_category=request.target_category,
        reference_audios=request.reference_audios,
        train_ratio=request.train_ratio
    )

    return result


@router.get("/sessions")
async def list_sessions():
    """
    列出所有数据集构建会话
    """
    sessions = []
    for session in BUILD_SESSIONS.values():
        sessions.append({
            "session_id": session.session_id,
            "source_type": session.source_type,
            "status": session.status,
            "created_at": session.created_at,
            "completed_at": session.completed_at,
            "total_segments": session.total_segments,
            "normal_count": session.normal_count,
            "anomaly_count": session.anomaly_count,
            "source_details": session.source_details,  # 返回详细来源信息
            "import_history": session.import_history  # 返回导入历史（去重用）
        })

    # 按时间倒序
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    return sessions


@router.get("/session/{session_id}")
async def get_session_detail(session_id: str):
    """
    获取会话详细信息
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return session


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    删除会话及其数据
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 删除会话目录
    session_dir = os.path.join(DATASET_BUILDER_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    
    # 删除会话文件
    session_file = os.path.join(DATASET_BUILDER_DIR, f"{session_id}.json")
    if os.path.exists(session_file):
        os.remove(session_file)
    
    # 从内存中移除
    del BUILD_SESSIONS[session_id]

    return {"success": True, "message": f"会话 {session_id} 已删除"}


@router.get("/available-tasks")
async def get_available_tasks(limit: int = 50, offset: int = 0):
    """
    获取可用于数据集构建的已完成离线检测任务列表
    """
    from backend.core.task_manager import task_manager

    all_tasks = list(task_manager.tasks.values())
    completed = []
    for t in all_tasks:
        if t.status == "completed" and t.results:
            completed.append({
                "task_id": t.id,
                "algorithm": t.algorithm,
                "file_count": len(t.files),
                "result_count": len(t.results),
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "status": t.status
            })

    # 按完成时间倒序
    completed.sort(key=lambda x: x.get("completed_at", ""), reverse=True)
    total = len(completed)
    paged = completed[offset:offset + limit]

    return {"total": total, "tasks": paged, "limit": limit, "offset": offset}


@router.get("/available-client-results")
async def get_available_client_results(limit: int = 50, offset: int = 0):
    """
    获取可用于数据集构建的客户端（在线）检测结果列表
    """
    from backend.core.client_monitor_service import client_detection_service

    total = client_detection_service.get_results_count()
    results = client_detection_service.get_results(limit=limit, offset=offset)

    # 精简返回，只保留关键信息
    simplified = []
    for r in results:
        simplified.append({
            "result_id": r.get("result_id"),
            "filename": r.get("filename"),
            "client_name": r.get("client_name"),
            "timestamp": r.get("timestamp"),
            "anomaly_score": r.get("anomaly_score"),
            "is_anomaly": r.get("is_anomaly"),
            "status": r.get("status"),
            "has_audio_slice": r.get("audio_slice_path") is not None,
            "music_name": (r.get("segment_info") or {}).get("music_name") or r.get("music_name")
        })

    return {"total": total, "results": simplified, "limit": limit, "offset": offset}


# 模块加载时从磁盘恢复会话
_load_sessions_from_disk()
