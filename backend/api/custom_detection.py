"""
自定义异常检测 API - 直接图片异常检测
支持所有可用的图像异常检测算法（排除音频专用和未实现的占位算法）
"""
import os
import sys
import json
import time
import uuid
import shutil
import threading
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from pydantic import BaseModel

logger = logging.getLogger("backend.custom_detection")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

router = APIRouter(tags=["custom-detection"])

# 需要在自定义检测中排除的算法（完全不可用的存根/不适用类型）
# 注：仅推理可用的算法（padim/dfkde/MuSc/SubspaceAD等）不在此列表，它们在检测页面正常可用
# 详细决策见 records/算法可用性总览.md
EXCLUDED_ALGORITHMS = {
    # === other_adapters.py 中的未实现存根（永久不实现）===
    # hiad/multiads/dictas: 空壳，无从零实现的算法代码
    # musc/subspacead: 旧版统一入口，新版独立适配器已覆盖（musc_clip_*, subspacead_dinov2_*）
    "hiad", "multiads", "musc", "dictas", "subspacead", "audio_feature_cluster",

    # === BaseASD 系列 — 依赖 TensorFlow/Keras（永久不安装）===
    # 原因: 与PyTorch环境冲突风险，性能已被Dinomaly/Anomalib超越
    "denseae", "cae", "vae", "aegan", "differnet",
}

# 算法族分组（前端展示用）
ALGORITHM_GROUPS = {
    "Dinomaly": ["dinomaly_dinov3_small", "dinomaly_dinov3_base", "dinomaly_dinov3_large",
                 "dinomaly_dinov2_small", "dinomaly_dinov2_base", "dinomaly_dinov2_large"],
    "Dinomaly2 (预览)": ["dinomaly2_dinov2_small", "dinomaly2_dinov2_base", "dinomaly2_dinov2_large",
                         "dinomaly2_dinov3_small", "dinomaly2_dinov3_base", "dinomaly2_dinov3_large"],
    "Anomalib": ["patchcore", "cfa", "csflow", "dfkde", "dfm", "draem", "dsr",
                 "efficient_ad", "fastflow", "fre", "padim", "reverse_distillation",
                 "stfpm", "ganomaly", "supersimplenet", "uflow", "uninet", "vlm_ad", "winclip",
                 # Anomalib v2.5.0 新增
                 "anomalyvfm", "cfm", "general_ad", "glass", "inp_former",
                 "l2bt", "patchflow", "anomaly_dino"],
    "MuSc (零样本)": ["musc_clip_b32_512", "musc_clip_b16_512", "musc_clip_l14_336",
                     "musc_clip_l14_518", "musc_dinov2_b14_336", "musc_dinov2_b14_518",
                     "musc_dinov2_l14_336", "musc_dinov2_l14_518"],
    "SubspaceAD (少样本)": ["subspacead_dinov2_large_672", "subspacead_dinov2_large_518",
                           "subspacead_dinov2_large_336", "subspacead_dinov2_base_672",
                           "subspacead_dinov2_base_518", "subspacead_dinov2_small_672"],
    "ADer": ["mambaad", "invad", "vitad", "unad", "cflow", "pyramidflow", "simplenet",
             "destseg", "realnet", "rdpp"],
}

# 反向映射：算法名 → 分组名
_ALG_TO_GROUP = {}
for group, algs in ALGORITHM_GROUPS.items():
    for alg in algs:
        _ALG_TO_GROUP[alg] = group


def get_algorithm_group(alg_name: str) -> str:
    """获取算法所属分组"""
    return _ALG_TO_GROUP.get(alg_name, "其他")


def log_operation(operation: str, details: str = "", status: str = "INFO"):
    """记录操作日志"""
    level = {"INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}.get(status, logging.INFO)
    logger.log(level, f"{operation} | {details}")


def _get_available_algorithms():
    """获取适用于图片检测的算法列表（动态检查可用性，fallback 到静态排除列表）"""
    from algorithms import list_available_algorithms
    all_algs = list_available_algorithms()

    # 尝试使用动态可用性检查缓存
    try:
        from backend.algorithm_availability import get_available_algorithms as _get_dynamic_available
        dynamic_available = _get_dynamic_available()
        if dynamic_available:
            # 动态模式：只返回实际可用的算法
            return sorted(dynamic_available)
    except Exception:
        pass

    # Fallback: 静态排除列表
    return sorted([alg for alg in all_algs if alg not in EXCLUDED_ALGORITHMS])


# ---- 数据模型 ----

class AlgorithmInfo(BaseModel):
    id: str
    name: str
    group: str


class DatasetInfo(BaseModel):
    name: str
    image_count: int
    categories: List[str]


class TaskStatus(BaseModel):
    task_id: str
    status: str  # queued / processing / completed / failed
    progress: float
    message: str


# ---- 任务存储 ----

# 任务状态存储（进程内）
# 结构: {task_id: {"status": str, "progress": float, "message": str, ...}}
CUSTOM_TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()


def _infer_algorithm_family(algorithm: str) -> str:
    """从算法名推断算法族"""
    algo_lower = algorithm.lower()
    if algo_lower.startswith("dinomaly2_") or algo_lower.startswith("dinomaly2-"):
        return "dinomaly2"
    elif algo_lower.startswith("dinomaly_") or algo_lower.startswith("dinomaly-"):
        return "dinomaly"
    elif algo_lower.startswith("ader_") or algo_lower.startswith("ader-"):
        return "ader"
    # Anomalib 算法名无前缀，用已知列表判断
    anomalib_algos = {
        "patchcore", "padim", "cfa", "csflow", "dfkde", "dfm", "draem", "dsr",
        "efficient_ad", "fastflow", "fre", "reverse_distillation", "stfpm",
        "ganomaly", "supersimplenet", "uflow", "uninet", "vlm_ad", "winclip",
        "anomalyvfm", "cfm", "general_ad", "glass", "inp_former", "l2bt",
        "patchflow", "anomaly_dino",
    }
    if algo_lower in anomalib_algos:
        return "anomalib"
    # ADer 算法名
    ader_algos = {"mambaad", "invad", "vitad", "unad", "cflow", "pyramidflow", "simplenet",
                  "destseg", "realnet", "rdpp"}
    if algo_lower in ader_algos:
        return "ader"
    return ""

# ---- 辅助路径 ----

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads", "custom_detection")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "vis", "custom_detection")
SPK_DIR = os.path.join(PROJECT_ROOT, "data", "spk")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved")


def _ensure_dirs():
    """确保目录存在"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---- API 端点 ----

@router.get("/trained-models")
async def list_trained_models():
    """获取已训练模型列表，供推理选择"""
    from datetime import datetime
    models = []
    if not os.path.exists(SAVED_MODELS_DIR):
        return {"models": models}

    for entry in os.listdir(SAVED_MODELS_DIR):
        entry_path = os.path.join(SAVED_MODELS_DIR, entry)
        if entry.startswith('_') or entry.endswith('.txt') or entry == 'log.txt':
            continue
        # 跳过 _best.pth 软链接/副本（训练产物的快捷方式，与原始模型重复，同训练页面逻辑）
        if entry.endswith('_best.pth'):
            continue
        # 跳过无效的模型文件格式（.omb, .json, .log, .yaml 等）
        if os.path.isfile(entry_path) and not entry.endswith(('.pth', '.ckpt', '.pt')):
            continue

        if os.path.isfile(entry_path) and entry.endswith(('.pth', '.ckpt', '.pt')):
            is_dir = False
            stat = os.stat(entry_path)
        elif os.path.isdir(entry_path):
            is_dir = True
            # 验证目录中包含有效的权重文件
            has_valid_weight = False
            total_size = 0
            for root, dirs, files in os.walk(entry_path):
                for f in files:
                    fp = os.path.join(root, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
                        if f.endswith(('.pth', '.ckpt', '.pt', '.safetensors')):
                            has_valid_weight = True
            if not has_valid_weight:
                continue
            stat = type('stat', (), {
                'st_size': total_size,
                'st_mtime': os.path.getmtime(entry_path)
            })()
        else:
            continue

        # 推断算法族（使用统一推断模块）
        from backend.core.model_meta import infer_model_meta
        meta = infer_model_meta(entry, SAVED_MODELS_DIR)
        algorithm_family = meta["algorithm_family"]
        algorithm_name = meta["algorithm_name"]
        category = meta["category"]

        # 尝试匹配到可用算法 ID
        matched_algorithm_id = ""
        if algorithm_name:
            all_algs = []
            for group_algs in ALGORITHM_GROUPS.values():
                all_algs.extend(group_algs)

            # 1. 精确匹配
            if algorithm_name in all_algs:
                matched_algorithm_id = algorithm_name
            # 2. 同族前缀匹配: family_algo_name
            elif algorithm_family:
                candidate = f"{algorithm_family}_{algorithm_name}"
                if candidate in all_algs:
                    matched_algorithm_id = candidate
            # 3. 宽松后缀匹配
            if not matched_algorithm_id:
                for alg_id in all_algs:
                    if alg_id.endswith(f"_{algorithm_name}"):
                        matched_algorithm_id = alg_id
                        break

        models.append({
            "name": entry,
            "path": entry_path,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "algorithm_family": algorithm_family,
            "algorithm_name": algorithm_name,
            "category": category,
            "matched_algorithm_id": matched_algorithm_id,
            "is_dir": is_dir,
        })

    models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"models": models}


@router.get("/algorithms", response_model=List[AlgorithmInfo])
async def list_algorithms():
    """获取适用于图片检测的算法列表"""
    algs = _get_available_algorithms()
    return [
        AlgorithmInfo(
            id=alg,
            name=_get_algorithm_display_name(alg),
            group=get_algorithm_group(alg)
        )
        for alg in algs
    ]


@router.get("/algorithms/detail")
async def list_algorithms_with_detail():
    """获取算法列表（含分组详细信息，前端展示用）"""
    algs = _get_available_algorithms()

    # 加载可用性缓存
    availability = {}
    try:
        from backend.algorithm_availability import get_all_availability
        availability = get_all_availability()
    except Exception:
        pass

    groups = {}
    for alg in algs:
        group = get_algorithm_group(alg)
        if group not in groups:
            groups[group] = []

        avail = availability.get(alg)
        groups[group].append({
            "id": alg,
            "name": _get_algorithm_display_name(alg),
            "inference_available": avail.inference_available if avail else True,
            "training_available": avail.training_available if avail else False,
            "reasons": avail.reasons if avail else [],
        })
    return {"groups": groups, "flat": algs}


def _get_algorithm_display_name(alg_name: str) -> str:
    """生成算法的显示名称"""
    name_map = {
        # Dinomaly
        "dinomaly_dinov3_small": "Dinomaly DINOv3 Small",
        "dinomaly_dinov3_base": "Dinomaly DINOv3 Base",
        "dinomaly_dinov3_large": "Dinomaly DINOv3 Large",
        "dinomaly_dinov2_small": "Dinomaly DINOv2 Small",
        "dinomaly_dinov2_base": "Dinomaly DINOv2 Base",
        "dinomaly_dinov2_large": "Dinomaly DINOv2 Large",
        # Anomalib
        "patchcore": "PatchCore",
        "cfa": "CFA",
        "csflow": "CS-Flow",
        "dfkde": "DFKDE",
        "dfm": "DFM",
        "draem": "DRAEM",
        "dsr": "DSR",
        "efficient_ad": "EfficientAD",
        "fastflow": "FastFlow",
        "fre": "FRE",
        "padim": "PaDiM",
        "reverse_distillation": "Reverse Distillation",
        "stfpm": "STFPM",
        "ganomaly": "GANomaly",
        "supersimplenet": "SuperSimpleNet",
        "uflow": "U-Flow",
        "uninet": "UniNet",
        "vlm_ad": "VLM-AD",
        "winclip": "WinCLIP",
        # BaseASD
        "denseae": "DenseAE",
        "cae": "CAE",
        "vae": "VAE",
        "aegan": "AEGAN",
        "differnet": "DifferNet",
        # MuSc
        "musc_clip_b32_512": "MuSc CLIP ViT-B/32@512px",
        "musc_clip_b16_512": "MuSc CLIP ViT-B/16@512px",
        "musc_clip_l14_336": "MuSc CLIP ViT-L/14@336px",
        "musc_clip_l14_518": "MuSc CLIP ViT-L/14@518px",
        "musc_dinov2_b14_336": "MuSc DINOv2 ViT-B/14@336px",
        "musc_dinov2_b14_518": "MuSc DINOv2 ViT-B/14@518px",
        "musc_dinov2_l14_336": "MuSc DINOv2 ViT-L/14@336px",
        "musc_dinov2_l14_518": "MuSc DINOv2 ViT-L/14@518px",
        # SubspaceAD
        "subspacead_dinov2_large_672": "SubspaceAD DINOv2-L@672px",
        "subspacead_dinov2_large_518": "SubspaceAD DINOv2-L@518px",
        "subspacead_dinov2_large_336": "SubspaceAD DINOv2-L@336px",
        "subspacead_dinov2_base_672": "SubspaceAD DINOv2-B@672px",
        "subspacead_dinov2_base_518": "SubspaceAD DINOv2-B@518px",
        "subspacead_dinov2_small_672": "SubspaceAD DINOv2-S@672px",
        # Dinomaly2
        "dinomaly2_dinov2_small": "Dinomaly2 DINOv2 Small",
        "dinomaly2_dinov2_base": "Dinomaly2 DINOv2 Base",
        "dinomaly2_dinov2_large": "Dinomaly2 DINOv2 Large",
        "dinomaly2_dinov3_small": "Dinomaly2 DINOv3 Small",
        "dinomaly2_dinov3_base": "Dinomaly2 DINOv3 Base",
        "dinomaly2_dinov3_large": "Dinomaly2 DINOv3 Large",
        # ADer
        "mambaad": "ADer MambaAD",
        "invad": "ADer InVAD",
        "vitad": "ADer ViTAD",
        "unad": "ADer UniAD",
        "cflow": "ADer CFLow-AD",
        "pyramidflow": "ADer PyramidFlow",
        "simplenet": "ADer SimpleNet",
        "destseg": "ADer DeSTSeg",
        "realnet": "ADer RealNet",
        "rdpp": "ADer RD++",
        # Anomalib v2.5.0 新增
        "anomalyvfm": "AnomalyVFM",
        "cfm": "CFM",
        "general_ad": "General AD",
        "glass": "GLASS",
        "inp_former": "InpFormer",
        "l2bt": "L2BT",
        "patchflow": "PatchFlow",
        "anomaly_dino": "Anomaly DINO",
    }
    return name_map.get(alg_name, alg_name)


@router.get("/datasets")
async def list_datasets():
    """扫描 data/spk/ 获取内置数据集列表及其图片"""
    if not os.path.exists(SPK_DIR):
        return {"datasets": []}

    datasets = []
    for category in sorted(os.listdir(SPK_DIR)):
        cat_path = os.path.join(SPK_DIR, category)
        if not os.path.isdir(cat_path):
            continue

        images = []
        # 扫描所有子目录中的图片
        for subdir in ["train/good", "test/good", "test/anomaly", "test/bad"]:
            test_dir = os.path.join(cat_path, subdir)
            if os.path.isdir(test_dir):
                for fname in sorted(os.listdir(test_dir)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        # 生成可通过 /data/spk/ 访问的 URL 路径
                        rel_path = os.path.relpath(os.path.join(test_dir, fname), SPK_DIR)
                        url_path = f"/data/spk/{rel_path}"
                        images.append({
                            "filename": fname,
                            "path": os.path.join(test_dir, fname),
                            "url": url_path,
                            "label": "good" if "good" in subdir else "anomaly",
                            "category": category
                        })

        if images:
            datasets.append({
                "name": category,
                "image_count": len(images),
                "images": images[:50],  # 限制返回数量，前端可分页或按需加载
                "has_more": len(images) > 50
            })

    return {"datasets": datasets}


@router.post("/upload")
async def upload_and_detect(
    files: List[UploadFile] = File(...),
    algorithm: str = Form(...),
    threshold: float = Form(0.5),
    model_path: str = Form(""),
):
    """
    上传图片并启动自定义异常检测

    - **files**: 图片文件列表 (支持 png, jpg, jpeg, bmp, webp)
    - **algorithm**: 算法名称
    - **threshold**: 异常判定阈值
    """
    _ensure_dirs()
    task_id = str(uuid.uuid4())

    log_operation("UPLOAD_START",
                   f"task_id={task_id}, algorithm={algorithm}, files={len(files)}, threshold={threshold}")

    # 验证算法
    available = _get_available_algorithms()
    if algorithm not in available:
        raise HTTPException(status_code=400, detail=f"不支持的算法: {algorithm}")

    # 验证文件
    if not files or len(files) < 1:
        raise HTTPException(status_code=400, detail="请至少上传1个图片文件")

    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    valid_files = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in allowed_extensions:
            valid_files.append(file)

    if not valid_files:
        raise HTTPException(status_code=400, detail="请上传有效的图片文件 (png/jpg/jpeg/bmp/webp)")

    # 保存上传文件
    task_upload_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(task_upload_dir, exist_ok=True)

    saved_paths = []
    try:
        for file in valid_files:
            # 处理文件名冲突
            safe_name = file.filename
            file_path = os.path.join(task_upload_dir, safe_name)
            # 如果文件名已存在，添加序号
            counter = 1
            while os.path.exists(file_path):
                name, ext = os.path.splitext(safe_name)
                file_path = os.path.join(task_upload_dir, f"{name}_{counter}{ext}")
                counter += 1

            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_paths.append(file_path)

        log_operation("FILES_SAVED", f"task_id={task_id}, saved {len(saved_paths)} files")

        # 初始化任务状态
        with TASKS_LOCK:
            CUSTOM_TASKS[task_id] = {
                "status": "queued",
                "progress": 0.0,
                "message": "任务已创建，等待处理...",
                "algorithm": algorithm,
                "algorithm_family": _infer_algorithm_family(algorithm),
                "threshold": threshold,
                "file_count": len(saved_paths),
                "results": None,
                "error": None,
                "created_at": datetime.now().isoformat()
            }

        # 启动后台检测线程
        thread = threading.Thread(
            target=_run_detection_task,
            args=(task_id, saved_paths, algorithm, threshold, model_path),
            daemon=True
        )
        thread.start()

        log_operation("TASK_QUEUED", f"task_id={task_id}, algorithm={algorithm}, files={len(saved_paths)}")

        return {
            "task_id": task_id,
            "status": "queued",
            "message": f"检测任务已创建，共 {len(saved_paths)} 个文件，算法: {algorithm}"
        }

    except Exception as e:
        # 清理
        if os.path.exists(task_upload_dir):
            shutil.rmtree(task_upload_dir, ignore_errors=True)
        log_operation("UPLOAD_ERROR", f"task_id={task_id}, error={str(e)}", "ERROR")
        raise HTTPException(status_code=500, detail=f"创建检测任务失败: {str(e)}")


@router.post("/from-dataset")
async def detect_from_dataset(
    body: Dict[str, Any] = Body(...),
):
    """
    从内置数据集选择图片进行检测

    请求体:
    ```json
    {
        "dataset": "category_name",
        "image_paths": ["/absolute/path/to/img1.png", ...],
        "algorithm": "padim",
        "threshold": 0.5
    }
    ```
    """
    dataset = body.get("dataset", "")
    image_paths = body.get("image_paths", [])
    algorithm = body.get("algorithm", "padim")
    threshold = body.get("threshold", 0.5)
    model_path = body.get("model_path", "")

    if not image_paths:
        raise HTTPException(status_code=400, detail="请选择至少1张图片")

    # 验证算法
    available = _get_available_algorithms()
    if algorithm not in available:
        raise HTTPException(status_code=400, detail=f"不支持的算法: {algorithm}")

    # 验证路径存在
    valid_paths = []
    for p in image_paths:
        if os.path.exists(p):
            valid_paths.append(p)

    if not valid_paths:
        raise HTTPException(status_code=400, detail="所有选择的图片路径均无效")

    task_id = str(uuid.uuid4())
    _ensure_dirs()

    # 复制图片到任务目录
    task_upload_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(task_upload_dir, exist_ok=True)
    os.chmod(task_upload_dir, 0o777)

    copied_paths = []
    for src_path in valid_paths:
        fname = os.path.basename(src_path)
        dst = os.path.join(task_upload_dir, fname)
        # 避免同名文件冲突：添加序号前缀
        if os.path.exists(dst):
            base, ext = os.path.splitext(fname)
            idx = len(copied_paths)
            dst = os.path.join(task_upload_dir, f"{base}_{idx}{ext}")
        shutil.copy2(src_path, dst)
        os.chmod(dst, 0o644)  # 确保文件可读写
        copied_paths.append(dst)

    # 初始化任务状态
    with TASKS_LOCK:
        CUSTOM_TASKS[task_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "任务已创建，等待处理...",
            "algorithm": algorithm,
            "algorithm_family": _infer_algorithm_family(algorithm),
            "threshold": threshold,
            "file_count": len(copied_paths),
            "results": None,
            "error": None,
            "created_at": time.time()
        }

    # 启动后台检测线程
    thread = threading.Thread(
        target=_run_detection_task,
        args=(task_id, copied_paths, algorithm, threshold, model_path),
        daemon=True
    )
    thread.start()

    log_operation("DATASET_TASK", f"task_id={task_id}, dataset={dataset}, algorithm={algorithm}")

    return {
        "task_id": task_id,
        "status": "queued",
        "message": f"从数据集 {dataset} 选取 {len(copied_paths)} 张图片，算法: {algorithm}"
    }


def _run_detection_task(
    task_id: str,
    image_paths: List[str],
    algorithm: str,
    threshold: float,
    model_path: str = "",
):
    """后台运行检测任务"""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    log_operation("TASK_START", f"task_id={task_id}, algorithm={algorithm}, images={len(image_paths)}, model_path={model_path}")

    try:
        # 更新状态
        with TASKS_LOCK:
            if task_id in CUSTOM_TASKS:
                CUSTOM_TASKS[task_id]["status"] = "processing"
                CUSTOM_TASKS[task_id]["progress"] = 5.0
                CUSTOM_TASKS[task_id]["message"] = "正在初始化检测器..."
                CUSTOM_TASKS[task_id]["started_at"] = time.time()

        # 创建检测器
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 初始化当前线程的 CUDA 上下文（修复 cuDNN 多线程问题）
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # 预热 cuDNN：在当前线程执行一次卷积操作，确保 cuDNN 被正确初始化
            try:
                dummy = torch.zeros(1, 3, 64, 64, device='cuda')
                _ = torch.nn.functional.conv2d(dummy, torch.ones(1, 3, 1, 1, device='cuda'))
                del dummy
                torch.cuda.synchronize()
            except Exception:
                pass  # 预热失败不影响后续（尝试用回退方式）
        from algorithms import create_detector

        kwargs = {}
        if model_path and os.path.exists(model_path):
            # 如果是目录，查找其中的权重文件
            actual_path = model_path
            if os.path.isdir(model_path):
                # 优先查找：.ckpt (Anomalib) > net.pth (ADer) > .pth 文件
                found = None
                for target in ['*.ckpt', 'net.pth', '*.pth']:
                    import glob
                    matches = glob.glob(os.path.join(model_path, '**', target), recursive=True)
                    if matches:
                        found = matches[0]
                        break
                if found:
                    actual_path = found
                    log_operation("MODEL_PATH_RESOLVE", f"task_id={task_id}, directory resolved to: {found}")
                else:
                    log_operation("MODEL_PATH_WARN", f"task_id={task_id}, no weight file found in directory: {model_path}", "WARNING")
            kwargs["model_path"] = actual_path
            log_operation("MODEL_PATH", f"task_id={task_id}, using trained model: {actual_path}")

        detector = create_detector(algorithm, **kwargs)
        detector.threshold = threshold

        log_operation("MODEL_LOAD", f"task_id={task_id}, loading model for {algorithm}")

        with TASKS_LOCK:
            if task_id in CUSTOM_TASKS:
                CUSTOM_TASKS[task_id]["progress"] = 15.0
                CUSTOM_TASKS[task_id]["message"] = f"正在加载模型: {algorithm}..."

        detector.load_model()

        log_operation("MODEL_LOADED", f"task_id={task_id}, model loaded")

        with TASKS_LOCK:
            if task_id in CUSTOM_TASKS:
                CUSTOM_TASKS[task_id]["progress"] = 30.0
                CUSTOM_TASKS[task_id]["message"] = "模型加载完成，开始推理..."

        # 创建结果输出目录
        result_dir = os.path.join(OUTPUT_DIR, task_id)
        os.makedirs(result_dir, exist_ok=True)

        # 执行批量推理
        log_operation("INFERENCE", f"task_id={task_id}, running predict_batch on {len(image_paths)} images")

        with TASKS_LOCK:
            if task_id in CUSTOM_TASKS:
                CUSTOM_TASKS[task_id]["progress"] = 40.0
                CUSTOM_TASKS[task_id]["message"] = f"正在对 {len(image_paths)} 张图片进行推理..."

        batch_results = detector.predict_batch(image_paths)

        # 处理结果
        results_list = []
        total = len(image_paths)

        for i, (img_path, result) in enumerate(zip(image_paths, batch_results)):
            try:
                progress = 40.0 + (i + 1) / total * 50.0
                with TASKS_LOCK:
                    if task_id in CUSTOM_TASKS:
                        CUSTOM_TASKS[task_id]["progress"] = min(progress, 95.0)
                        CUSTOM_TASKS[task_id]["message"] = f"处理中: {os.path.basename(img_path)} ({i+1}/{total})"

                log_operation("PROCESS", f"task_id={task_id}, [{i+1}/{total}] {os.path.basename(img_path)}, "
                                          f"score={result.anomaly_score:.4f}")

                # 生成可视化结果
                heatmap_url = None
                overlay_url = None
                original_url = None

                # 1. 读取原图并保存
                try:
                    original_img = cv2.imread(img_path)
                    if original_img is not None:
                        orig_filename = f"original_{i:04d}.png"
                        orig_path = os.path.join(result_dir, orig_filename)
                        cv2.imwrite(orig_path, original_img)
                        original_url = f"/visualize/custom_detection/{task_id}/{orig_filename}"
                except Exception as e:
                    log_operation("ORIG_SAVE_WARN", f"task_id={task_id}, save original failed: {e}", "WARNING")

                # 2. 如果有异常图，生成热力图和叠加图
                # 优先使用 anomaly_map 数组，其次使用 metadata 中的热力图文件路径（Dinomaly 等）
                metadata_heatmap = None
                metadata_overlay = None
                if result.metadata:
                    metadata_heatmap = result.metadata.get('heatmap_path')
                    metadata_overlay = result.metadata.get('overlay_path')

                if result.anomaly_map is not None:
                    try:
                        # 读取原图（RGB）
                        original_rgb = cv2.cvtColor(
                            cv2.imread(img_path), cv2.COLOR_BGR2RGB
                        ) if original_img is not None else None
                        h, w = original_rgb.shape[:2] if original_rgb is not None else result.anomaly_map.shape[:2]

                        # 处理热力图: 转为 numpy，squeeze 掉多余的 channel/batch 维度
                        amap_raw = result.anomaly_map
                        if hasattr(amap_raw, 'cpu'):
                            amap_raw = amap_raw.cpu().numpy()
                        elif not isinstance(amap_raw, np.ndarray):
                            amap_raw = np.array(amap_raw)
                        amap = amap_raw.astype(np.float32)

                        # squeeze 多余维度: [1, H, W] -> [H, W]
                        while amap.ndim > 2:
                            if amap.shape[0] == 1:
                                amap = amap.squeeze(0)
                            elif amap.shape[-1] == 1:
                                amap = amap.squeeze(-1)
                            else:
                                amap = amap.mean(axis=0)  # 多通道取平均
                                break

                        if amap.shape[:2] != (h, w):
                            amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)

                        # 归一化
                        amap_min, amap_max = amap.min(), amap.max()
                        if amap_max > amap_min:
                            amap_norm = (amap - amap_min) / (amap_max - amap_min)
                        else:
                            amap_norm = np.zeros_like(amap)

                        # 热力图（独立）
                        hm_filename = f"heatmap_{i:04d}.png"
                        hm_path = os.path.join(result_dir, hm_filename)
                        plt.figure(figsize=(6, 6))
                        plt.imshow(amap_norm, cmap='jet', vmin=0, vmax=1)
                        plt.colorbar(label='Anomaly Score')
                        plt.title(f"Score: {result.anomaly_score:.4f}")
                        plt.axis('off')
                        plt.savefig(hm_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
                        plt.close()
                        heatmap_url = f"/visualize/custom_detection/{task_id}/{hm_filename}"

                        # 叠加图
                        if original_rgb is not None:
                            ol_filename = f"overlay_{i:04d}.png"
                            ol_path = os.path.join(result_dir, ol_filename)
                            heatmap_color = cm.jet(amap_norm)[:, :, :3]
                            heatmap_color = (heatmap_color * 255).astype(np.uint8)
                            alpha = 0.6
                            overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_color, alpha, 0)
                            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(ol_path, overlay_bgr)
                            overlay_url = f"/visualize/custom_detection/{task_id}/{ol_filename}"

                    except Exception as e:
                        log_operation("HEATMAP_WARN", f"task_id={task_id}, heatmap failed: {e}, "
                                                      f"anomaly_map_type={type(result.anomaly_map)}, "
                                                      f"anomaly_map_shape={result.anomaly_map.shape if hasattr(result.anomaly_map, 'shape') else 'N/A'}", "WARNING")
                elif metadata_heatmap and os.path.exists(metadata_heatmap):
                    # 使用算法适配器自己生成的热力图（Dinomaly 等通过 metadata 传递路径）
                    try:
                        hm_filename = f"heatmap_{i:04d}.png"
                        hm_path = os.path.join(result_dir, hm_filename)
                        shutil.copy2(metadata_heatmap, hm_path)
                        heatmap_url = f"/visualize/custom_detection/{task_id}/{hm_filename}"
                        if metadata_overlay and os.path.exists(metadata_overlay):
                            ol_filename = f"overlay_{i:04d}.png"
                            ol_path = os.path.join(result_dir, ol_filename)
                            shutil.copy2(metadata_overlay, ol_path)
                            overlay_url = f"/visualize/custom_detection/{task_id}/{ol_filename}"
                    except Exception as e:
                        log_operation("HEATMAP_COPY_WARN", f"task_id={task_id}, copy heatmap failed: {e}", "WARNING")

                results_list.append({
                    "filename": os.path.basename(img_path),
                    "anomaly_score": float(result.anomaly_score),
                    "is_anomaly": bool(result.is_anomaly),
                    "inference_time_ms": float(result.inference_time),
                    "original_url": original_url,
                    "heatmap_url": heatmap_url,
                    "overlay_url": overlay_url,
                    "has_heatmap": heatmap_url is not None,
                    "metadata": result.metadata or {}
                })

            except Exception as e:
                log_operation("FILE_ERROR", f"task_id={task_id}, [{i+1}/{total}] {os.path.basename(img_path)}: {e}", "ERROR")
                results_list.append({
                    "filename": os.path.basename(img_path),
                    "anomaly_score": 0.0,
                    "is_anomaly": False,
                    "inference_time_ms": 0.0,
                    "original_url": None,
                    "heatmap_url": None,
                    "overlay_url": None,
                    "has_heatmap": False,
                    "error": str(e)
                })

        # 计算统计
        anomaly_count = sum(1 for r in results_list if r.get("is_anomaly", False))
        scores = [r["anomaly_score"] for r in results_list if r.get("anomaly_score", 0) > 0]
        avg_score = float(np.mean(scores)) if scores else 0.0
        times = [r["inference_time_ms"] for r in results_list if r.get("inference_time_ms", 0) > 0]
        avg_time = float(np.mean(times)) if times else 0.0

        # 保存结果
        result_data = {
            "task_id": task_id,
            "status": "completed",
            "progress": 100.0,
            "algorithm": algorithm,
            "threshold": threshold,
            "results": results_list,
            "summary": {
                "total": len(results_list),
                "anomaly_count": anomaly_count,
                "normal_count": len(results_list) - anomaly_count,
                "avg_anomaly_score": round(avg_score, 4),
                "avg_inference_time_ms": round(avg_time, 2),
                "threshold": threshold
            }
        }

        result_file = os.path.join(result_dir, "result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        # 更新内存状态
        with TASKS_LOCK:
            if task_id in CUSTOM_TASKS:
                CUSTOM_TASKS[task_id]["status"] = "completed"
                CUSTOM_TASKS[task_id]["progress"] = 100.0
                CUSTOM_TASKS[task_id]["message"] = f"检测完成。异常: {anomaly_count}/{len(results_list)}"
                CUSTOM_TASKS[task_id]["results"] = results_list
                CUSTOM_TASKS[task_id]["completed_at"] = time.time()
                CUSTOM_TASKS[task_id]["summary"] = result_data["summary"]

        # 释放检测器
        detector.release()

        log_operation("TASK_SUCCESS",
                       f"task_id={task_id}, anomaly={anomaly_count}/{len(results_list)}, "
                       f"avg_score={avg_score:.4f}, avg_time={avg_time:.2f}ms")

    except Exception as e:
        import traceback
        log_operation("TASK_ERROR", f"task_id={task_id}, error={str(e)}\n{traceback.format_exc()}", "ERROR")

        # 保存错误信息
        result_dir = os.path.join(OUTPUT_DIR, task_id)
        os.makedirs(result_dir, exist_ok=True)
        error_data = {
            "task_id": task_id,
            "status": "failed",
            "progress": 0.0,
            "algorithm": algorithm,
            "error": str(e),
            "results": [],
            "summary": {"total": 0, "anomaly_count": 0, "normal_count": 0,
                        "avg_anomaly_score": 0.0, "avg_inference_time_ms": 0.0}
        }
        with open(os.path.join(result_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)

        with TASKS_LOCK:
            if task_id in CUSTOM_TASKS:
                CUSTOM_TASKS[task_id]["status"] = "failed"
                CUSTOM_TASKS[task_id]["error"] = str(e)
                CUSTOM_TASKS[task_id]["message"] = f"检测失败: {str(e)}"
                CUSTOM_TASKS[task_id]["completed_at"] = time.time()


@router.get("/result/{task_id}")
async def get_result(task_id: str):
    """获取检测结果"""
    # 首先检查内存中的任务状态
    with TASKS_LOCK:
        task = CUSTOM_TASKS.get(task_id)

    if task:
        if task["status"] in ("queued", "processing"):
            return {
                "task_id": task_id,
                "status": task["status"],
                "progress": task["progress"],
                "message": task["message"],
                "algorithm": task.get("algorithm", ""),
                "results": None,
                "summary": None
            }
        elif task["status"] == "completed" and task.get("results"):
            return {
                "task_id": task_id,
                "status": "completed",
                "progress": 100.0,
                "algorithm": task["algorithm"],
                "threshold": task["threshold"],
                "results": task["results"],
                "summary": task.get("summary", {}),
                "message": task["message"]
            }
        elif task["status"] == "failed":
            # 尝试从文件读取详情
            pass

    # 从文件读取持久化结果
    result_dir = os.path.join(OUTPUT_DIR, task_id)
    result_file = os.path.join(result_dir, "result.json")

    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    # 检查任务是否还在处理中
    task_upload_dir = os.path.join(UPLOAD_DIR, task_id)
    if os.path.exists(task_upload_dir):
        return {
            "task_id": task_id,
            "status": "processing",
            "progress": 50.0,
            "message": "检测仍在进行中..."
        }

    raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")


@router.delete("/result/{task_id}")
async def delete_result(task_id: str):
    """删除检测结果和临时文件"""
    # 清理上传目录
    task_upload_dir = os.path.join(UPLOAD_DIR, task_id)
    if os.path.exists(task_upload_dir):
        shutil.rmtree(task_upload_dir, ignore_errors=True)

    # 清理输出目录
    task_output_dir = os.path.join(OUTPUT_DIR, task_id)
    if os.path.exists(task_output_dir):
        shutil.rmtree(task_output_dir, ignore_errors=True)

    # 清理内存状态
    with TASKS_LOCK:
        CUSTOM_TASKS.pop(task_id, None)

    return {"message": f"任务 {task_id} 已删除"}


@router.get("/history")
async def list_history():
    """获取检测历史记录列表"""
    history = []
    with TASKS_LOCK:
        for task_id, task in CUSTOM_TASKS.items():
            history.append({
                "task_id": task_id,
                "status": task["status"],
                "algorithm": task.get("algorithm", ""),
                "file_count": task.get("file_count", 0),
                "created_at": task.get("created_at", 0),
                "message": task.get("message", "")
            })

    # 按创建时间倒序
    history.sort(key=lambda x: x["created_at"], reverse=True)

    return {"history": history}
