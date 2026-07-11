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
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from pydantic import BaseModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

router = APIRouter(tags=["custom-detection"])

# 需要排除的算法（存根或音频专用）
EXCLUDED_ALGORITHMS = {
    # other_adapters.py 中的未实现存根
    "hiad", "multiads", "musc", "dictas", "subspacead", "audio_feature_cluster",
    # ADer 系列 - 音频→频谱图管道（独特算法后续补全图片推理）
    "mambaad", "invad", "vitad", "unad", "cflow", "pyramidflow", "simplenet",
    # ADer 新增（图片推理待实现: destseg, realnet, rdpp）
    "destseg", "realnet", "rdpp",
    # BaseASD 系列 - 依赖 tensorflow/keras，未安装
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
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [CustomDetection] [{status}] {operation} | {details}")


def _get_available_algorithms():
    """获取适用于图片检测的算法列表"""
    from algorithms import list_available_algorithms
    all_algs = list_available_algorithms()
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

# ---- 辅助路径 ----

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads", "custom_detection")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "vis", "custom_detection")
SPK_DIR = os.path.join(PROJECT_ROOT, "data", "spk")


def _ensure_dirs():
    """确保目录存在"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---- API 端点 ----

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
    groups = {}
    for alg in algs:
        group = get_algorithm_group(alg)
        if group not in groups:
            groups[group] = []
        groups[group].append({
            "id": alg,
            "name": _get_algorithm_display_name(alg)
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
                "threshold": threshold,
                "file_count": len(saved_paths),
                "results": None,
                "error": None,
                "created_at": time.time()
            }

        # 启动后台检测线程
        thread = threading.Thread(
            target=_run_detection_task,
            args=(task_id, saved_paths, algorithm, threshold),
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
            "threshold": threshold,
            "file_count": len(copied_paths),
            "results": None,
            "error": None,
            "created_at": time.time()
        }

    # 启动后台检测线程
    thread = threading.Thread(
        target=_run_detection_task,
        args=(task_id, copied_paths, algorithm, threshold),
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
):
    """后台运行检测任务"""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    log_operation("TASK_START", f"task_id={task_id}, algorithm={algorithm}, images={len(image_paths)}")

    try:
        # 更新状态
        with TASKS_LOCK:
            if task_id in CUSTOM_TASKS:
                CUSTOM_TASKS[task_id]["status"] = "processing"
                CUSTOM_TASKS[task_id]["progress"] = 5.0
                CUSTOM_TASKS[task_id]["message"] = "正在初始化检测器..."

        # 创建检测器
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from algorithms import create_detector
        detector = create_detector(algorithm)
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
                if result.anomaly_map is not None:
                    try:
                        # 读取原图（RGB）
                        original_rgb = cv2.cvtColor(
                            cv2.imread(img_path), cv2.COLOR_BGR2RGB
                        ) if original_img is not None else None
                        h, w = original_rgb.shape[:2] if original_rgb is not None else result.anomaly_map.shape[:2]

                        # 调整热力图尺寸
                        amap = result.anomaly_map.astype(np.float32)
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
                        log_operation("HEATMAP_WARN", f"task_id={task_id}, heatmap failed: {e}", "WARNING")

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
