"""
模型训练 API
支持 Dinomaly DINOv2/DINOv3 模型训练
"""
import os
import sys
import json
import time
import subprocess
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "spk")
SAVED_RESULTS_DIR = os.path.join(PROJECT_ROOT, "saved_results")
os.makedirs(SAVED_RESULTS_DIR, exist_ok=True)

# 训练任务状态存储
TRAINING_TASKS: Dict[str, dict] = {}


class DatasetInfo(BaseModel):
    """数据集信息"""
    name: str
    train_normal_count: int
    test_normal_count: int
    test_anomaly_count: int
    total_count: int
    trainable: bool


class TrainingRequest(BaseModel):
    """训练请求"""
    categories: List[str]  # 选择的训练类别
    model_type: str = "dinov3"  # dinov2 或 dinov3
    model_size: str = "small"  # small, base, large
    total_iters: int = 1000
    batch_size: int = 8


class TrainingStatus(BaseModel):
    """训练状态"""
    task_id: str
    status: str  # pending, running, completed, failed
    categories: List[str]
    model_type: str
    model_size: str
    total_iters: int
    progress: str = ""
    message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model_path: Optional[str] = None
    log: str = ""


class TrainedModel(BaseModel):
    """已训练模型信息"""
    name: str
    path: str
    size_mb: float
    created_at: str
    model_type: str


@router.get("/datasets", response_model=List[DatasetInfo])
async def get_training_datasets():
    """
    获取可用于训练的数据集列表
    训练要求至少有一个类别的训练集数据
    """
    datasets = []
    if os.path.exists(DATASET_ROOT):
        for category in os.listdir(DATASET_ROOT):
            category_path = os.path.join(DATASET_ROOT, category)
            if not os.path.isdir(category_path) or category == "split_log.jsonl":
                continue

            train_good = os.path.join(category_path, "train", "good")
            test_good = os.path.join(category_path, "test", "good")
            test_anomaly = os.path.join(category_path, "test", "anomaly")
            test_bad = os.path.join(category_path, "test", "bad")

            train_count = len([f for f in os.listdir(train_good) if f.endswith(('.png', '.jpg', '.bmp', '.JPG'))]) if os.path.exists(train_good) else 0
            test_normal = len([f for f in os.listdir(test_good) if f.endswith(('.png', '.jpg', '.bmp', '.JPG'))]) if os.path.exists(test_good) else 0
            test_anomaly_count = len([f for f in os.listdir(test_anomaly) if f.endswith(('.png', '.jpg', '.bmp', '.JPG'))]) if os.path.exists(test_anomaly) else 0
            test_bad_count = len([f for f in os.listdir(test_bad) if f.endswith(('.png', '.jpg', '.bmp', '.JPG'))]) if os.path.exists(test_bad) else 0
            test_anomaly_count += test_bad_count

            trainable = train_count >= 10  # 至少需要10张训练图像

            datasets.append(DatasetInfo(
                name=category,
                train_normal_count=train_count,
                test_normal_count=test_normal,
                test_anomaly_count=test_anomaly_count,
                total_count=train_count + test_normal + test_anomaly_count,
                trainable=trainable
            ))

    datasets.sort(key=lambda x: x.name)
    return datasets


@router.get("/dataset-stats/{category}")
async def get_dataset_stats(category: str):
    """获取指定类别的详细统计信息"""
    category_path = os.path.join(DATASET_ROOT, category)
    if not os.path.exists(category_path):
        raise HTTPException(status_code=404, detail="类别不存在")

    stats = {"name": category, "train": {}, "test": {}}

    for split_type in ["train", "test"]:
        split_path = os.path.join(category_path, split_type)
        if os.path.exists(split_path):
            for sub_dir in os.listdir(split_path):
                sub_path = os.path.join(split_path, sub_dir)
                if os.path.isdir(sub_path):
                    file_count = len([f for f in os.listdir(sub_path) if f.endswith(('.png', '.jpg', '.bmp', '.JPG'))])
                    stats[split_type][sub_dir] = file_count

    return stats


@router.get("/models", response_model=List[TrainedModel])
async def get_trained_models():
    """获取已训练的模型列表"""
    models = []
    if os.path.exists(SAVED_RESULTS_DIR):
        for filename in os.listdir(SAVED_RESULTS_DIR):
            if filename.endswith('.pth'):
                filepath = os.path.join(SAVED_RESULTS_DIR, filename)
                stat = os.stat(filepath)
                # 从文件名推断模型类型
                model_type = "dinov3" if "dinov3" in filename.lower() else "dinov2"
                models.append(TrainedModel(
                    name=filename,
                    path=filepath,
                    size_mb=round(stat.st_size / (1024 * 1024), 2),
                    created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    model_type=model_type
                ))

    models.sort(key=lambda x: x.created_at, reverse=True)
    return models


@router.get("/models/{filename}")
async def get_model_detail(filename: str):
    """获取模型详细信息"""
    filepath = os.path.join(SAVED_RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="模型不存在")

    stat = os.stat(filepath)
    return {
        "name": filename,
        "path": filepath,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


@router.post("/start")
async def start_training(request: TrainingRequest):
    """
    启动训练任务

    - **categories**: 选择的训练类别列表
    - **model_type**: 模型类型 (dinov2 或 dinov3)
    - **model_size**: 模型大小 (small, base, large)
    - **total_iters**: 训练迭代次数
    - **batch_size**: 批次大小
    """
    # 验证类别
    valid_categories = []
    for cat in request.categories:
        cat_path = os.path.join(DATASET_ROOT, cat)
        if not os.path.exists(cat_path):
            raise HTTPException(status_code=400, detail=f"类别 '{cat}' 不存在")

        train_dir = os.path.join(cat_path, "train", "good")
        if not os.path.exists(train_dir) or len(os.listdir(train_dir)) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"类别 '{cat}' 训练数据不足（至少需要10张图像）"
            )
        valid_categories.append(cat)

    if not valid_categories:
        raise HTTPException(status_code=400, detail="没有有效的训练类别")

    # 生成任务ID
    task_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cat_str = "_".join(valid_categories)
    save_name = f"dinomaly_{request.model_type}_{request.model_size}_{cat_str}"

    TRAINING_TASKS[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "categories": valid_categories,
        "model_type": request.model_type,
        "model_size": request.model_size,
        "total_iters": request.total_iters,
        "progress": "准备中...",
        "message": "",
        "started_at": None,
        "completed_at": None,
        "model_path": None,
        "log": "",
        "process": None,
        "save_name": save_name,
    }

    # 在后台线程启动训练
    thread = threading.Thread(
        target=_run_training,
        args=(task_id, valid_categories, request.model_type, request.model_size,
              request.total_iters, request.batch_size, save_name),
        daemon=True
    )
    thread.start()

    return {"success": True, "task_id": task_id, "message": f"训练任务已启动，任务ID: {task_id}"}


def _ensure_ground_truth(categories: List[str]):
    """确保每个类别的 test 异常目录都有对应的 ground_truth mask"""
    for cat in categories:
        cat_path = os.path.join(DATASET_ROOT, cat)
        test_path = os.path.join(cat_path, "test")
        gt_path = os.path.join(cat_path, "ground_truth")

        if not os.path.exists(test_path):
            continue

        for subdir in os.listdir(test_path):
            if subdir == "good":
                continue  # good 不需要 ground truth

            subdir_path = os.path.join(test_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            gt_subdir = os.path.join(gt_path, subdir)
            if os.path.exists(gt_subdir):
                continue  # 已存在

            os.makedirs(gt_subdir, exist_ok=True)

            # 为每个测试图像创建空白的黑色 mask
            for filename in os.listdir(subdir_path):
                if filename.endswith(('.png', '.jpg', '.bmp', '.JPG')):
                    mask_path = os.path.join(gt_subdir, filename)
                    try:
                        from PIL import Image
                        img = Image.open(os.path.join(subdir_path, filename))
                        mask = Image.new('L', img.size, 0)
                        mask.save(mask_path)
                    except Exception:
                        pass


def _run_training(task_id: str, categories: List[str], model_type: str,
                  model_size: str, total_iters: int, batch_size: int, save_name: str):
    """在后台运行训练任务"""
    task = TRAINING_TASKS.get(task_id)
    if not task:
        return

    task["status"] = "running"
    task["started_at"] = datetime.now().isoformat()
    task["progress"] = "正在初始化训练环境..."

    try:
        # 训练前：确保 ground_truth 目录与 test 目录匹配
        _ensure_ground_truth(categories)

        # 构建命令行参数 - 使用模块方式运行以支持相对导入
        data_path = DATASET_ROOT
        save_dir = SAVED_RESULTS_DIR

        cmd = [
            sys.executable, "-m", "algorithms.Dinomaly.dinomaly_train_evaluate",
            "--data_path", data_path,
            "--save_dir", save_dir,
            "--save_name", save_name,
            "--model_size", model_size,
            "--model_type", model_type,
            "--batch_size", str(batch_size),
            "--total_iters", str(total_iters),
            "--categories",
        ] + categories

        task["progress"] = f"正在启动训练进程..."
        task["log"] = f"命令: {' '.join(cmd)}\n"

        venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
        if os.path.exists(venv_python):
            cmd[0] = venv_python

        # 加载配置文件中的环境变量
        env_vars = {**os.environ}
        env_vars["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        env_vars["DINOMALY_ENCODER_DIR"] = os.path.join(PROJECT_ROOT, "pre_trained")
        env_vars["PRETRAINED_MODELS_DIR"] = os.path.join(PROJECT_ROOT, "pre_trained")
        env_vars["TORCH_HOME"] = os.path.join(PROJECT_ROOT, "pre_trained")

        # 尝试从 config.yaml 加载完整环境变量
        config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                for key, value in config.get('environments', {}).items():
                    if value and key not in env_vars:
                        env_vars[key] = str(value)
            except Exception:
                pass

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=PROJECT_ROOT,
            env=env_vars
        )

        task["process"] = process

        # 实时读取输出
        for line in process.stdout:
            task["log"] += line
            # 更新进度（基于日志内容）
            if "epoch" in line.lower() or "iter" in line.lower():
                task["progress"] = line.strip()[:200]
            # 限制日志长度
            if len(task["log"]) > 50000:
                task["log"] = task["log"][-40000:]

        process.wait()

        if process.returncode == 0:
            task["status"] = "completed"
            task["progress"] = "训练完成"
            # 查找生成的模型文件
            for f in os.listdir(save_dir):
                if save_name in f and f.endswith('.pth'):
                    task["model_path"] = os.path.join(save_dir, f)
                    break
        else:
            task["status"] = "failed"
            task["progress"] = f"训练失败 (退出码: {process.returncode})"

    except Exception as e:
        task["status"] = "failed"
        task["progress"] = f"训练异常: {str(e)}"
        task["log"] += f"\n[错误] {str(e)}"

    task["completed_at"] = datetime.now().isoformat()
    task["process"] = None


@router.get("/status/{task_id}", response_model=TrainingStatus)
async def get_training_status(task_id: str):
    """获取训练任务状态"""
    task = TRAINING_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    return TrainingStatus(
        task_id=task["task_id"],
        status=task["status"],
        categories=task["categories"],
        model_type=task["model_type"],
        model_size=task["model_size"],
        total_iters=task["total_iters"],
        progress=task["progress"],
        message=task["message"],
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        model_path=task.get("model_path"),
        log=task["log"][-5000:] if task.get("log") else "",
    )


@router.get("/status")
async def list_training_tasks():
    """列出所有训练任务"""
    tasks = []
    for task_id, task in TRAINING_TASKS.items():
        tasks.append({
            "task_id": task["task_id"],
            "status": task["status"],
            "categories": task["categories"],
            "model_type": task["model_type"],
            "model_size": task["model_size"],
            "progress": task["progress"],
            "started_at": task.get("started_at"),
            "completed_at": task.get("completed_at"),
        })
    tasks.sort(key=lambda x: x["task_id"], reverse=True)
    return tasks


@router.post("/stop/{task_id}")
async def stop_training(task_id: str):
    """停止训练任务"""
    task = TRAINING_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task["status"] not in ("pending", "running"):
        raise HTTPException(status_code=400, detail="任务不在运行中")

    process = task.get("process")
    if process:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    task["status"] = "failed"
    task["progress"] = "用户手动停止"
    task["completed_at"] = datetime.now().isoformat()

    return {"success": True, "message": "训练已停止"}
