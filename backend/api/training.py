"""
模型训练 API
支持多算法族训练（Dinomaly / Anomalib / ADer / BaseASD）
"""
import os
import sys
import json
import time
import subprocess
import threading
import re
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "spk")
SAVED_RESULTS_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
os.makedirs(SAVED_RESULTS_DIR, exist_ok=True)

# 训练任务状态存储
TRAINING_TASKS: Dict[str, dict] = {}

# 支持的算法族及其训练模式
ALGORITHM_FAMILIES = {
    "dinomaly": {
        "name": "Dinomaly",
        "algorithms": ["dinomaly_dinov3_small", "dinomaly_dinov3_base", "dinomaly_dinov3_large",
                       "dinomaly_dinov2_small", "dinomaly_dinov2_base", "dinomaly_dinov2_large"],
        "trainable": True,
    },
    "anomalib": {
        "name": "Anomalib",
        "algorithms": ["patchcore", "padim", "cfa", "csflow", "dfkde", "dfm", "draem",
                       "dsr", "efficient_ad", "fastflow", "fre", "reverse_distillation",
                       "stfpm", "ganomaly", "supersimplenet", "uflow", "uninet",
                       "general_ad", "glass", "inp_former", "l2bt", "patchflow"],
        "trainable": True,
    },
    "ader": {
        "name": "ADer",
        "algorithms": ["mambaad", "invad", "vitad", "unad", "cflow", "pyramidflow", "simplenet"],
        "trainable": True,
    },
}


class DatasetInfo(BaseModel):
    """数据集信息"""
    name: str
    train_normal_count: int
    test_normal_count: int
    test_anomaly_count: int
    total_count: int
    trainable: bool


class TrainingRequest(BaseModel):
    """训练请求 - 支持多算法族"""
    categories: List[str]
    algorithm_family: str = "dinomaly"  # dinomaly, anomalib, ader
    algorithm_name: str = ""           # 具体算法名，为空则使用默认
    model_type: str = "dinov3"         # (Dinomaly) dinov2 或 dinov3
    model_size: str = "small"          # (Dinomaly) small, base, large
    total_iters: int = 1000
    batch_size: int = 8
    learning_rate: float = 0.0001
    save_interval: int = 100
    enable_augmentation: bool = True
    validation_ratio: float = 0.1


class TrainingConfig(BaseModel):
    """训练配置"""
    categories: List[str]
    algorithm_family: str = "dinomaly"
    algorithm_name: str = ""
    model_type: str
    model_size: str
    total_iters: int
    batch_size: int
    learning_rate: float
    save_interval: int
    enable_augmentation: bool
    validation_ratio: float
    gpu_id: int = 0
    num_workers: int = 4
    seed: int = 42


class TrainingStatus(BaseModel):
    """训练状态"""
    task_id: str
    status: str
    categories: List[str]
    algorithm_family: str = "dinomaly"
    model_type: str = ""
    model_size: str = ""
    total_iters: int
    current_iter: int = 0
    progress: str = ""
    message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model_path: Optional[str] = None
    log: str = ""
    metrics: Dict[str, Any] = {}
    loss_history: List[float] = []
    learning_rate: float = 0.0001
    estimated_time_remaining: Optional[str] = None


class TrainingMetrics(BaseModel):
    """训练指标"""
    task_id: str
    current_iter: int
    total_iters: int
    loss: float
    learning_rate: float
    epoch_time: float
    avg_iter_time: float
    estimated_remaining: str
    gpu_memory_used: Optional[float] = None
    gpu_utilization: Optional[float] = None


class TrainingVisualization(BaseModel):
    """训练可视化"""
    task_id: str
    loss_curve: List[Dict[str, Any]]
    learning_rate_curve: List[Dict[str, Any]]
    iter_time_curve: List[Dict[str, Any]]
    current_metrics: Dict[str, Any]


class TrainedModel(BaseModel):
    """已训练模型信息"""
    name: str
    path: str
    size_mb: float
    created_at: str
    algorithm_family: str = "dinomaly"
    model_type: str = ""
    model_size: str = ""


# ============================================================================
# API 端点
# ============================================================================

@router.get("/families")
async def get_algorithm_families():
    """获取支持的算法族及其可训练算法列表"""
    return ALGORITHM_FAMILIES


@router.get("/datasets", response_model=List[DatasetInfo])
async def get_training_datasets():
    """获取可用于训练的数据集列表"""
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

            trainable = train_count >= 10

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
    """获取指定类别的详细统计"""
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
    """获取已训练的模型列表（所有算法族）"""
    models = []
    if os.path.exists(SAVED_RESULTS_DIR):
        for filename in os.listdir(SAVED_RESULTS_DIR):
            if filename.endswith('.pth'):
                filepath = os.path.join(SAVED_RESULTS_DIR, filename)
                stat = os.stat(filepath)
                # 从文件名推断算法族和类型
                fname = filename.lower()
                if "anomalib_" in fname:
                    algorithm_family = "anomalib"
                elif "ader_" in fname:
                    algorithm_family = "ader"
                elif "baseasd_" in fname:
                    algorithm_family = "baseasd"
                else:
                    algorithm_family = "dinomaly"

                model_type = ""
                model_size = ""
                if algorithm_family == "dinomaly":
                    model_type = "dinov3" if "dinov3" in fname else "dinov2"
                    model_size = "large" if "large" in fname else ("base" if "base" in fname else "small")

                models.append(TrainedModel(
                    name=filename,
                    path=filepath,
                    size_mb=round(stat.st_size / (1024 * 1024), 2),
                    created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    algorithm_family=algorithm_family,
                    model_type=model_type,
                    model_size=model_size
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
    """启动训练任务（多算法族支持）"""
    # 验证算法族
    family = request.algorithm_family
    if family not in ALGORITHM_FAMILIES:
        raise HTTPException(status_code=400, detail=f"不支持的算法族: {family}")

    # 验证算法名称
    if request.algorithm_name and request.algorithm_name not in ALGORITHM_FAMILIES[family]["algorithms"]:
        raise HTTPException(status_code=400, detail=f"算法 '{request.algorithm_name}' 不属于 {family} 族")

    # 验证数据集类别
    valid_categories = []
    dataset_stats = {}
    for cat in request.categories:
        cat_path = os.path.join(DATASET_ROOT, cat)
        if not os.path.exists(cat_path):
            raise HTTPException(status_code=400, detail=f"类别 '{cat}' 不存在")

        train_dir = os.path.join(cat_path, "train", "good")
        train_count = len([f for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg', '.bmp', '.JPG'))]) if os.path.exists(train_dir) else 0

        if train_count < 10:
            raise HTTPException(
                status_code=400,
                detail=f"类别 '{cat}' 训练数据不足（至少10个样本，当前{train_count}个）"
            )
        valid_categories.append(cat)
        dataset_stats[cat] = {"train": train_count}

    if not valid_categories:
        raise HTTPException(status_code=400, detail="没有有效的训练类别")

    # 生成任务ID和保存名称
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    task_id = f"train_{timestamp}"
    cat_str = "_".join(valid_categories)
    algo_name = request.algorithm_name or family
    save_name = f"{family}_{algo_name}_{cat_str}_{timestamp}"

    # 创建训练配置
    config = TrainingConfig(
        categories=valid_categories,
        algorithm_family=family,
        algorithm_name=request.algorithm_name,
        model_type=request.model_type,
        model_size=request.model_size,
        total_iters=request.total_iters,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        save_interval=request.save_interval,
        enable_augmentation=request.enable_augmentation,
        validation_ratio=request.validation_ratio
    )

    TRAINING_TASKS[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "categories": valid_categories,
        "algorithm_family": family,
        "algorithm_name": request.algorithm_name,
        "model_type": request.model_type,
        "model_size": request.model_size,
        "total_iters": request.total_iters,
        "current_iter": 0,
        "progress": "准备中...",
        "message": "",
        "started_at": None,
        "completed_at": None,
        "model_path": None,
        "log": "",
        "process": None,
        "save_name": save_name,
        "config": config.model_dump(),
        "dataset_stats": dataset_stats,
        "metrics": {
            "loss_history": [],
            "learning_rate_history": [],
            "iter_time_history": [],
            "best_loss": float('inf'),
            "best_iter": 0
        },
        "start_time": None
    }

    # 根据算法族启动相应的训练执行器
    thread = threading.Thread(
        target=_dispatch_training,
        args=(task_id, config, save_name),
        daemon=True
    )
    thread.start()

    return {
        "success": True,
        "task_id": task_id,
        "message": f"{family} 训练任务已启动",
        "config": config.model_dump(),
        "dataset_stats": dataset_stats
    }


def _dispatch_training(task_id: str, config: TrainingConfig, save_name: str):
    """根据算法族分派到不同的训练执行器"""
    family = config.algorithm_family
    if family == "dinomaly":
        _run_dinomaly_training(task_id, config, save_name)
    elif family == "anomalib":
        _run_anomalib_training(task_id, config, save_name)
    elif family == "ader":
        _run_ader_training(task_id, config, save_name)
    else:
        task = TRAINING_TASKS.get(task_id)
        if task:
            task["status"] = "failed"
            task["progress"] = f"不支持的算法族: {family}"


# ============================================================================
# 训练执行器
# ============================================================================

def _run_dinomaly_training(task_id: str, config: TrainingConfig, save_name: str):
    """Dinomaly 训练执行器"""
    import time
    task = TRAINING_TASKS.get(task_id)
    if not task:
        return

    task["status"] = "running"
    task["started_at"] = datetime.now().isoformat()
    task["start_time"] = time.time()
    task["progress"] = "正在初始化 Dinomaly 训练环境..."

    try:
        data_path = DATASET_ROOT
        save_dir = SAVED_RESULTS_DIR

        cmd = [
            sys.executable, "-m", "algorithms.Dinomaly.dinomaly_train_evaluate",
            "--data_path", data_path,
            "--save_dir", save_dir,
            "--save_name", save_name,
            "--model_size", config.model_size,
            "--model_type", config.model_type,
            "--batch_size", str(config.batch_size),
            "--total_iters", str(config.total_iters),
            "--categories",
        ] + config.categories

        if config.enable_augmentation:
            cmd.append("--enable_augmentation")

        _run_subprocess_with_logging(task_id, cmd)

    except Exception as e:
        task = TRAINING_TASKS.get(task_id)
        if task:
            task["status"] = "failed"
            task["progress"] = f"训练异常: {str(e)}"
            task["log"] += f"\n[错误] {str(e)}"


def _run_anomalib_training(task_id: str, config: TrainingConfig, save_name: str):
    """Anomalib 训练执行器（通过 Anomalib CLI/Engine）"""
    import time
    task = TRAINING_TASKS.get(task_id)
    if not task:
        return

    task["status"] = "running"
    task["started_at"] = datetime.now().isoformat()
    task["start_time"] = time.time()
    task["progress"] = "正在初始化 Anomalib 训练环境..."

    try:
        # 使用 Anomalib CLI 进行训练
        cmd = [
            sys.executable, "-m", "Anomalib.cli",
            "--model", config.algorithm_name or "padim",
            "--data", DATASET_ROOT,
            "--train_batch_size", str(config.batch_size),
            "--max_epochs", str(max(1, config.total_iters // 100)),
        ]

        _run_subprocess_with_logging(task_id, cmd)

    except Exception as e:
        task = TRAINING_TASKS.get(task_id)
        if task:
            task["status"] = "failed"
            task["progress"] = f"训练异常: {str(e)}"
            task["log"] += f"\n[错误] {str(e)}"


def _run_ader_training(task_id: str, config: TrainingConfig, save_name: str):
    """ADer 训练执行器（通过 ADerTaskAssigner）"""
    import time
    task = TRAINING_TASKS.get(task_id)
    if not task:
        return

    task["status"] = "running"
    task["started_at"] = datetime.now().isoformat()
    task["start_time"] = time.time()
    task["progress"] = "正在初始化 ADer 训练环境..."

    try:
        # ADer 训练通过调用 ADer 内置的训练脚本
        method_name = _ader_method_name(config.algorithm_name or "mambaad")

        script_path = os.path.join(PROJECT_ROOT, "algorithms", "ADer", "run.py")
        cmd = [
            sys.executable, script_path,
            "-c", f"ADer/configs/{method_name.lower()}/{method_name.lower()}_spk.py",
            "-m", "train",
        ]

        _run_subprocess_with_logging(task_id, cmd)

    except Exception as e:
        task = TRAINING_TASKS.get(task_id)
        if task:
            task["status"] = "failed"
            task["progress"] = f"训练异常: {str(e)}"
            task["log"] += f"\n[错误] {str(e)}"


def _ader_method_name(algo_name: str) -> str:
    """将算法名映射到 ADer 内部方法名"""
    mapping = {
        "mambaad": "MambaAD",
        "invad": "InVad",
        "vitad": "ViTAD",
        "unad": "UniAD",
        "cflow": "CFlow",
        "pyramidflow": "PyramidFlow",
        "simplenet": "SimpleNet",
    }
    return mapping.get(algo_name, "MambaAD")


# ============================================================================
# 子进程管理
# ============================================================================

def _run_subprocess_with_logging(task_id: str, cmd: List[str]):
    """运行子进程并实时解析日志"""
    import time
    task = TRAINING_TASKS.get(task_id)
    if not task:
        return

    task["log"] = f"命令: {' '.join(cmd)}\n"

    venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        cmd[0] = venv_python

    # 设置环境变量
    env_vars = {**os.environ}
    env_vars["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env_vars["DINOMALY_ENCODER_DIR"] = os.path.join(PROJECT_ROOT, "models", "pre_trained")
    env_vars["PRETRAINED_MODELS_DIR"] = os.path.join(PROJECT_ROOT, "models", "pre_trained")
    env_vars["TORCH_HOME"] = os.path.join(PROJECT_ROOT, "models", "pre_trained")

    config_path = os.path.join(PROJECT_ROOT, "backend/config", "config.yaml")
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            for key, value in cfg.get('environments', {}).items():
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

    iter_start_time = time.time()
    for line in process.stdout:
        task = TRAINING_TASKS.get(task_id)
        if not task:
            return
        task["log"] += line

        # 解析迭代/损失信息（通用解析）
        iter_match = re.search(r'Iter\s*\[(\d+)/(\d+)\].*Loss:\s*([\d.]+)', line, re.IGNORECASE)
        if iter_match:
            current_iter = int(iter_match.group(1))
            total_iters = int(iter_match.group(2))
            loss = float(iter_match.group(3))

            task["current_iter"] = current_iter
            task["progress"] = f"训练进度: {current_iter}/{total_iters} ({current_iter/total_iters*100:.1f}%)"

            task["metrics"]["loss_history"].append(loss)
            if len(task["metrics"]["loss_history"]) > 1000:
                task["metrics"]["loss_history"] = task["metrics"]["loss_history"][-1000:]

            if loss < task["metrics"]["best_loss"]:
                task["metrics"]["best_loss"] = loss
                task["metrics"]["best_iter"] = current_iter

            iter_end_time = time.time()
            iter_duration = iter_end_time - iter_start_time
            task["metrics"]["iter_time_history"].append(iter_duration)
            if len(task["metrics"]["iter_time_history"]) > 100:
                task["metrics"]["iter_time_history"] = task["metrics"]["iter_time_history"][-100:]
            iter_start_time = iter_end_time

            if current_iter > 0:
                avg = sum(task["metrics"]["iter_time_history"]) / len(task["metrics"]["iter_time_history"])
                remaining_seconds = avg * (total_iters - current_iter)
                task["estimated_time_remaining"] = _format_duration(remaining_seconds)

        lr_match = re.search(r'LR:\s*([\d.e-]+)', line, re.IGNORECASE)
        if lr_match:
            task["metrics"]["learning_rate_history"].append(float(lr_match.group(1)))

        if len(task["log"]) > 50000:
            task["log"] = task["log"][-40000:]

    process.wait()

    task = TRAINING_TASKS.get(task_id)
    if not task:
        return

    if process.returncode == 0:
        task["status"] = "completed"
        task["progress"] = "训练完成"
        save_dir = SAVED_RESULTS_DIR
        for f in os.listdir(save_dir):
            if task["save_name"] in f and f.endswith('.pth'):
                task["model_path"] = os.path.join(save_dir, f)
                break
    else:
        task["status"] = "failed"
        task["progress"] = f"训练失败 (退出码: {process.returncode})"

    task["completed_at"] = datetime.now().isoformat()
    task["process"] = None


def _format_duration(seconds: float) -> str:
    """格式化时长"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        return f"{int(seconds/60)}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


# ============================================================================
# 状态查询端点
# ============================================================================

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
        algorithm_family=task.get("algorithm_family", "dinomaly"),
        model_type=task.get("model_type", ""),
        model_size=task.get("model_size", ""),
        total_iters=task["total_iters"],
        current_iter=task.get("current_iter", 0),
        progress=task["progress"],
        message=task["message"],
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        model_path=task.get("model_path"),
        log=task["log"][-5000:] if task.get("log") else "",
        metrics=task.get("metrics", {}),
        loss_history=task.get("metrics", {}).get("loss_history", []),
        learning_rate=task.get("config", {}).get("learning_rate", 0.0001),
        estimated_time_remaining=task.get("estimated_time_remaining")
    )


@router.get("/visualization/{task_id}", response_model=TrainingVisualization)
async def get_training_visualization(task_id: str):
    """获取训练可视化数据"""
    task = TRAINING_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    metrics = task.get("metrics", {})
    loss_curve = []
    for i, loss in enumerate(metrics.get("loss_history", [])):
        loss_curve.append({"iter": i + 1, "loss": loss,
                          "smooth_loss": _calculate_smooth_loss(metrics.get("loss_history", []), i)})

    lr_curve = []
    for i, lr in enumerate(metrics.get("learning_rate_history", [])):
        lr_curve.append({"iter": i + 1, "learning_rate": lr})

    iter_time_curve = []
    for i, t in enumerate(metrics.get("iter_time_history", [])):
        iter_time_curve.append({"iter": i + 1, "time_ms": t * 1000})

    current_metrics = {
        "current_iter": task.get("current_iter", 0),
        "total_iters": task["total_iters"],
        "best_loss": metrics.get("best_loss", float('inf')),
        "best_iter": metrics.get("best_iter", 0),
        "avg_iter_time_ms": sum(metrics.get("iter_time_history", [0])) / max(len(metrics.get("iter_time_history", [1])), 1) * 1000,
        "progress_percent": (task.get("current_iter", 0) / task["total_iters"] * 100) if task["total_iters"] > 0 else 0
    }

    return TrainingVisualization(
        task_id=task_id,
        loss_curve=loss_curve,
        learning_rate_curve=lr_curve,
        iter_time_curve=iter_time_curve,
        current_metrics=current_metrics
    )


def _calculate_smooth_loss(loss_history: List[float], index: int, window: int = 10) -> float:
    if not loss_history:
        return 0.0
    start = max(0, index - window // 2)
    end = min(len(loss_history), index + window // 2 + 1)
    window_losses = loss_history[start:end]
    return sum(window_losses) / len(window_losses) if window_losses else 0.0


@router.get("/metrics/{task_id}", response_model=TrainingMetrics)
async def get_training_metrics(task_id: str):
    """获取训练实时指标"""
    task = TRAINING_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    metrics = task.get("metrics", {})
    gpu_memory = None
    gpu_util = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except:
                pass
    except:
        pass

    avg_iter_time = sum(metrics.get("iter_time_history", [0])) / max(len(metrics.get("iter_time_history", [1])), 1)

    return TrainingMetrics(
        task_id=task_id,
        current_iter=task.get("current_iter", 0),
        total_iters=task["total_iters"],
        loss=metrics.get("loss_history", [0])[-1] if metrics.get("loss_history") else 0,
        learning_rate=metrics.get("learning_rate_history", [0.0001])[-1] if metrics.get("learning_rate_history") else 0.0001,
        epoch_time=avg_iter_time * task["total_iters"] if task["total_iters"] > 0 else 0,
        avg_iter_time=avg_iter_time,
        estimated_remaining=task.get("estimated_time_remaining", "计算中..."),
        gpu_memory_used=gpu_memory,
        gpu_utilization=gpu_util
    )


@router.get("/tasks")
async def list_training_tasks():
    """列出所有训练任务"""
    tasks = []
    for _, task in TRAINING_TASKS.items():
        tasks.append({
            "task_id": task["task_id"],
            "status": task["status"],
            "categories": task["categories"],
            "algorithm_family": task.get("algorithm_family", "dinomaly"),
            "model_type": task.get("model_type", ""),
            "model_size": task.get("model_size", ""),
            "progress": task["progress"],
            "current_iter": task.get("current_iter", 0),
            "total_iters": task["total_iters"],
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
