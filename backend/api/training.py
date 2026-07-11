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

# 开源数据集配置: key -> (显示名, 数据目录)
# 注意：仅支持 train/good + test/{good,defect_type} 标准目录结构的数据集
PUBLIC_DATASETS = {
    "mvtec": ("MVTec AD", os.path.join(PROJECT_ROOT, "data", "public_dataset", "mvtec")),
}

# 训练任务状态存储
TRAINING_TASKS: Dict[str, dict] = {}

# 支持的算法族及其训练模式
ALGORITHM_FAMILIES = {
    "dinomaly": {
        "name": "Dinomaly",
        "description": "基于 DINOv2/v3 骨干网络的特征异常检测，支持多种模型规模",
        "trainable": True,
        "param_schema": "encoder_size",
        "algorithms": [
            {"id": "dinomaly_dinov3_small", "name": "Dinomaly DINOv3 Small", "type": "feature_based",
             "description": "基于 DINOv3 ViT-S/16 编码器，384维特征，快速训练",
             "performance": "MVTec AD AUROC ~96%，训练速度快，显存需求低", "gpu_memory": "~2GB", "input_size": "600x600"},
            {"id": "dinomaly_dinov3_base", "name": "Dinomaly DINOv3 Base", "type": "feature_based",
             "description": "基于 DINOv3 ViT-B/16 编码器，768维特征，精度与速度均衡",
             "performance": "MVTec AD AUROC ~98%，中等显存需求", "gpu_memory": "~4GB", "input_size": "600x600"},
            {"id": "dinomaly_dinov3_large", "name": "Dinomaly DINOv3 Large", "type": "feature_based",
             "description": "基于 DINOv3 ViT-L/16 编码器，1024维特征，最高精度",
             "performance": "MVTec AD AUROC ~99%，高精度但需大显存", "gpu_memory": "~8GB", "input_size": "600x600"},
            {"id": "dinomaly_dinov2_small", "name": "Dinomaly DINOv2 Small", "type": "feature_based",
             "description": "基于 DINOv2 ViT-S/14 编码器，384维特征，轻量级",
             "performance": "MVTec AD AUROC ~95%，训练快，适合快速验证", "gpu_memory": "~2GB", "input_size": "600x600"},
            {"id": "dinomaly_dinov2_base", "name": "Dinomaly DINOv2 Base", "type": "feature_based",
             "description": "基于 DINOv2 ViT-B/14 编码器，768维特征",
             "performance": "MVTec AD AUROC ~97%，性价比较高", "gpu_memory": "~4GB", "input_size": "600x600"},
            {"id": "dinomaly_dinov2_large", "name": "Dinomaly DINOv2 Large", "type": "feature_based",
             "description": "基于 DINOv2 ViT-L/14 编码器，1024维特征",
             "performance": "MVTec AD AUROC ~98%，高精度", "gpu_memory": "~8GB", "input_size": "600x600"},
        ],
    },
    "dinomaly2": {
        "name": "Dinomaly2",
        "description": "Dinomaly2 — 统一全频谱异常检测，支持 Context-Aware Recentering + Linear Attention + Loose Constraint",
        "trainable": True,
        "param_schema": "encoder_size",
        "algorithms": [
            {"id": "dinomaly2_dinov2_small", "name": "Dinomaly2 DINOv2 Small", "type": "feature_based",
             "description": "基于 DINOv2-reg ViT-S/14 编码器，支持 Linear Attention 和 Loose Constraint",
             "performance": "MVTec AD AUROC ~97%，比 Dinomaly v1 更稳定", "gpu_memory": "~2GB", "input_size": "448x448"},
            {"id": "dinomaly2_dinov2_base", "name": "Dinomaly2 DINOv2 Base", "type": "feature_based",
             "description": "基于 DINOv2-reg ViT-B/14 编码器，Context-Aware Recentering",
             "performance": "MVTec AD AUROC ~98%，精度与速度均衡", "gpu_memory": "~4GB", "input_size": "448x448"},
            {"id": "dinomaly2_dinov2_large", "name": "Dinomaly2 DINOv2 Large", "type": "feature_based",
             "description": "基于 DINOv2-reg ViT-L/14 编码器，全特性支持",
             "performance": "MVTec AD AUROC ~99%，当前 SOTA 级别", "gpu_memory": "~8GB", "input_size": "448x448"},
        ],
    },
    "anomalib": {
        "name": "Anomalib",
        "description": "Intel 开源异常检测库，支持 30+ 种算法，覆盖特征嵌入、重建、流模型等多种检测范式",
        "trainable": True,
        "param_schema": "algorithm_select",
        "algorithms": [
            {"id": "patchcore", "name": "PatchCore", "type": "feature_embedding",
             "description": "基于核心集的特征嵌入方法，无需训练梯度更新，内存库匹配",
             "performance": "MVTec AD AUROC ~99.1%，工业异常检测经典 SOTA", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "padim", "name": "PaDiM", "type": "feature_embedding",
             "description": "基于参数化异常检测方法，使用多维度高斯分布建模正常特征（需下载预训练 backbone）",
             "performance": "MVTec AD AUROC ~95.3%，训练极快", "gpu_memory": "~2GB", "input_size": "256x256",
             "trainable": False},
            {"id": "efficient_ad", "name": "EfficientAD", "type": "lightweight",
             "description": "轻量级异常检测，知识蒸馏 + 自编码器，推理速度极快（需 ImageNette 数据集做知识蒸馏，暂不可训练）",
             "performance": "MVTec AD AUROC ~97.2%，推理延迟 <1ms/图", "gpu_memory": "~1GB", "input_size": "256x256",
             "trainable": False},
            {"id": "cfa", "name": "CFA", "type": "feature_embedding",
             "description": "耦合超球面特征适应，将正常特征映射到超球面",
             "performance": "MVTec AD AUROC ~96.5%", "gpu_memory": "~3GB", "input_size": "256x256"},
            {"id": "csflow", "name": "CS-Flow", "type": "flow_based",
             "description": "跨尺度归一化流模型，多尺度特征联合建模",
             "performance": "MVTec AD AUROC ~96.0%", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "dfkde", "name": "DFKDE", "type": "feature_embedding",
             "description": "深度特征核密度估计，非参数化异常评分（需下载预训练 backbone）",
             "performance": "MVTec AD AUROC ~93%", "gpu_memory": "~2GB", "input_size": "256x256",
             "trainable": False},
            {"id": "dfm", "name": "DFM", "type": "feature_embedding",
             "description": "深度特征建模，基于 PCA 的特征降维与重建误差",
             "performance": "MVTec AD AUROC ~94%", "gpu_memory": "~2GB", "input_size": "256x256"},
            {"id": "draem", "name": "DRAEM", "type": "reconstruction",
             "description": "判别性异常检测重建，生成模拟异常样本训练",
             "performance": "MVTec AD AUROC ~98.0%", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "dsr", "name": "DSR", "type": "reconstruction",
             "description": "双空间重构，同时建模特征空间和图像空间",
             "performance": "MVTec AD AUROC ~95%", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "fastflow", "name": "FastFlow", "type": "flow_based",
             "description": "快速归一化流异常检测，2D 归一化流建模特征分布（需下载预训练 backbone）",
             "performance": "MVTec AD AUROC ~96.6%，训练快", "gpu_memory": "~3GB", "input_size": "256x256",
             "trainable": False},
            {"id": "fre", "name": "FRE", "type": "feature_embedding",
             "description": "特征重建误差，基于自编码器重建预训练特征",
             "performance": "MVTec AD AUROC ~94%", "gpu_memory": "~2GB", "input_size": "256x256"},
            {"id": "reverse_distillation", "name": "Reverse Distillation", "type": "knowledge_distillation",
             "description": "反向蒸馏，学生网络从中间层反向学习教师特征",
             "performance": "MVTec AD AUROC ~96.7%", "gpu_memory": "~3GB", "input_size": "256x256"},
            {"id": "stfpm", "name": "STFPM", "type": "knowledge_distillation",
             "description": "师生特征金字塔匹配，多尺度特征蒸馏（需下载预训练 backbone）",
             "performance": "MVTec AD AUROC ~95.5%", "gpu_memory": "~3GB", "input_size": "256x256",
             "trainable": False},
            {"id": "ganomaly", "name": "GANomaly", "type": "generative",
             "description": "基于 GAN 的异常检测，编码器-解码器-编码器结构",
             "performance": "MVTec AD AUROC ~76%，较早期方法", "gpu_memory": "~3GB", "input_size": "256x256"},
            {"id": "supersimplenet", "name": "SuperSimpleNet", "type": "feature_learning",
             "description": "超简单网络，特征判别器 + 异常评分头",
             "performance": "MVTec AD AUROC ~97%", "gpu_memory": "~2GB", "input_size": "256x256"},
            {"id": "uflow", "name": "U-Flow", "type": "flow_based",
             "description": "U-Net 结构归一化流，多层级特征流建模（需下载预训练 backbone）",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~3GB", "input_size": "256x256",
             "trainable": False},
            {"id": "uninet", "name": "UniNet", "type": "unified",
             "description": "统一异常检测网络，融合多种检测范式",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "vlm_ad", "name": "VLM-AD", "type": "vision_language",
             "description": "基于视觉语言模型的异常检测，利用文本-图像对齐（需 Ollama 服务，仅推理可用）",
             "performance": "零样本检测 AUROC ~85%，需 CLIP 模型", "gpu_memory": "~4GB", "input_size": "256x256",
             "trainable": False},
            {"id": "winclip", "name": "WinCLIP", "type": "vision_language",
             "description": "窗口级 CLIP 异常检测，多尺度窗口比较",
             "performance": "零样本 AUROC ~91%，少样本进一步提升", "gpu_memory": "~4GB", "input_size": "256x256"},
            # Anomalib v2.5.0 新增
            {"id": "anomalyvfm", "name": "AnomalyVFM", "type": "zero_shot",
             "description": "基于视觉基础模型的零样本异常检测，无需训练数据",
             "performance": "零样本 AUROC ~89%", "gpu_memory": "~6GB", "input_size": "256x256"},
            {"id": "cfm", "name": "CFM", "type": "cross_modal",
             "description": "跨模态融合异常检测，结合视觉和文本信息（需下载预训练 backbone）",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~4GB", "input_size": "256x256",
             "trainable": False},
            {"id": "general_ad", "name": "GeneralAD", "type": "feature_embedding",
             "description": "通用异常检测框架，自适应特征选择",
             "performance": "MVTec AD AUROC ~97%", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "glass", "name": "GLASS", "type": "synthesis",
             "description": "基于合成异常的检测方法，GLocal Anomaly Synthesis",
             "performance": "MVTec AD AUROC ~98%", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "inp_former", "name": "INP-Former", "type": "prototype",
             "description": "基于原型的异常检测，学习正常原型特征",
             "performance": "MVTec AD AUROC ~98%", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "l2bt", "name": "L2BT", "type": "feature_embedding",
             "description": "Learn to Be Thorough，细粒度特征嵌入检测",
             "performance": "MVTec AD AUROC ~97%", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "patchflow", "name": "PatchFlow", "type": "flow_based",
             "description": "Patch 级归一化流异常检测，结合 PatchCore 和 Flow",
             "performance": "MVTec AD AUROC ~97%", "gpu_memory": "~3GB", "input_size": "256x256"},
            {"id": "anomaly_dino", "name": "AnomalyDINO", "type": "few_shot",
             "description": "基于 DINO 的少样本异常检测，利用自监督特征",
             "performance": "少样本 AUROC ~94%", "gpu_memory": "~4GB", "input_size": "256x256"},
        ],
    },
    "ader": {
        "name": "ADer",
        "description": "异常检测框架，集成多种前沿算法（状态空间模型、Transformer、流模型等）",
        "trainable": True,
        "param_schema": "algorithm_select",
        "algorithms": [
            {"id": "mambaad", "name": "MambaAD", "type": "state_space_model",
             "description": "基于状态空间模型（Mamba）的异常检测，线性复杂度长序列建模",
             "performance": "MVTec AD AUROC ~97%，推理速度快", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "invad", "name": "InvAD", "type": "generative",
             "description": "逆生成式异常检测，学习正常数据分布的逆向映射",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "vitad", "name": "ViTAD", "type": "transformer",
             "description": "基于 Vision Transformer 的异常检测，全局注意力机制",
             "performance": "MVTec AD AUROC ~97%", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "unad", "name": "UniAD", "type": "unified",
             "description": "统一异常检测框架，单一模型处理多类别",
             "performance": "MVTec AD AUROC ~96%，支持多类别统一训练", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "cflow", "name": "CFlow (ADer)", "type": "flow_based",
             "description": "条件归一化流异常检测，ADer 框架实现版本",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~3GB", "input_size": "256x256"},
            {"id": "pyramidflow", "name": "PyramidFlow", "type": "flow_based",
             "description": "金字塔级归一化流，多尺度特征异常检测",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~4GB", "input_size": "256x256"},
            {"id": "simplenet", "name": "SimpleNet", "type": "feature_learning",
             "description": "简单网络异常检测，特征空间判别器，轻量高效",
             "performance": "MVTec AD AUROC ~97%，训练快推理快", "gpu_memory": "~2GB", "input_size": "256x256"},
        ],
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
    source: str = "spk"          # spk / mvtec / visa
    source_label: str = "SPK"    # 显示名


class TrainingRequest(BaseModel):
    """训练请求 - 支持多算法族"""
    categories: List[str]
    data_source: str = "spk"           # spk / mvtec / visa — 数据来源
    algorithm_family: str = "dinomaly"  # dinomaly, dinomaly2, anomalib, ader
    algorithm_name: str = ""           # 具体算法名，为空则使用默认
    model_type: str = "dinov3"         # (Dinomaly/Dinomaly2) dinov2 或 dinov3
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
    data_source: str = "spk"
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
    """获取可用于训练的数据集列表（含本地开源数据集）"""
    datasets = []

    def _scan_dir(root: str, source: str, source_label: str):
        """扫描指定目录下的数据集类别"""
        results = []
        if not os.path.exists(root):
            return results
        for category in os.listdir(root):
            category_path = os.path.join(root, category)
            if not os.path.isdir(category_path) or category in ("split_log.jsonl", "license.txt", "readme.txt", "meta.json"):
                continue

            train_good = os.path.join(category_path, "train", "good")
            test_good = os.path.join(category_path, "test", "good")
            test_anomaly = os.path.join(category_path, "test", "anomaly")
            test_bad = os.path.join(category_path, "test", "bad")

            # MVTec 的异常子目录不是 bad/anomaly，而是具体缺陷名
            if not os.path.exists(test_anomaly) and not os.path.exists(test_bad):
                # 扫描 test/ 下非 good 的子目录
                anomaly_count = 0
                test_path = os.path.join(category_path, "test")
                if os.path.exists(test_path):
                    for sub in os.listdir(test_path):
                        sub_path = os.path.join(test_path, sub)
                        if sub != "good" and os.path.isdir(sub_path):
                            anomaly_count += len([f for f in os.listdir(sub_path)
                                                  if f.lower().endswith(('.png', '.jpg', '.bmp'))])
                test_anomaly_count = anomaly_count
            else:
                test_anomaly_count = len([f for f in os.listdir(test_anomaly)
                                          if f.lower().endswith(('.png', '.jpg', '.bmp'))]) if os.path.exists(test_anomaly) else 0
                test_bad_count = len([f for f in os.listdir(test_bad)
                                      if f.lower().endswith(('.png', '.jpg', '.bmp'))]) if os.path.exists(test_bad) else 0
                test_anomaly_count += test_bad_count

            train_count = len([f for f in os.listdir(train_good)
                               if f.lower().endswith(('.png', '.jpg', '.bmp'))]) if os.path.exists(train_good) else 0
            test_normal = len([f for f in os.listdir(test_good)
                               if f.lower().endswith(('.png', '.jpg', '.bmp'))]) if os.path.exists(test_good) else 0

            trainable = train_count >= 10

            results.append(DatasetInfo(
                name=category,
                train_normal_count=train_count,
                test_normal_count=test_normal,
                test_anomaly_count=test_anomaly_count,
                total_count=train_count + test_normal + test_anomaly_count,
                trainable=trainable,
                source=source,
                source_label=source_label,
            ))
        return results

    # 扫描 SPK 自建数据集
    datasets.extend(_scan_dir(DATASET_ROOT, "spk", "SPK"))

    # 扫描本地开源数据集
    for ds_key, (ds_label, ds_path) in PUBLIC_DATASETS.items():
        datasets.extend(_scan_dir(ds_path, ds_key, ds_label))

    datasets.sort(key=lambda x: (x.source, x.name))
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
    """获取已训练的模型列表（所有算法族，支持 .pth 文件和目录形式）"""
    models = []
    if os.path.exists(SAVED_RESULTS_DIR):
        for entry in os.listdir(SAVED_RESULTS_DIR):
            entry_path = os.path.join(SAVED_RESULTS_DIR, entry)

            # 跳过临时训练脚本和日志
            if entry.startswith('_') or entry.endswith('.txt') or entry == 'log.txt':
                continue

            # 支持两种形式：.pth 文件 或 模型目录
            if os.path.isfile(entry_path) and entry.endswith('.pth'):
                is_dir = False
                stat = os.stat(entry_path)
            elif os.path.isdir(entry_path):
                is_dir = True
                # 计算目录总大小
                total_size = 0
                for root, dirs, files in os.walk(entry_path):
                    for f in files:
                        fp = os.path.join(root, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
                stat = type('stat', (), {
                    'st_size': total_size,
                    'st_mtime': os.path.getmtime(entry_path)
                })()
            else:
                continue

            # 从名称推断算法族
            fname = entry.lower()
            if fname.startswith("anomalib_"):
                algorithm_family = "anomalib"
            elif fname.startswith("ader_"):
                algorithm_family = "ader"
            elif fname.startswith("dinomaly2_"):
                algorithm_family = "dinomaly2"
            elif fname.startswith("dinomaly_"):
                algorithm_family = "dinomaly"
            elif "anomalib" in fname:
                algorithm_family = "anomalib"
            elif "ader" in fname:
                algorithm_family = "ader"
            else:
                algorithm_family = "dinomaly"

            # 推断模型类型和大小
            model_type = ""
            model_size = ""
            if algorithm_family == "dinomaly":
                model_type = "dinov3" if "dinov3" in fname else "dinov2"
                model_size = "large" if "large" in fname else ("base" if "base" in fname else "small")
            elif algorithm_family == "dinomaly2":
                model_type = "dinov2"
                model_size = "large" if "large" in fname else ("base" if "base" in fname else "small")

            # 提取算法名称
            algorithm_name = ""
            if algorithm_family == "anomalib":
                # anomalib_patchcore_bottle_... -> patchcore
                parts = entry.split("_")
                if len(parts) >= 2:
                    algorithm_name = parts[1]
            elif algorithm_family == "ader":
                # ader_invad_bottle_... -> invad
                parts = entry.split("_")
                if len(parts) >= 2:
                    algorithm_name = parts[1]

            models.append(TrainedModel(
                name=entry,
                path=entry_path,
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
    family_algorithms = ALGORITHM_FAMILIES[family]["algorithms"]
    family_algo_ids = [a["id"] if isinstance(a, dict) else a for a in family_algorithms]
    if request.algorithm_name and request.algorithm_name not in family_algo_ids:
        raise HTTPException(status_code=400, detail=f"算法 '{request.algorithm_name}' 不属于 {family} 族")

    # 确定数据根目录
    data_source = request.data_source
    if data_source == "spk":
        data_root = DATASET_ROOT
    elif data_source in PUBLIC_DATASETS:
        data_root = PUBLIC_DATASETS[data_source][1]
    else:
        raise HTTPException(status_code=400, detail=f"不支持的数据来源: {data_source}")

    # 验证数据集类别
    valid_categories = []
    dataset_stats = {}
    for cat in request.categories:
        cat_path = os.path.join(data_root, cat)
        if not os.path.exists(cat_path):
            raise HTTPException(status_code=400, detail=f"类别 '{cat}' 在 {data_source} 数据集中不存在")

        train_dir = os.path.join(cat_path, "train", "good")
        train_count = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))]) if os.path.exists(train_dir) else 0

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
        data_source=data_source,
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
        "data_source": data_source,
        "algorithm_family": request.algorithm_family,
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
    elif family == "dinomaly2":
        _run_dinomaly2_training(task_id, config, save_name)
    elif family == "anomalib":
        _run_anomalib_training(task_id, config, save_name)
    elif family == "ader":
        _run_ader_training(task_id, config, save_name)
    else:
        task = TRAINING_TASKS.get(task_id)
        if task:
            task["status"] = "failed"
            task["progress"] = f"不支持的算法族: {family}"


# Anomalib 训练 API 算法名 → anomalib 注册名映射
# Anomalib 的 snake_case 转换会将缩写拆开: VFM→v_f_m, CFM→c_f_m 等
_ANOMALIB_NAME_MAP = {
    "anomalyvfm": "anomaly_v_f_m",
    "cfm": "c_f_m",
    "general_ad": "general_a_d",
    "glass": "glass",
    "inp_former": "inp_former",
    "l2bt": "l2_b_t",
    "patchflow": "patchflow",
    "anomaly_dino": "anomaly_d_i_n_o",
    "winclip": "win_clip",
    "uninet": "uni_net",
    "vlm_ad": "vlm_ad",
    "efficient_ad": "efficient_ad",
}


def _resolve_data_root(data_source: str) -> str:
    """根据 data_source 返回对应的数据根目录"""
    if data_source == "spk":
        return DATASET_ROOT
    if data_source in PUBLIC_DATASETS:
        return PUBLIC_DATASETS[data_source][1]
    return DATASET_ROOT


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
        data_path = _resolve_data_root(config.data_source)
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

        _run_subprocess_with_logging(task_id, cmd)

    except Exception as e:
        task = TRAINING_TASKS.get(task_id)
        if task:
            task["status"] = "failed"
            task["progress"] = f"训练异常: {str(e)}"
            task["log"] += f"\n[错误] {str(e)}"


# Dinomaly2 backbone 映射 (仅支持 DINOv2)
_D2_BACKBONE_MAP = {
    ("dinov2", "small"): "dinov2reg_vit_small_14",
    ("dinov2", "base"): "dinov2reg_vit_base_14",
    ("dinov2", "large"): "dinov2reg_vit_large_14",
}


def _run_dinomaly2_training(task_id: str, config: TrainingConfig, save_name: str):
    """Dinomaly2 训练执行器"""
    import time
    task = TRAINING_TASKS.get(task_id)
    if not task:
        return

    task["status"] = "running"
    task["started_at"] = datetime.now().isoformat()
    task["start_time"] = time.time()
    task["progress"] = "正在初始化 Dinomaly2 训练环境..."

    try:
        data_path = _resolve_data_root(config.data_source)
        save_dir = SAVED_RESULTS_DIR

        backbone = _D2_BACKBONE_MAP.get(
            (config.model_type, config.model_size),
            "dinov2reg_vit_small_14"
        )

        script_path = os.path.join(PROJECT_ROOT, "algorithms", "Dinomaly2", "dinomaly_2D.py")
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        categories_str = ",".join(config.categories)
        cmd = [
            sys.executable, script_path,
            "--data_path", data_path,
            "--save_dir", save_dir,
            "--save_name", save_name,
            "--backbone", backbone,
            "--cuda", str(gpu_id),
            "--categories", categories_str,
            "--total_iters", str(config.total_iters),
            "--image_size", "448",
            "--crop_size", "392",
            "--la", "1",
            "--lc", "2",
            "--cr", "1",
        ]

        _run_subprocess_with_logging(task_id, cmd)

    except Exception as e:
        task = TRAINING_TASKS.get(task_id)
        if task:
            task["status"] = "failed"
            task["progress"] = f"训练异常: {str(e)}"
            task["log"] += f"\n[错误] {str(e)}"


def _run_anomalib_training(task_id: str, config: TrainingConfig, save_name: str):
    """Anomalib 训练执行器（通过 anomalib Engine API）"""
    import time
    task = TRAINING_TASKS.get(task_id)
    if not task:
        return

    task["status"] = "running"
    task["started_at"] = datetime.now().isoformat()
    task["start_time"] = time.time()
    task["progress"] = "正在初始化 Anomalib 训练环境..."

    try:
        data_root = _resolve_data_root(config.data_source)
        algorithm = config.algorithm_name or "patchcore"
        max_epochs = max(1, config.total_iters // 100)
        batch_size = config.batch_size

        # 映射训练 API 算法名到 Anomalib 注册名（snake_case 缩写转换差异）
        anomalib_algo = _ANOMALIB_NAME_MAP.get(algorithm, algorithm)

        # EfficientAD 要求 batch_size=1
        if algorithm == "efficient_ad" and batch_size != 1:
            batch_size = 1

        category = config.categories[0] if config.categories else "bottle"
        algorithms_dir = os.path.join(PROJECT_ROOT, "algorithms")
        train_script = f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
sys.path.insert(0, '{algorithms_dir}')
from anomalib.engine import Engine
from anomalib.models import get_model

model = get_model('{anomalib_algo}')
engine = Engine(
    max_epochs={max_epochs},
    default_root_dir='{SAVED_RESULTS_DIR}/{save_name}',
)
from anomalib.data import MVTecAD
datamodule = MVTecAD(root='{data_root}', category='{category}', train_batch_size={batch_size})
engine.fit(model=model, datamodule=datamodule)
print(f'Training completed: {algorithm}')
"""
        script_path = os.path.join(SAVED_RESULTS_DIR, f"_anomalib_train_{task_id}.py")
        with open(script_path, 'w') as f:
            f.write(train_script)

        cmd = [sys.executable, script_path]
        _run_subprocess_with_logging(task_id, cmd)

        # 清理临时脚本
        if os.path.exists(script_path):
            os.remove(script_path)

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
        data_root = _resolve_data_root(config.data_source)

        script_path = os.path.join(PROJECT_ROOT, "algorithms", "ADer", "run.py")
        ader_root = os.path.join(PROJECT_ROOT, "algorithms")
        ader_cwd = os.path.join(PROJECT_ROOT, "algorithms", "ADer")

        # 配置路径检查: ADer 用 importlib 导入配置，需要模块路径
        # 格式: ADer/configs/{method}/{method}_spk.py → ADer.configs.{method}.{method}_spk
        cfg_dir = os.path.join(PROJECT_ROOT, "algorithms", "ADer", "configs", method_name.lower())
        cfg_file = os.path.join(cfg_dir, f"{method_name.lower()}_spk.py")

        if not os.path.isfile(cfg_file):
            # 降级到 benchmark 配置（如果存在）
            # benchmark 配置命名格式: {method}_256_100e.py 或 {method}_mvtec.py
            benchmark_dir = os.path.join(PROJECT_ROOT, "algorithms", "ADer", "configs",
                                         "benchmark", method_name.lower())
            benchmark_cfg = None
            for candidate in [f"{method_name.lower()}_mvtec.py", f"{method_name.lower()}_256_100e.py"]:
                path = os.path.join(benchmark_dir, candidate)
                if os.path.isfile(path):
                    benchmark_cfg = path
                    break
            if benchmark_cfg:
                os.makedirs(cfg_dir, exist_ok=True)
                import shutil
                shutil.copy2(benchmark_cfg, cfg_file)
                task["log"] += f"\n[INFO] 从 benchmark 配置创建: {cfg_file}"
            else:
                raise FileNotFoundError(f"ADer 配置文件不存在: {cfg_file}")

        # 设置 PYTHONPATH 使 ADer 内部模块可导入
        env = os.environ.copy()
        python_path = f"{ader_root}:{env.get('PYTHONPATH', '')}"
        env["PYTHONPATH"] = python_path

        # CWD 必须是 algorithms/ADer/ 目录，这样：
        # - ADer 内部所有相对导入（from util.xxx, from model.xxx, from configs.xxx）正常工作
        # - trainer/model 注册到正确的 TRAINER/MODEL 实例，避免实例分裂
        # - glob 路径（trainer/[!_]*.py 等）正确匹配
        cfg_path = f"configs/{method_name.lower()}/{method_name.lower()}_spk.py"

        # 计算数据目录相对于 algorithms/ADer/ 的路径
        # ADer 内部 data.root 是相对于 CWD (algorithms/ADer/) 的
        ader_abs = os.path.abspath(ader_cwd)
        data_abs = os.path.abspath(data_root)
        try:
            data_rel = os.path.relpath(data_abs, ader_abs)
        except ValueError:
            data_rel = data_abs  # 不同驱动器时回退到绝对路径

        # 计算 models/saved/ 相对于 algorithms/ADer/ 的路径，让 ADer 直接输出到该目录
        saved_results_abs = os.path.abspath(SAVED_RESULTS_DIR)
        try:
            checkpoint_rel = os.path.relpath(saved_results_abs, ader_abs)
        except ValueError:
            checkpoint_rel = saved_results_abs

        cmd = [
            sys.executable, script_path,
            "-c", cfg_path,
            "-m", "train",
            f"data.root={data_rel}",
            f"trainer.checkpoint={checkpoint_rel}",
        ]

        _run_subprocess_with_logging(task_id, cmd, env=env, cwd=ader_cwd)

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

def _run_subprocess_with_logging(task_id: str, cmd: List[str], env: dict = None, cwd: str = None):
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
    if env:
        env_vars.update(env)
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
        cwd=cwd or PROJECT_ROOT,
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

        # 查找模型文件：优先 .pth 文件，其次目录
        model_found = False
        for f in os.listdir(save_dir):
            if task["save_name"] in f:
                fpath = os.path.join(save_dir, f)
                if f.endswith('.pth') or os.path.isdir(fpath):
                    task["model_path"] = fpath
                    model_found = True
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
