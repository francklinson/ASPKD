"""
模型训练 API
支持多算法族训练（Dinomaly / Anomalib / ADer / BaseASD）
"""
import os
import sys
import json
import math
import time
import subprocess
import threading
import re
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

logger = logging.getLogger("backend.training")

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

# 训练任务持久化文件
TRAINING_TASKS_FILE = os.path.join(PROJECT_ROOT, "data", "training_tasks.json")


def _save_training_tasks():
    """持久化训练任务到磁盘"""
    try:
        serializable = {}
        for tid, task in TRAINING_TASKS.items():
            t = dict(task)
            # 移除不可序列化的字段
            t.pop("process", None)
            t.pop("start_time", None)
            serializable[tid] = t
        os.makedirs(os.path.dirname(TRAINING_TASKS_FILE), exist_ok=True)
        with open(TRAINING_TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存训练任务失败: {e}")


def _load_training_tasks():
    """从磁盘恢复训练任务"""
    if not os.path.exists(TRAINING_TASKS_FILE):
        return
    try:
        with open(TRAINING_TASKS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for tid, task_data in data.items():
            # 只恢复已完成/失败的任务，运行中的任务无法恢复
            if task_data.get("status") in ("completed", "failed", "stopped"):
                task_data["process"] = None
                task_data["start_time"] = None
                TRAINING_TASKS[tid] = task_data
        logger.info(f"从磁盘恢复 {len(TRAINING_TASKS)} 个训练任务")
    except Exception as e:
        logger.warning(f"加载训练任务失败: {e}")


# 启动时加载
_load_training_tasks()

# 支持的算法族及其训练模式
ALGORITHM_FAMILIES = {
    "dinomaly": {
        "name": "Dinomaly",
        "description": "基于 DINOv2/v3 骨干网络的特征异常检测（CVPR 2025）。核心思想：The less is more philosophy in multi-class unsupervised anomaly detection。采用 Context-Aware Recentering 将多类特征重新中心化到统一空间，Loose Constraint 放松正则化避免过拟合，Linear Attention 实现高效长距离特征交互。在 MVTec AD 上取得 SOTA 级别 AUROC。",
        "trainable": True,
        "param_schema": "encoder_size",
        "algorithms": [
            {"id": "dinomaly_dinov3_small", "name": "Dinomaly DINOv3 Small", "type": "feature_based",
             "description": "基于 DINOv3 ViT-S/16 编码器，384维特征，Context-Aware Recentering + Linear Attention，快速训练",
             "performance": "MVTec AD AUROC ~96%，训练速度快，显存需求低", "gpu_memory": "~2GB", "input_size": "600x600", "supports_multi_category": True},
            {"id": "dinomaly_dinov3_base", "name": "Dinomaly DINOv3 Base", "type": "feature_based",
             "description": "基于 DINOv3 ViT-B/16 编码器，768维特征，精度与速度均衡，Context-Aware Recentering 稳定训练",
             "performance": "MVTec AD AUROC ~98%，中等显存需求", "gpu_memory": "~4GB", "input_size": "600x600", "supports_multi_category": True},
            {"id": "dinomaly_dinov3_large", "name": "Dinomaly DINOv3 Large", "type": "feature_based",
             "description": "基于 DINOv3 ViT-L/16 编码器，1024维特征，最高精度，全特性支持",
             "performance": "MVTec AD AUROC ~99%，高精度但需大显存", "gpu_memory": "~8GB", "input_size": "600x600", "supports_multi_category": True},
            {"id": "dinomaly_dinov2_small", "name": "Dinomaly DINOv2 Small", "type": "feature_based",
             "description": "基于 DINOv2 ViT-S/14 编码器，384维特征，轻量级快速验证",
             "performance": "MVTec AD AUROC ~95%，训练快，适合快速验证", "gpu_memory": "~2GB", "input_size": "600x600", "supports_multi_category": True},
            {"id": "dinomaly_dinov2_base", "name": "Dinomaly DINOv2 Base", "type": "feature_based",
             "description": "基于 DINOv2 ViT-B/14 编码器，768维特征，性价比较高",
             "performance": "MVTec AD AUROC ~97%，性价比较高", "gpu_memory": "~4GB", "input_size": "600x600", "supports_multi_category": True},
            {"id": "dinomaly_dinov2_large", "name": "Dinomaly DINOv2 Large", "type": "feature_based",
             "description": "基于 DINOv2 ViT-L/14 编码器，1024维特征，高精度检测",
             "performance": "MVTec AD AUROC ~98%，高精度", "gpu_memory": "~8GB", "input_size": "600x600", "supports_multi_category": True},
        ],
"supports_multi_category": True,

    },
    "dinomaly2": {
        "name": "Dinomaly2",
        "description": "One Dinomaly2 Detect Them All：统一全频谱异常检测框架（arXiv 2025）。首个统一框架无缝处理多种数据模态（2D、多视角、RGB-3D、RGB-IR）、多种任务设置（单类、多类、推理统一多类、少样本）和应用领域（工业、生物、户外）。核心技术：Context-Aware Recentering + Linear Attention + Loose Constraint，在多个基准上取得前所未有的性能。",
        "trainable": True,
        "param_schema": "encoder_size",
        "algorithms": [
            {"id": "dinomaly2_dinov2_small", "name": "Dinomaly2 DINOv2 Small", "type": "feature_based",
             "description": "基于 DINOv2-reg ViT-S/14 编码器，Linear Attention + Loose Constraint，比 Dinomaly v1 训练更稳定",
             "performance": "MVTec AD AUROC ~97%，比 Dinomaly v1 更稳定", "gpu_memory": "~2GB", "input_size": "448x448", "supports_multi_category": True},
            {"id": "dinomaly2_dinov2_base", "name": "Dinomaly2 DINOv2 Base", "type": "feature_based",
             "description": "基于 DINOv2-reg ViT-B/14 编码器，Context-Aware Recentering 稳定训练，精度与速度均衡",
             "performance": "MVTec AD AUROC ~98%，精度与速度均衡", "gpu_memory": "~4GB", "input_size": "448x448", "supports_multi_category": True},
            {"id": "dinomaly2_dinov2_large", "name": "Dinomaly2 DINOv2 Large", "type": "feature_based",
             "description": "基于 DINOv2-reg ViT-L/14 编码器，全特性支持，当前 SOTA 级别性能",
             "performance": "MVTec AD AUROC ~99%，当前 SOTA 级别", "gpu_memory": "~8GB", "input_size": "448x448", "supports_multi_category": True},
            {"id": "dinomaly2_dinov3_small", "name": "Dinomaly2 DINOv3 Small", "type": "feature_based",
             "description": "基于 DINOv3 ViT-S/16 编码器，新一代视觉Transformer，更强特征表示",
             "performance": "MVTec AD AUROC ~97%，DINOv3 特征更强", "gpu_memory": "~2GB", "input_size": "448x448", "supports_multi_category": True},
            {"id": "dinomaly2_dinov3_base", "name": "Dinomaly2 DINOv3 Base", "type": "feature_based",
             "description": "基于 DINOv3 ViT-B/16 编码器，精度与速度均衡，DINOv3 增强特征",
             "performance": "MVTec AD AUROC ~98%，DINOv3 精度更高", "gpu_memory": "~4GB", "input_size": "448x448", "supports_multi_category": True},
            {"id": "dinomaly2_dinov3_large", "name": "Dinomaly2 DINOv3 Large", "type": "feature_based",
             "description": "基于 DINOv3 ViT-L/16 编码器，最高精度，DINOv3 最强特征表示",
             "performance": "MVTec AD AUROC ~99%，DINOv3 SOTA 级别", "gpu_memory": "~8GB", "input_size": "448x448", "supports_multi_category": True},
        ],
"supports_multi_category": True,

    },
    "anomalib": {
        "name": "Anomalib",
        "description": "Intel 开源异常检测库（v2.5.0），覆盖 30+ 种算法。支持特征嵌入、重建、归一化流、知识蒸馏、生成式、视觉语言等多种检测范式。提供统一的训练/推理/评估管线，支持零样本、少样本和全监督设置。已训练模型自动保存为目录格式（含 .ckpt 权重文件）。",
        "trainable": True,
        "param_schema": "algorithm_select",
        "algorithms": [
            {"id": "patchcore", "name": "PatchCore", "type": "feature_embedding",
             "description": "基于核心集的补丁特征嵌入方法（CVPR 2022）。将图像划分为 patch，通过预训练网络提取中间层特征并存入记忆库，推理时通过核心集子采样近似最近邻搜索计算异常分数。无需梯度训练，MVTec AD 图像级 AUROC 98.0%（WRN-50）。",
             "performance": "MVTec AD AUROC 98.0%（WRN-50），工业异常检测经典 SOTA", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "padim", "name": "PaDiM", "type": "feature_embedding",
             "description": "参数化异常检测方法（ICPR 2021）。使用多维度高斯分布建模正常特征，通过对预训练特征进行参数化估计实现异常评分。训练极快但需下载预训练 backbone。",
             "performance": "MVTec AD AUROC ~95.3%，训练极快", "gpu_memory": "~2GB", "input_size": "256x256",
             "trainable": False, "supports_multi_category": False},
            {"id": "efficient_ad", "name": "EfficientAD", "type": "lightweight",
             "description": "轻量级异常检测（CVPR 2024）。知识蒸馏 + 自编码器，推理延迟 <1ms/图，适合实时检测场景。需 ImageNette 数据集做知识蒸馏，暂不可训练。",
             "performance": "MVTec AD AUROC ~97.2%，推理延迟 <1ms/图", "gpu_memory": "~1GB", "input_size": "256x256",
             "trainable": False, "supports_multi_category": False},
            {"id": "cfa", "name": "CFA", "type": "feature_embedding",
             "description": "耦合超球面特征适应方法（Access 2022）。通过可学习的补丁描述符将正常特征映射到超球面，结合可扩展记忆库实现目标导向的异常定位。",
             "performance": "MVTec AD AUROC ~96.5%", "gpu_memory": "~3GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "csflow", "name": "CS-Flow", "type": "flow_based",
             "description": "跨尺度全卷积归一化流模型（WACV 2022）。联合处理多尺度特征，通过跨尺度耦合块提升细粒度表示能力，支持同时进行异常检测和定位。",
             "performance": "MVTec AD AUROC ~96.0%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "dfkde", "name": "DFKDE", "type": "feature_embedding",
             "description": "深度特征核密度估计。非参数化异常评分方法，通过核密度估计建模正常特征分布。需下载预训练 backbone。",
             "performance": "MVTec AD AUROC ~93%", "gpu_memory": "~2GB", "input_size": "256x256",
             "trainable": False, "supports_multi_category": False},
            {"id": "dfm", "name": "DFM", "type": "feature_embedding",
             "description": "深度特征建模异常检测。基于 PCA 的特征降维与重建误差评估，简单高效的经典方法。",
             "performance": "MVTec AD AUROC ~94%", "gpu_memory": "~2GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "draem", "name": "DRAEM", "type": "reconstruction",
             "description": "判别性重建异常检测方法（ICCV 2021）。由重建子网络和判别子网络组成，使用 Perlin 噪声生成模拟异常样本训练，结合 L2+SSIM 损失和 Focal Loss，对弱缺陷检测效果显著。",
             "performance": "MVTec AD AUROC ~98.0%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "dsr", "name": "DSR", "type": "reconstruction",
             "description": "双空间重构异常检测。通过量化特征学习，使用编码器和双解码器架构，分别在特征空间和图像空间建模正常数据分布。",
             "performance": "MVTec AD AUROC ~95%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "fastflow", "name": "FastFlow", "type": "flow_based",
             "description": "快速 2D 归一化流异常检测（CVPR 2021）。使用 2D 归一化流建模预训练特征的正常分布，训练速度快。需下载预训练 backbone。",
             "performance": "MVTec AD AUROC ~96.6%，训练快", "gpu_memory": "~3GB", "input_size": "256x256",
             "trainable": False, "supports_multi_category": False},
            {"id": "fre", "name": "FRE", "type": "feature_embedding",
             "description": "特征重建误差异常检测。基于自编码器重建预训练特征，通过重建误差评估异常，简单直接。",
             "performance": "MVTec AD AUROC ~94%", "gpu_memory": "~2GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "reverse_distillation", "name": "Reverse Distillation", "type": "knowledge_distillation",
             "description": "反向蒸馏异常检测（CVPR 2022）。学生解码器从中间层反向学习教师特征提取器的特征表示，通过一类别瓶颈嵌入强制特征映射相似性，实现高精度定位。",
             "performance": "MVTec AD AUROC ~96.7%", "gpu_memory": "~3GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "stfpm", "name": "STFPM", "type": "knowledge_distillation",
             "description": "师生特征金字塔匹配。教师-学生网络结构，通过多尺度特征匹配和层级知识融合实现不同尺寸异常检测。需下载预训练 backbone。",
             "performance": "MVTec AD AUROC ~95.5%", "gpu_memory": "~3GB", "input_size": "256x256",
             "trainable": False, "supports_multi_category": False},
            {"id": "ganomaly", "name": "GANomaly", "type": "generative",
             "description": "基于条件 GAN 的异常检测（ACCV 2018）。编码器-解码器-编码器结构，通过比较潜在向量与重构向量的差异评估异常得分，较早期方法。",
             "performance": "MVTec AD AUROC ~76%，较早期方法", "gpu_memory": "~3GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "supersimplenet", "name": "SuperSimpleNet", "type": "feature_learning",
             "description": "超简单网络异常检测（ICPR 2024 / JIMS 2025）。特征提取+特征适配+特征级合成异常生成+分割检测模块，推理时跳过异常生成直接预测，轻量高效。",
             "performance": "MVTec AD AUROC ~97%", "gpu_memory": "~2GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "uflow", "name": "U-Flow", "type": "flow_based",
             "description": "U-Net 结构归一化流异常检测（WACV 2024）。多层级 U 形归一化流建模特征分布，通过 a contrario 框架自动计算异常阈值。需下载预训练 backbone。",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~3GB", "input_size": "256x256",
             "trainable": False, "supports_multi_category": False},
            {"id": "uninet", "name": "UniNet", "type": "unified",
             "description": "统一异常检测网络，融合特征嵌入、重建等多种检测范式的统一框架。",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "vlm_ad", "name": "VLM-AD", "type": "vision_language",
             "description": "基于视觉语言模型的异常检测。利用 CLIP 等模型的文本-图像对齐能力进行零样本异常检测。需 Ollama 服务，仅推理可用。",
             "performance": "零样本检测 AUROC ~85%，需 CLIP 模型", "gpu_memory": "~4GB", "input_size": "256x256",
             "trainable": False, "supports_multi_category": True},
            {"id": "winclip", "name": "WinCLIP", "type": "vision_language",
             "description": "窗口级 CLIP 零样本/少样本异常检测（CVPR 2023）。利用预训练 CLIP 提取图像和文本嵌入，通过余弦相似度计算异常分数，多尺度滑动窗口实现像素级定位。",
             "performance": "零样本 AUROC ~91%，少样本进一步提升", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": True},
            # Anomalib v2.5.0 新增
            {"id": "anomalyvfm", "name": "AnomalyVFM", "type": "zero_shot",
             "description": "零样本视觉基础模型异常检测（CVPR 2026）。将预训练 VFM 转化为零样本异常检测器，先通过 FLUX 生成合成图像训练，直接预测异常分数和异常掩码。",
             "performance": "零样本 AUROC ~89%", "gpu_memory": "~6GB", "input_size": "256x256", "supports_multi_category": True},
            {"id": "cfm", "name": "CFM", "type": "cross_modal",
             "description": "跨模态融合异常检测。结合视觉和文本信息进行跨模态异常检测。需下载预训练 backbone。",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~4GB", "input_size": "256x256",
             "trainable": False, "supports_multi_category": False},
            {"id": "general_ad", "name": "GeneralAD", "type": "feature_embedding",
             "description": "跨域通用异常检测（arXiv 2024）。利用 ViT patch 结构，自监督构造伪异常样本（噪声注入、打乱、复制），注意力判别器逐 patch 评分，跨语义域和工业域均有效。",
             "performance": "MVTec AD AUROC ~97%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": True},
            {"id": "glass", "name": "GLASS", "type": "synthesis",
             "description": "统一异常合成策略 GLocal Anomaly Synthesis。三分支训练：正常分支提取适应特征，GAS 分支通过梯度上升合成近分布异常，LAS 分支叠加 Perlin 噪声纹理模拟远分布异常。",
             "performance": "MVTec AD AUROC ~98%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "inp_former", "name": "INP-Former", "type": "prototype",
             "description": "固有正常原型检测（CVPR 2025）。从测试图像中直接提取固有正常原型（INP），通过交叉注意力聚合预训练 ViT 特征，INP 引导解码器约束输出为正常模式，无需存储训练集原型。",
             "performance": "MVTec AD AUROC ~98%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "l2bt", "name": "L2BT", "type": "feature_embedding",
             "description": "Learn to Be a Transformer 异常检测（IEEE Access）。将异常检测转化为 Transformer 学习任务，精确定位异常区域。",
             "performance": "MVTec AD AUROC ~97%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "patchflow", "name": "PatchFlow", "type": "flow_based",
             "description": "基于 Patch 特征的归一化流异常检测（2025）。结合局部邻域感知 patch 特征与归一化流，引入适配器模块对齐预训练表示与工业图像分布，瓶颈耦合结构降低计算复杂度。",
             "performance": "MVTec AD AUROC ~97%", "gpu_memory": "~3GB", "input_size": "256x256", "supports_multi_category": False},
            {"id": "anomaly_dino", "name": "AnomalyDINO", "type": "few_shot",
             "description": "基于 DINO 的少样本异常检测。利用 DINO 自监督特征的强泛化能力，在少样本设置下进行异常检测和定位。",
             "performance": "少样本 AUROC ~94%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": False},
        ],
"supports_multi_category": False,

    },
    "ader": {
        "name": "ADer",
        "description": "ADer 异常检测工具箱（arXiv 2024）：A Comprehensive Benchmark for Multi-class Visual Anomaly Detection。集成多种前沿算法，覆盖增强式（DRAEM/SimpleNet）、嵌入式（CFA/PatchCore/CFlow/PyramidFlow）、重建式（ViTAD/InvAD/MambaAD）、混合式（UniAD）等范式。默认支持多类无监督异常检测（MUAD）设置，支持可视化和 DDP 训练。已训练模型保存为目录格式（含 net.pth 权重文件）。",
        "trainable": True,
        "param_schema": "algorithm_select",
        "algorithms": [
            {"id": "mambaad", "name": "MambaAD", "type": "state_space_model",
             "description": "基于状态空间模型 Mamba 的异常检测（arXiv 2024）。引入选择性扫描机制，在视觉 Transformer 和 CNN 中实现线性复杂度的长距离依赖建模，适合多类统一训练。",
             "performance": "MVTec AD AUROC ~97%，推理速度快", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": True},
            {"id": "invad", "name": "InvAD", "type": "generative",
             "description": "逆生成式异常检测（arXiv 2024 / COCO-AD）。Learning Feature Inversion for Multi-class Anomaly Detection，基于 StyleGAN2 架构学习正常数据分布的逆向映射，结合像素标准化和可学习风格映射层实现特征反转。",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": True},
            {"id": "vitad", "name": "ViTAD", "type": "transformer",
             "description": "基于 Plain ViT 重建的异常检测（CVIU 2025）。Exploring Plain ViT Reconstruction for Multi-class Unsupervised Anomaly Detection，使用视觉 Transformer 作为骨干，结合融合模块和可变形注意力机制，实现多类统一异常检测与高精度定位。",
             "performance": "MVTec AD AUROC ~97%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": True},
            {"id": "unad", "name": "UniAD", "type": "unified",
             "description": "统一异常检测框架（NeurIPS 2022）。单一模型处理多类别，使用多尺度特征金字塔和模块化卷积网络，结合注意力机制融合不同尺度特征，支持多类别统一训练。",
             "performance": "MVTec AD AUROC ~96%，支持多类别统一训练", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": True},
            {"id": "cflow", "name": "CFlow (ADer)", "type": "flow_based",
             "description": "条件归一化流异常检测（WACV 2022）ADer 框架实现版本。判别式预训练编码器 + 多尺度生成解码器，通过估计编码特征的似然性生成异常图，条件向量提供位置信息。",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~3GB", "input_size": "256x256", "supports_multi_category": True},
            {"id": "pyramidflow", "name": "PyramidFlow", "type": "flow_based",
             "description": "金字塔级归一化流异常检测（CVPR 2023）。多尺度特征金字塔上的归一化流，逐层建模特征分布实现多粒度异常检测，金字塔结构同时捕获全局和局部异常。",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": True},
            {"id": "simplenet", "name": "SimpleNet", "type": "feature_learning",
             "description": "简单网络异常检测（CVPR 2023）。特征空间判别器方法：预训练特征提取+特征适配+高斯噪声合成异常+判别评分，轻量高效，训练快推理快。",
             "performance": "MVTec AD AUROC ~97%，训练快推理快", "gpu_memory": "~2GB", "input_size": "256x256", "supports_multi_category": True},
            {"id": "destseg", "name": "DeSTSeg", "type": "segmentation",
             "description": "分割式异常检测（CVPR 2023）。利用合成异常训练分割网络，结合编码器-解码器架构和歧视性分割网络实现高精度像素级异常定位。",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": True},
            {"id": "realnet", "name": "RealNet", "type": "real_world",
             "description": "真实场景异常检测（CVPR 2024）。面向真实工业场景，利用合成异常和特征级重建实现高精度检测与定位。需要 HuggingFace SDAS 模型，暂不可训练。",
             "performance": "MVTec AD AUROC ~97%", "gpu_memory": "~4GB", "input_size": "256x256",
             "trainable": False, "supports_multi_category": True},
            {"id": "rdpp", "name": "RD++", "type": "knowledge_distillation",
             "description": "增强反向蒸馏异常检测（arXiv 2024）。Reverse Distillation 增强版，结合特征反转和多尺度蒸馏实现更精确的异常检测。",
             "performance": "MVTec AD AUROC ~96%", "gpu_memory": "~4GB", "input_size": "256x256", "supports_multi_category": True},
            ],
"supports_multi_category": True,

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
    gpu_id: int = 0


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
    algorithm_name: str = ""
    model_type: str = ""
    model_size: str = ""
    data_source: str = ""
    error: Optional[str] = None
    total_iters: int
    current_iter: int = 0
    progress: str = ""
    message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model_path: Optional[str] = None
    log: str = ""
    metrics: Dict[str, Any] = {}
    test_metrics: Optional[Dict[str, Any]] = None
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
    algorithm_name: str = ""
    model_type: str = ""
    model_size: str = ""
    category: str = ""
    data_source: str = ""


class RenameRequest(BaseModel):
    """模型重命名请求"""
    new_name: str


# ============================================================================
# API 端点
# ============================================================================

@router.get("/families")
async def get_algorithm_families():
    """获取支持的算法族及其可训练算法列表（含动态可用性检查）"""
    import copy
    families = copy.deepcopy(ALGORITHM_FAMILIES)

    # 尝试加载动态可用性缓存
    try:
        from backend.algorithm_availability import get_all_availability
        availability = get_all_availability()
        if availability:
            for fam_key, fam_info in families.items():
                family_trainable = False
                for algo in fam_info.get("algorithms", []):
                    alg_id = algo["id"]
                    if alg_id in availability:
                        avail = availability[alg_id]
                        algo["trainable"] = avail.training_available
                        algo["available"] = avail.inference_available
                        algo["reasons"] = avail.reasons
                        if avail.training_available:
                            family_trainable = True
                    else:
                        # 算法未在注册表中，保留原有 trainable 配置
                        pass
                # 更新族的可训练状态
                fam_info["trainable"] = family_trainable or fam_info.get("trainable", False)
            return families
    except Exception:
        pass

    return families


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
            # 跳过调试/测试/临时目录
            if entry.startswith('test_') or entry.startswith('debug_') or entry == 'tmp':
                continue
            # 跳过 _best.pth 软链接/副本文件（它们是训练产物的快捷方式，不是独立模型）
            if entry.endswith('_best.pth'):
                continue

            # 支持两种形式：.pth 文件 或 模型目录
            if os.path.isfile(entry_path) and entry.endswith('.pth'):
                is_dir = False
                stat = os.stat(entry_path)
            elif os.path.isdir(entry_path):
                is_dir = True
                # 计算目录总大小
                total_size = 0
                has_weight_file = False
                for root, dirs, files in os.walk(entry_path):
                    for f in files:
                        fp = os.path.join(root, f)
                        if os.path.exists(fp):
                            fsize = os.path.getsize(fp)
                            total_size += fsize
                            if f.endswith(('.pth', '.ckpt', '.pt')):
                                has_weight_file = True
                # 跳过没有权重文件的空目录（训练失败产物）
                if not has_weight_file:
                    continue
                stat = type('stat', (), {
                    'st_size': total_size,
                    'st_mtime': os.path.getmtime(entry_path)
                })()
            else:
                continue

            # 从名称推断算法族（使用统一推断模块）
            from backend.core.model_meta import infer_model_meta
            meta = infer_model_meta(entry, SAVED_RESULTS_DIR)
            algorithm_family = meta["algorithm_family"]
            algorithm_name = meta["algorithm_name"]
            model_type = meta["model_type"]
            model_size = meta["model_size"]
            category = meta["category"]
            data_source = meta["data_source"]

            models.append(TrainedModel(
                name=entry,
                path=entry_path,
                size_mb=round(stat.st_size / (1024 * 1024), 2),
                created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                algorithm_family=algorithm_family,
                algorithm_name=algorithm_name,
                model_type=model_type,
                model_size=model_size,
                category=category,
                data_source=data_source
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


@router.delete("/models/{name}")
async def delete_trained_model(name: str):
    """删除已训练的模型"""
    import shutil

    model_path = os.path.join(SAVED_RESULTS_DIR, name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"模型 '{name}' 不存在")

    # 拒绝删除 _best.pth 快捷方式
    if name.endswith('_best.pth'):
        raise HTTPException(status_code=400, detail="不能删除 _best.pth 快捷方式，请删除原始模型")

    try:
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        else:
            os.remove(model_path)
        logger.info(f"[Models] 已删除模型: {name}")

        # 清理关联的 _best.pth 符号链接
        for entry in os.listdir(SAVED_RESULTS_DIR):
            if entry.endswith('_best.pth'):
                link_path = os.path.join(SAVED_RESULTS_DIR, entry)
                if os.path.islink(link_path):
                    target = os.readlink(link_path)
                    if os.path.basename(target) == name or os.path.basename(os.path.dirname(target)) == name:
                        os.remove(link_path)
                        logger.info(f"[Models] 已清理关联链接: {entry}")

        return {"success": True, "message": f"模型 '{name}' 已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")


@router.put("/models/{name}/rename")
async def rename_trained_model(name: str, request: RenameRequest):
    """重命名已训练的模型"""
    new_name = request.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="新名称不能为空")
    if new_name == name:
        return {"success": True, "message": "名称未变化"}

    old_path = os.path.join(SAVED_RESULTS_DIR, name)
    new_path = os.path.join(SAVED_RESULTS_DIR, new_name)

    if not os.path.exists(old_path):
        raise HTTPException(status_code=404, detail=f"模型 '{name}' 不存在")
    if os.path.exists(new_path):
        raise HTTPException(status_code=400, detail=f"模型名 '{new_name}' 已存在")

    try:
        os.rename(old_path, new_path)
        logger.info(f"[Models] 模型重命名: {name} -> {new_name}")
        return {"success": True, "message": f"模型已重命名为 '{new_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重命名失败: {str(e)}")


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

    # 验证算法训练可用性
    if request.algorithm_name:
        try:
            from backend.algorithm_availability import get_algorithm_availability
            avail = get_algorithm_availability(request.algorithm_name)
            if avail and not avail.training_available:
                reasons = "; ".join(avail.reasons) if avail.reasons else "该算法暂不支持训练"
                raise HTTPException(
                    status_code=400,
                    detail=f"算法 '{request.algorithm_name}' 当前不可训练: {reasons}"
                )
        except HTTPException:
            raise
        except Exception:
            pass  # 可用性缓存未初始化时跳过

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
        validation_ratio=request.validation_ratio,
        gpu_id=request.gpu_id
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
        "progress_pct": 0.0,
        "message": "",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "model_path": None,
        "log": "",
        "log_file": os.path.join(PROJECT_ROOT, "logs", "training", f"{task_id}.log"),
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


def ensure_meta_json(data_root: str):
    """确保数据目录下存在 meta.json（ADer 框架必需）
    如果不存在则自动扫描目录结构生成，格式与 MVTec AD 的 meta.json 一致。
    目录结构: {category}/train/good/, {category}/test/good/, {category}/test/anomaly/
    """
    meta_path = os.path.join(data_root, "meta.json")
    if os.path.exists(meta_path):
        return

    import glob as _glob
    meta = {"train": {}, "test": {}}
    img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    for category in sorted(os.listdir(data_root)):
        cat_path = os.path.join(data_root, category)
        if not os.path.isdir(cat_path):
            continue
        # 跳过非数据目录
        if category.startswith('.') or category == 'meta.json':
            continue

        train_good = os.path.join(cat_path, "train", "good")
        test_good = os.path.join(cat_path, "test", "good")
        test_anomaly_dir = os.path.join(cat_path, "test", "anomaly")

        # train 数据
        train_entries = []
        if os.path.isdir(train_good):
            for fname in sorted(os.listdir(train_good)):
                if os.path.splitext(fname)[1].lower() in img_exts:
                    train_entries.append({
                        "img_path": f"{category}/train/good/{fname}",
                        "mask_path": "",
                        "cls_name": category,
                        "specie_name": "good",
                        "anomaly": 0
                    })
        if train_entries:
            meta["train"][category] = train_entries

        # test 数据（good + anomaly 子目录）
        test_entries = []
        if os.path.isdir(test_good):
            for fname in sorted(os.listdir(test_good)):
                if os.path.splitext(fname)[1].lower() in img_exts:
                    test_entries.append({
                        "img_path": f"{category}/test/good/{fname}",
                        "mask_path": "",
                        "cls_name": category,
                        "specie_name": "good",
                        "anomaly": 0
                    })
        if os.path.isdir(test_anomaly_dir):
            for fname in sorted(os.listdir(test_anomaly_dir)):
                if os.path.splitext(fname)[1].lower() in img_exts:
                    test_entries.append({
                        "img_path": f"{category}/test/anomaly/{fname}",
                        "mask_path": f"{category}/ground_truth/anomaly/{os.path.splitext(fname)[0]}_mask.png",
                        "cls_name": category,
                        "specie_name": "anomaly",
                        "anomaly": 1
                    })
        if test_entries:
            meta["test"][category] = test_entries

    if meta["train"] or meta["test"]:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[Training] 自动生成 meta.json: {meta_path} "
              f"(train={sum(len(v) for v in meta['train'].values())}, "
              f"test={sum(len(v) for v in meta['test'].values())})")


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


# Dinomaly2 backbone 映射 (支持 DINOv2 和 DINOv3)
_D2_BACKBONE_MAP = {
    ("dinov2", "small"): "dinov2reg_vit_small_14",
    ("dinov2", "base"): "dinov2reg_vit_base_14",
    ("dinov2", "large"): "dinov2reg_vit_large_14",
    ("dinov3", "small"): "dinov3_vit_small_16",
    ("dinov3", "base"): "dinov3_vit_base_16",
    ("dinov3", "large"): "dinov3_vit_large_16",
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
        # CUDA_VISIBLE_DEVICES 已经映射了物理 GPU，子进程始终使用 cuda:0
        # 例如 CUDA_VISIBLE_DEVICES=5 → 子进程内 cuda:0 对应物理 GPU 5
        gpu_id = 0
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

        # DINOv3 需要 use_get_intermediate（缺少 prepare_tokens 方法）
        if "v3" in backbone:
            cmd.append("--use_get_intermediate")

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
        batch_size = config.batch_size

        # 统一训练量语义：total_iters 表示 optimizer step 次数（与 Dinomaly 一致）
        # Anomalib 以 epoch 为单位，需根据数据集大小和 batch_size 换算
        # formula: max_epochs = ceil(total_iters / batches_per_epoch)
        #           batches_per_epoch = ceil(train_images / batch_size)
        total_train = config.dataset_stats.get(
            config.categories[0] if config.categories else "unknown", {}
        ).get("train", 0)
        if total_train > 0:
            batches_per_epoch = max(1, math.ceil(total_train / batch_size))
            max_epochs = max(1, math.ceil(config.total_iters / batches_per_epoch))
            logger.info(
                f"Anomalib 训练量换算: total_iters={config.total_iters} → "
                f"{max_epochs} epochs (train_images={total_train}, batch_size={batch_size}, "
                f"batches_per_epoch={batches_per_epoch})"
            )
        else:
            max_epochs = max(1, config.total_iters // 100)

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
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '{os.path.join(PROJECT_ROOT, "models", "pre_trained", "huggingface")}'
os.environ['TORCH_HOME'] = '{os.path.join(PROJECT_ROOT, "models", "pre_trained")}'
from anomalib.engine import Engine
from anomalib.models import get_model
from lightning.pytorch.callbacks import Callback

class LossStdoutCallback(Callback):
    \"\"\"逐 batch 将 loss 打印到 stdout，供后端日志解析绘制曲线。

    Lightning/Rich 进度条不向 stdout 输出可解析的 loss 文本
    （指标名与数值被 Rich 渲染拆到不同行），因此显式打印
    'Epoch X/Y loss=Z' 行，与后端 Anomalib 解析正则对应。
    \"\"\"
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = None
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = float(outputs['loss'])
        elif outputs is not None and hasattr(outputs, 'item'):
            loss = float(outputs.item())
        else:
            for k, v in trainer.callback_metrics.items():
                if 'loss' in k:
                    try:
                        loss = float(v)
                        break
                    except (TypeError, ValueError):
                        continue
        if loss is not None:
            lr_txt = ''
            try:
                opt = trainer.optimizers[0] if trainer.optimizers else None
                if opt is not None and opt.param_groups:
                    lr_txt = f' lr:{{opt.param_groups[0]["lr"]:.6e}}'
            except (IndexError, KeyError, TypeError):
                pass
            print(f'Epoch {{trainer.current_epoch + 1}}/{{trainer.max_epochs}} loss={{loss:.6f}}{{lr_txt}}', flush=True)

model = get_model('{anomalib_algo}')
engine = Engine(
    max_epochs={max_epochs},
    default_root_dir='{SAVED_RESULTS_DIR}/{save_name}',
    accelerator='gpu',
    devices=1,
    callbacks=[LossStdoutCallback()],
)
from anomalib.data import MVTecAD
datamodule = MVTecAD(root='{data_root}', category='{category}', train_batch_size={batch_size})
engine.fit(model=model, datamodule=datamodule)
print(f'Training completed: {algorithm}')

# 训练完成后在测试集上评估
print('Begin final model eval!!!')
test_results = engine.test(model=model, datamodule=datamodule)
if test_results and isinstance(test_results, list) and len(test_results) > 0:
    result = test_results[0]
    for k, v in result.items():
        if isinstance(v, (int, float)):
            print(f'Test {k}: {v:.4f}')
        else:
            print(f'Test {k}: {v}')
"""
        script_path = os.path.join(SAVED_RESULTS_DIR, f"_anomalib_train_{task_id}.py")
        with open(script_path, 'w') as f:
            f.write(train_script)

        cmd = [sys.executable, script_path]
        # Anomalib 需要 algorithms/ 在 PYTHONPATH 中以找到本地 anomalib 包
        ader_root = os.path.join(PROJECT_ROOT, "algorithms")
        _run_subprocess_with_logging(task_id, cmd, env={"PYTHONPATH": f"{ader_root}:{os.environ.get('PYTHONPATH', '')}"})

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

        # 确保 meta.json 存在（ADer 必需）
        ensure_meta_json(data_root)

        script_path = os.path.join(PROJECT_ROOT, "algorithms", "ADer", "run.py")
        ader_root = os.path.join(PROJECT_ROOT, "algorithms")
        ader_cwd = os.path.join(PROJECT_ROOT, "algorithms", "ADer")

        # 配置路径: 优先使用 benchmark 配置（通用、可维护），通过命令行参数覆盖数据路径等
        # 格式: configs/benchmark/{method}/{method}_256_100e.py
        algo_name = config.algorithm_name or "mambaad"
        method_lower = method_name.lower()
        # 配置目录和文件名前缀：优先查 _ADER_CFG_DIR_MAP（支持 (目录, 文件前缀) 元组），否则使用算法名小写
        cfg_entry = _ADER_CFG_DIR_MAP.get(algo_name, algo_name)
        if isinstance(cfg_entry, tuple):
            cfg_method_dir, cfg_file_prefix = cfg_entry
        else:
            cfg_method_dir = cfg_entry
            cfg_file_prefix = algo_name
        cfg_file_name = f"{cfg_file_prefix}_256_100e.py"

        benchmark_dir = os.path.join(PROJECT_ROOT, "algorithms", "ADer", "configs", "benchmark", cfg_method_dir)
        benchmark_cfg = None
        for candidate in [cfg_file_name, f"{cfg_file_prefix}_mvtec.py", f"{cfg_file_prefix}_256_300e.py"]:
            path = os.path.join(benchmark_dir, candidate)
            if os.path.isfile(path):
                benchmark_cfg = path
                break

        if not benchmark_cfg:
            raise FileNotFoundError(f"ADer benchmark 配置不存在: {benchmark_dir}/{cfg_file_name}")

        # 设置 PYTHONPATH 使 ADer 内部模块可导入
        env = os.environ.copy()
        python_path = f"{ader_root}:{env.get('PYTHONPATH', '')}"
        env["PYTHONPATH"] = python_path

        # CWD 必须是 algorithms/ADer/ 目录，这样：
        # - ADer 内部所有相对导入（from util.xxx, from model.xxx, from configs.xxx）正常工作
        # - trainer/model 注册到正确的 TRAINER/MODEL 实例，避免实例分裂
        # - glob 路径（trainer/[!_]*.py 等）正确匹配
        # 配置路径（相对于 algorithms/ADer/）
        cfg_path = f"configs/benchmark/{cfg_method_dir}/{os.path.basename(benchmark_cfg)}"

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

        # 统一训练量语义：total_iters 表示 optimizer step 次数（与 Dinomaly 一致）
        # ADer 以 epoch 为单位，需根据数据集大小和 batch_size 换算
        # ADer 内部 scheduler.py: iter_full = train_size * epoch_full，其中 train_size = batches_per_epoch
        # formula: epoch_full = ceil(total_iters / batches_per_epoch)
        total_train = sum(s.get("train", 0) for s in config.dataset_stats.values())
        if total_train > 0:
            # ADer 使用自身配置文件的 batch_size（与 TrainingConfig.batch_size 可能不同）
            # 用 config.batch_size 近似，典型 ADer 配置 batch_size=16
            batches_per_epoch = max(1, math.ceil(total_train / config.batch_size))
            epoch_full = max(1, math.ceil(config.total_iters / batches_per_epoch))
            logger.info(
                f"ADer 训练量换算: total_iters={config.total_iters} → "
                f"{epoch_full} epochs (train_images={total_train}, batch_size≈{config.batch_size}, "
                f"batches_per_epoch≈{batches_per_epoch})"
            )
        else:
            epoch_full = max(1, config.total_iters // 100)
        test_start_epoch = epoch_full
        test_per_epoch = max(1, epoch_full // 10)

        cmd = [
            sys.executable, script_path,
            "-c", cfg_path,
            "-m", "train",
            f"data.root={data_rel}",
            f"trainer.checkpoint={checkpoint_rel}",
            f"data.cls_names={config.categories}",
            f"epoch_full={epoch_full}",
            f"test_start_epoch={test_start_epoch}",
            f"test_per_epoch={test_per_epoch}",
            f"trainer.epoch_full={epoch_full}",
            f"trainer.test_start_epoch={test_start_epoch}",
            f"trainer.test_per_epoch={test_per_epoch}",
            "use_adeval=False",
            "metrics=['mAUROC_sp_max','mAUROC_sp_mean']",
            "evaluator.kwargs.use_adeval=False",
            "evaluator.kwargs.metrics=['mAUROC_sp_max','mAUROC_sp_mean']",
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
        "destseg": "DeSTSeg",
        "realnet": "RealNet",
        "rdpp": "RDPTrainer",
    }
    return mapping.get(algo_name, "MambaAD")


# ADer 算法名 → 配置目录名/文件名前缀的映射（部分算法的配置目录名与算法名不同）
_ADER_CFG_DIR_MAP = {
    # 格式: 算法ID → (配置目录名, 配置文件名前缀)
    # 仅当目录名或文件名前缀与算法ID不同时才需配置
    "rdpp": ("rd++", "rdpp"),       # rdpp 配置在 rd++/ 目录下，文件名 rdpp_256_100e.py
    "unad": ("uniad", "uniad"),     # unad 配置目录和文件名前缀都是 uniad（带 i）
}


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

    # 创建日志文件
    log_file_path = task.get("log_file")
    log_fh = None
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        log_fh = open(log_file_path, 'w', encoding='utf-8')
        log_fh.write(task["log"])
        log_fh.flush()

    venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        cmd[0] = venv_python

    # 设置环境变量
    env_vars = {**os.environ}
    if env:
        env_vars.update(env)
    # 根据 gpu_id 设置 CUDA_VISIBLE_DEVICES
    # gpu_id 是物理 GPU 编号，通过 CUDA_VISIBLE_DEVICES 映射到子进程的 cuda:0
    gpu_id = task.get("config", {}).get("gpu_id", 0)
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(f"[{task_id}] 使用 GPU {gpu_id}, CUDA_VISIBLE_DEVICES={gpu_id}")
    env_vars["DINOMALY_ENCODER_DIR"] = os.path.join(PROJECT_ROOT, "models", "pre_trained")
    env_vars["PRETRAINED_MODELS_DIR"] = os.path.join(PROJECT_ROOT, "models", "pre_trained")
    env_vars["TORCH_HOME"] = os.path.join(PROJECT_ROOT, "models", "pre_trained")
    # HuggingFace 离线模式：优先从本地缓存加载，避免网络不通导致训练失败
    env_vars["HF_HUB_OFFLINE"] = "1"
    env_vars["HF_HOME"] = os.path.join(PROJECT_ROOT, "models", "pre_trained", "huggingface")

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

    logger.info(f"[{task_id}] 训练子进程已启动, PID={process.pid}, cmd={' '.join(cmd[:5])}...")

    task["process"] = process

    # 需要在页面日志中过滤的噪音行（warning、info 提示等）
    _LOG_SKIP_PATTERNS = [
        re.compile(r'^\s*$'),                                    # 空行
        re.compile(r'UserWarning'),                              # PyTorch UserWarning
        re.compile(r'FutureWarning'),                            # FutureWarning
        re.compile(r'DeprecationWarning'),                       # DeprecationWarning
        re.compile(r'is deprecated'),                            # 弃用警告
        re.compile(r'__new__\(\) got an unexpected'),            # pytorch_lightning 兼容警告
        re.compile(r'Consider setting'),                         # Lightning 配置建议
        re.compile(r'GPU available:'),                           # Lightning GPU 检测
        re.compile(r'LOCAL_RANK: 0'),                            # 分布式环境变量
        re.compile(r'CUDA_VISIBLE_DEVICES'),                     # CUDA 环境
    ]

    iter_start_time = time.time()
    for line in process.stdout:
        task = TRAINING_TASKS.get(task_id)
        if not task:
            return

        # 过滤噪音日志行（warning、环境信息等）
        skip_line = any(p.search(line) for p in _LOG_SKIP_PATTERNS)
        if not skip_line:
            task["log"] += line
        if log_fh:
            log_fh.write(line)
            log_fh.flush()

        # 解析迭代/损失信息（通用解析）
        # Dinomaly 格式: Iter [X/Y] ... Loss: Z
        iter_match = re.search(r'Iter\s*\[(\d+)/(\d+)\].*Loss:\s*([\d.]+)', line, re.IGNORECASE)

        # ADer 标准格式: Train: XX% [iter/total] [epoch/total] [loss_name value (avg)] ...
        if not iter_match:
            ader_match = re.search(r'Train:\s*([\d.]+)%\s*\[(\d+)/(\d+)\]', line)
            if ader_match:
                current_iter = int(ader_match.group(2))
                total_iters = int(ader_match.group(3))
                # 从 [name value] 对中提取 loss 值，跳过计时/lr 项
                # 格式: [name value] 或 [name value (add_name avg)]
                # ADer AvgMeter.__str__: [name val (add_name avg)] — add_name 可以是 'avg' 等字样
                loss_terms = re.findall(r'\[(\w+)\s+([\d.]+)(?:\s*\([^)]*\))?\]', line)
                skip_terms = {'batch_t', 'data_t', 'optim_t', 'lr'}
                loss = None
                for name, val in loss_terms:
                    if name not in skip_terms:
                        loss = float(val)
                        break

                # 始终更新进度（即使没有 loss 值）
                task["current_iter"] = current_iter
                task["progress"] = f"训练进度: {current_iter}/{total_iters} ({current_iter/total_iters*100:.1f}%)"

                # 更新 ETA
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

                if loss is not None:
                    task["metrics"]["loss_history"].append(loss)
                    if len(task["metrics"]["loss_history"]) > 1000:
                        task["metrics"]["loss_history"] = task["metrics"]["loss_history"][-1000:]

                    if loss < task["metrics"]["best_loss"]:
                        task["metrics"]["best_loss"] = loss
                        task["metrics"]["best_iter"] = current_iter

                iter_match = None  # 已处理，避免重复

        # CFlow 格式: Epoch: X.Y \t train loss: Z, lr=W
        if not iter_match:
            cflow_match = re.search(r'Epoch:\s*(\d+)\.(\d+)\s+train loss:\s*([\d.]+)', line)
            if cflow_match:
                meta_epoch = int(cflow_match.group(1))
                sub_epoch = int(cflow_match.group(2))
                loss = float(cflow_match.group(3))

                task["metrics"]["loss_history"].append(loss)
                if len(task["metrics"]["loss_history"]) > 1000:
                    task["metrics"]["loss_history"] = task["metrics"]["loss_history"][-1000:]

                if loss < task["metrics"]["best_loss"]:
                    task["metrics"]["best_loss"] = loss

                task["progress"] = f"训练进度: Epoch {meta_epoch}.{sub_epoch}, loss={loss:.4f}"

                iter_end_time = time.time()
                iter_duration = iter_end_time - iter_start_time
                task["metrics"]["iter_time_history"].append(iter_duration)
                if len(task["metrics"]["iter_time_history"]) > 100:
                    task["metrics"]["iter_time_history"] = task["metrics"]["iter_time_history"][-100:]
                iter_start_time = iter_end_time

                iter_match = None  # 已处理，避免重复

        # Anomalib/Lightning 格式: Epoch X/Y ... loss=Z 或 loss: Z
        if not iter_match:
            iter_match = re.search(r'Epoch\s+(\d+)(?:/(\d+))?.*?loss[=:]\s*([\d.]+)', line, re.IGNORECASE)
        # Dinomaly2 格式: Iter [X/Y] ... Loss: Z (same as Dinomaly, already matched above)
        if iter_match:
            current_iter = int(iter_match.group(1))
            total_iters = int(iter_match.group(2)) if iter_match.group(2) else 0
            loss = float(iter_match.group(3))

            task["current_iter"] = current_iter
            if total_iters > 0:
                task["progress"] = f"训练进度: {current_iter}/{total_iters} ({current_iter/total_iters*100:.1f}%)"
            else:
                task["progress"] = f"训练进度: Epoch {current_iter}, loss={loss:.4f}"

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
        if not lr_match:
            # ADer 格式: [lr 0.000100]
            lr_match = re.search(r'\[lr\s+([\d.eE+-]+)\]', line)
        if lr_match:
            task["metrics"]["learning_rate_history"].append(float(lr_match.group(1)))
            if len(task["metrics"]["learning_rate_history"]) > 1000:
                task["metrics"]["learning_rate_history"] = task["metrics"]["learning_rate_history"][-1000:]

        if len(task["log"]) > 50000:
            task["log"] = task["log"][-40000:]

    process.wait()

    # 关闭日志文件
    if log_fh:
        log_fh.close()
        log_fh = None

    task = TRAINING_TASKS.get(task_id)
    if not task:
        return

    if process.returncode == 0:
        task["status"] = "completed"
        task["progress"] = "训练完成"
        logger.info(f"[{task_id}] 训练完成, 模型保存目录: {SAVED_RESULTS_DIR}")

        # 从训练日志中解析测试集性能指标
        test_metrics = _parse_test_metrics(
            task.get("log", ""),
            task.get("algorithm_family", ""),
        )
        if test_metrics:
            task["test_metrics"] = test_metrics
            logger.info(f"[{task_id}] 解析到测试指标: {json.dumps(test_metrics, ensure_ascii=False)}")

        save_dir = SAVED_RESULTS_DIR

        # 查找模型文件：优先 .pth 文件，其次目录
        model_found = False
        for f in os.listdir(save_dir):
            if task["save_name"] in f:
                fpath = os.path.join(save_dir, f)
                if f.endswith('.pth') or os.path.isdir(fpath):
                    task["model_path"] = fpath
                    model_found = True
                    _save_training_metadata(save_dir, f, task)
                    break

        # 兜底：按修改时间查找训练期间新创建的最新的 .pth 文件
        if not model_found:
            candidates = []
            started = task.get("started_at")
            started_ts = None
            if started:
                try:
                    started_ts = datetime.fromisoformat(started).timestamp()
                except (ValueError, TypeError):
                    pass
            for f in os.listdir(save_dir):
                fpath = os.path.join(save_dir, f)
                if f.endswith('.pth') or os.path.isdir(fpath):
                    mtime = os.path.getmtime(fpath)
                    if started_ts is None or mtime >= started_ts - 10:
                        candidates.append((mtime, fpath))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                task["model_path"] = candidates[0][1]
                model_found = True
                _save_training_metadata(save_dir, os.path.basename(candidates[0][1]), task)
                logger.info(f"[{task_id}] 兜底匹配到模型文件: {candidates[0][1]}")

        # ADer 框架不使用 save_name 命名，需按训练器名称模式查找
        if not model_found and task.get("algorithm_family") == "ader":
            from backend.core.model_meta import _ADER_TRAINER_MAP
            method_name = _ader_method_name(task.get("algorithm_name", "mambaad"))
            for trainer_name, algo_id in _ADER_TRAINER_MAP.items():
                if algo_id == task.get("algorithm_name", ""):
                    # 找到最新的匹配目录
                    candidates = []
                    for f in os.listdir(save_dir):
                        if f.startswith(trainer_name) and os.path.isdir(os.path.join(save_dir, f)):
                            # 检查是否在训练时间范围内（启发式）
                            candidates.append(f)
                    if candidates:
                        # 按修改时间排序取最新的
                        candidates.sort(
                            key=lambda x: os.path.getmtime(os.path.join(save_dir, x)),
                            reverse=True
                        )
                        fpath = os.path.join(save_dir, candidates[0])
                        task["model_path"] = fpath
                        model_found = True
                        _save_training_metadata(save_dir, candidates[0], task)
                        break

        # 保存阈值信息到模型目录（供自定义检测自动读取）
        test_metrics = task.get("test_metrics", {})
        if test_metrics and test_metrics.get("optimal_threshold") and task.get("model_path"):
            th_info = test_metrics["optimal_threshold"]
            model_path = task["model_path"]
            if os.path.isdir(model_path):
                # 目录模型: {model_dir}/threshold_info.json
                th_file = os.path.join(model_path, "threshold_info.json")
            else:
                # 文件模型: {model_name}.threshold_info.json（避免多模型同目录冲突）
                th_file = model_path + ".threshold_info.json"
            try:
                with open(th_file, "w", encoding="utf-8") as f:
                    json.dump(th_info, f, ensure_ascii=False, indent=2)
                logger.info(f"[{task_id}] 阈值信息已保存: {th_file}")
            except OSError as e:
                logger.warning(f"[{task_id}] 保存阈值信息失败: {e}")

        # 注: 曾在此创建 {algo}_best.pth 链接/副本,但无任何消费方且会在
        # 离线检测列表中产生无法解析的 custom: 条目,已移除生成逻辑。
        # 列表过滤与删除清理仍保留,用于兼容历史遗留的 _best.pth 文件。
    else:
        task["status"] = "failed"
        task["progress"] = f"训练失败 (退出码: {process.returncode})"
        logger.error(f"[{task_id}] 训练失败, 退出码: {process.returncode}")

    task["completed_at"] = datetime.now().isoformat()
    task["process"] = None
    _save_training_tasks()


def _parse_test_metrics(log_text: str, algorithm_family: str) -> dict:
    """从训练 stdout 日志中解析测试集性能指标

    支持的格式:
      Dinomaly V1/V3:  {item}: I-Auroc:0.xx, I-AP:0.xx, I-F1:0.xx
                       Mean: I-Auroc:0.xx, I-AP:0.xx, I-F1:0.xx
      Dinomaly2:       {item}: I-Auroc:0.xx, ..., P-AUROC:0.xx, ..., P-AUPRO:0.xx
                       Mean: I-Auroc:0.xx, ..., P-AUROC:0.xx, ..., P-AUPRO:0.xx
      Anomalib:        Test {metric_name}: {value}
    """
    metrics = {}

    # 定位最后一段评估输出（优先取 "Begin final model eval!!!" 之后的文本）
    eval_start = log_text.rfind("Begin final model eval!!!")
    if eval_start < 0:
        eval_start = log_text.rfind("Begin model eval!!!")
    eval_section = log_text[eval_start:] if eval_start >= 0 else log_text

    # ── Dinomaly2 格式（含 P-AUROC / P-AUPRO） ──
    d2_per_class = re.findall(
        r'^(\S+):\s+I-Auroc:([\d.]+),\s*I-AP:([\d.]+),\s*I-F1:([\d.]+),\s*'
        r'P-AUROC:([\d.]+),\s*P-AP:([\d.]+),\s*P-F1:([\d.]+),\s*P-AUPRO:([\d.]+)',
        eval_section, re.MULTILINE
    )
    d2_mean = re.search(
        r'^Mean:\s+I-Auroc:([\d.]+),\s*I-AP:([\d.]+),\s*I-F1:([\d.]+),\s*'
        r'P-AUROC:([\d.]+),\s*P-AP:([\d.]+),\s*P-F1:([\d.]+),\s*P-AUPRO:([\d.]+)',
        eval_section, re.MULTILINE
    )

    if d2_per_class:
        metrics["per_class"] = {}
        for item, ia, iap, if1, pa, pap, pf1, paupro in d2_per_class:
            if item != "Mean":
                metrics["per_class"][item] = {
                    "I-AUROC": float(ia), "I-AP": float(iap), "I-F1": float(if1),
                    "P-AUROC": float(pa), "P-AP": float(pap), "P-F1": float(pf1),
                    "P-AUPRO": float(paupro),
                }
    if d2_mean:
        metrics["mean"] = {
            "I-AUROC": float(d2_mean.group(1)), "I-AP": float(d2_mean.group(2)),
            "I-F1": float(d2_mean.group(3)), "P-AUROC": float(d2_mean.group(4)),
            "P-AP": float(d2_mean.group(5)), "P-F1": float(d2_mean.group(6)),
            "P-AUPRO": float(d2_mean.group(7)),
        }

    # ── Dinomaly V1/V3 格式（仅 I-Auroc / I-AP / I-F1） ──
    if not metrics.get("mean"):
        d1_per_class = re.findall(
            r'^(\S+):\s+I-Auroc:([\d.]+),\s*I-AP:([\d.]+),\s*I-F1:([\d.]+)',
            eval_section, re.MULTILINE
        )
        d1_mean = re.search(
            r'^Mean:\s+I-Auroc:([\d.]+),\s*I-AP:([\d.]+),\s*I-F1:([\d.]+)',
            eval_section, re.MULTILINE
        )
        if d1_per_class:
            metrics["per_class"] = {}
            for item, ia, iap, if1 in d1_per_class:
                if item != "Mean":
                    metrics["per_class"][item] = {
                        "I-AUROC": float(ia), "I-AP": float(iap), "I-F1": float(if1),
                    }
        if d1_mean:
            metrics["mean"] = {
                "I-AUROC": float(d1_mean.group(1)),
                "I-AP": float(d1_mean.group(2)),
                "I-F1": float(d1_mean.group(3)),
            }

    # ── Anomalib 格式: Test {metric}: {value} ──
    test_kv = re.findall(r'^Test\s+([^:]+):\s*([\d.]+)', eval_section, re.MULTILINE)
    if test_kv:
        if "mean" not in metrics:
            metrics["mean"] = {}
        for name, value in test_kv:
            # 将 Anomalib 的指标名标准化（例如 image_AUROC → I-AUROC）
            normalized = name.replace("image_", "I-").replace("pixel_", "P-")
            metrics["mean"][normalized] = float(value)

    # ── 最优决策阈值 ──
    th_match = re.search(
        r'^Optimal threshold:\s*([\d.]+)\s*\(F1=([\d.]+),\s*method=(\S+)\)',
        eval_section, re.MULTILINE
    )
    if th_match:
        metrics["optimal_threshold"] = {
            "value": float(th_match.group(1)),
            "f1": float(th_match.group(2)),
            "method": th_match.group(3),
        }

    return metrics


def _save_training_metadata(save_dir: str, model_entry: str, task: dict):
    """训练完成后保存 metadata.json 到模型目录"""
    from backend.core.model_meta import save_model_meta
    meta = {
        "algorithm_family": task.get("algorithm_family", ""),
        "algorithm_name": task.get("algorithm_name", ""),
        "model_type": task.get("model_type", ""),
        "model_size": task.get("model_size", ""),
        "category": "_".join(task.get("categories", [])),
        "data_source": task.get("data_source", ""),
        "total_iters": task.get("total_iters", 0),
        "batch_size": task.get("config", {}).get("batch_size", 0),
        "learning_rate": task.get("config", {}).get("learning_rate", 0),
        "completed_at": task.get("completed_at", ""),
    }
    save_model_meta(save_dir, model_entry, **meta)


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
        algorithm_name=task.get("algorithm_name", ""),
        model_type=task.get("model_type", ""),
        model_size=task.get("model_size", ""),
        data_source=task.get("data_source", ""),
        error=task.get("error"),
        total_iters=task["total_iters"],
        current_iter=task.get("current_iter", 0),
        progress=task["progress"],
        message=task["message"],
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        model_path=task.get("model_path"),
        log=task["log"][-5000:] if task.get("log") else "",
        metrics=task.get("metrics", {}),
        test_metrics=task.get("test_metrics"),
        loss_history=task.get("metrics", {}).get("loss_history", []),
        learning_rate=task.get("config", {}).get("learning_rate", 0.0001),
        estimated_time_remaining=task.get("estimated_time_remaining")
    )


@router.get("/log/{task_id}")
async def get_training_full_log(task_id: str):
    """获取完整训练日志（页面状态接口仅返回末尾 5000 字符）"""
    # 防路径穿越：task_id 仅允许字母数字下划线
    if not re.fullmatch(r'[A-Za-z0-9_]+', task_id):
        raise HTTPException(status_code=400, detail="非法任务 ID")

    log_path = os.path.join(PROJECT_ROOT, "logs", "training", f"{task_id}.log")
    if os.path.exists(log_path):
        return FileResponse(log_path, media_type="text/plain; charset=utf-8",
                            filename=f"{task_id}.log")

    # 日志文件缺失时回退到内存日志（旧任务/异常场景）
    task = TRAINING_TASKS.get(task_id)
    if task and task.get("log"):
        return PlainTextResponse(task["log"])
    raise HTTPException(status_code=404, detail="日志不存在")


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
    _save_training_tasks()

    return {"success": True, "message": "训练已停止"}
