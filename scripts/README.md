# 模型下载脚本使用说明

## 概述

本项目提供了一个完整的模型下载和管理系统，用于自动下载ASD_for_SPK项目所需的所有预训练模型。

## 功能特性

- **自动识别模型**: 自动识别项目中所有需要使用的预训练模型
- **断点续传**: 支持下载中断后从断点继续下载
- **完整性验证**: 支持文件大小和哈希值验证
- **进度反馈**: 实时显示下载进度和速度
- **灵活配置**: 支持通过配置文件或命令行参数指定模型版本和来源

## 文件结构

```
scripts/
├── download_models.py      # 模型下载脚本主程序
├── models_config.yaml      # 模型配置文件
├── models_registry.json    # 导出的模型注册表（由脚本生成）
└── README.md              # 本文件
```

## 使用方法

### 1. 下载所有必需模型

```bash
python scripts/download_models.py --all
```

### 2. 下载所有模型（包括可选）

```bash
python scripts/download_models.py --all --include-optional
```

### 3. 下载特定模型

```bash
python scripts/download_models.py --model dinov2-large
```

### 4. 验证已下载模型

```bash
python scripts/download_models.py --verify
```

### 5. 强制重新下载

```bash
python scripts/download_models.py --all --force
```

### 6. 导出模型配置

```bash
python scripts/download_models.py --export-config models_config.json
```

### 7. 指定预训练目录

```bash
python scripts/download_models.py --all --pretrained-dir /path/to/pre_trained
```

## 支持的模型来源

### HuggingFace 模型
- facebook/dinov2-small
- facebook/dinov2-base
- facebook/dinov2-large
- facebook/dinov2-with-registers-small
- facebook/dinov2-with-registers-base
- facebook/dinov2-with-registers-large
- facebook/dinov2-with-registers-giant

### OpenCLIP 模型
- openai/ViT-B-32
- openai/ViT-B-16
- openai/ViT-L-14
- openai/ViT-L-14-336px

### 直接下载模型
- dinov2_vits14_pretrain.pth
- dinov2_vitb14_pretrain.pth
- dinov2_vitl14_pretrain.pth
- resnet18-f37072fd.pth
- wide_resnet50_2-95faca4d.pth

### timm 模型
- resnet200
- efficientnet_b5

## 模型存储结构

下载的模型将统一存放于 `pre_trained` 目录：

```
pre_trained/
├── huggingface/              # HuggingFace模型缓存
│   └── hub/
├── dinov2-small/             # DINOv2 Small模型
├── dinov2-base/              # DINOv2 Base模型
├── dinov2-large/             # DINOv2 Large模型
├── dinov2-with-registers-small/
├── dinov2-with-registers-base/
├── dinov2-with-registers-large/
├── dinov2-with-registers-giant/
├── dinov2_vits14_pretrain.pth
├── dinov2_vitb14_pretrain.pth
├── dinov2_vitl14_pretrain.pth
├── ViT-B-32.pt
├── ViT-B-16.pt
├── ViT-L-14.pt
├── ViT-L-14-336px.pt
├── resnet18-f37072fd.pth
├── wide_resnet50_2-95faca4d.pth
└── timm/                     # timm模型缓存
```

## 环境变量配置

在运行项目前，建议设置以下环境变量：

```bash
# HuggingFace缓存目录
export HF_HOME="pre_trained/huggingface"
export TRANSFORMERS_CACHE="pre_trained/huggingface"
export HUGGINGFACE_HUB_CACHE="pre_trained/huggingface"

# PyTorch缓存目录
export TORCH_HOME="pre_trained"

# timm缓存目录
export TIMM_CACHE="pre_trained/timm"
```

## 模型信息

### 必需模型（9个）

| 模型名称 | 大小 | 用途 |
|---------|------|------|
| dinov2-with-registers-large | ~1.1GB | SubspaceAD少样本检测（推荐） |
| clip-vit-b-32 | ~340MB | MuSc零样本检测（快速） |
| clip-vit-b-16 | ~340MB | MuSc零样本检测（标准） |
| clip-vit-l-14 | ~890MB | MuSc零样本检测（高精度） |
| clip-vit-l-14-336 | ~890MB | MuSc零样本检测（高分辨率） |
| dinov2_vits14_pretrain | ~88MB | Dinomaly轻量级检测 |
| dinov2_vitb14_pretrain | ~330MB | Dinomaly标准检测 |
| dinov2_vitl14_pretrain | ~1.1GB | Dinomaly高精度检测 |
| resnet18 | ~45MB | PatchCore/PaDiM骨干网络 |
| wide-resnet50-2 | ~132MB | PatchCore/PaDiM骨干网络 |

**必需模型总大小**: 约 **5.2GB**

### 可选模型（9个）

包括其他DINOv2变体和timm模型库中的模型。

**所有模型总大小**: 约 **11GB**

## 依赖要求

```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.35.0
timm==0.9.12
open_clip_torch
huggingface_hub>=0.19.0
requests>=2.28.0
pyyaml>=6.0
```

## 常见问题

### Q1: 下载速度慢怎么办？

A: 可以使用镜像源或代理：
```bash
# 使用HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 然后运行下载脚本
python scripts/download_models.py --all
```

### Q2: 如何只下载特定算法的模型？

A: 目前需要手动指定模型名称：
```bash
# 只下载SubspaceAD需要的模型
python scripts/download_models.py --model dinov2-with-registers-large

# 只下载MuSc需要的模型
python scripts/download_models.py --model clip-vit-l-14
```

### Q3: 下载中断后如何继续？

A: 脚本自动支持断点续传，直接重新运行即可：
```bash
python scripts/download_models.py --all
```

### Q4: 如何验证模型完整性？

A: 使用 `--verify` 选项：
```bash
python scripts/download_models.py --verify
```

## 日志文件

下载日志会保存在当前目录的 `model_download.log` 文件中，可以通过查看该文件了解下载详情。

## 技术支持

如有问题，请查看项目文档或提交Issue。
