# ASD_for_SPK 项目模型全面分析报告

## 1. 项目模型概览

本项目是一个音频异常检测系统，使用多种深度学习模型进行零样本和少样本异常检测。以下是项目中使用的所有模型的详细分析。

---

## 2. 模型分类与用途

### 2.1 少样本异常检测模型 (SubspaceAD)

#### DINOv2 系列模型

| 模型名称 | 参数数量 | 模型大小 | 用途 | 加载方式 |
|---------|---------|---------|------|---------|
| facebook/dinov2-small | 22M | ~88MB | 轻量级少样本检测 | HuggingFace transformers |
| facebook/dinov2-base | 86M | ~330MB | 标准少样本检测 | HuggingFace transformers |
| facebook/dinov2-large | 300M | ~1.1GB | 高精度少样本检测 | HuggingFace transformers |
| facebook/dinov2-with-registers-small | 22M | ~88MB | 改进版轻量级检测 | HuggingFace transformers |
| facebook/dinov2-with-registers-base | 86M | ~330MB | 改进版标准检测 | HuggingFace transformers |
| facebook/dinov2-with-registers-large | 300M | ~1.1GB | 改进版高精度检测 | HuggingFace transformers |
| facebook/dinov2-with-registers-giant | 1.1B | ~2.3GB | 最高精度检测 | HuggingFace transformers |

**技术细节：**
- **加载库**: `transformers>=4.35.0`
- **加载API**: `AutoModel.from_pretrained()`, `AutoImageProcessor.from_pretrained()`
- **加载参数**:
  ```python
  model = AutoModel.from_pretrained(
      "facebook/dinov2-with-registers-large",
      local_files_only=True,  # 强制从本地加载
      torch_dtype=torch.float32,
      low_cpu_mem_usage=True
  )
  ```
- **输入尺寸**: 224x224, 336x336, 518x518, 672x672
- **特征层**: 根据模型大小选择不同层 (Small: 最后4层, Base: 最后6层, Large: 最后7层)

---

### 2.2 零样本异常检测模型 (MuSc)

#### CLIP 系列模型

| 模型名称 | 参数数量 | 模型大小 | 用途 | 加载方式 |
|---------|---------|---------|------|---------|
| ViT-B-32 | 149M | ~340MB | 快速零样本检测 | open_clip |
| ViT-B-16 | 149M | ~340MB | 标准零样本检测 | open_clip |
| ViT-L-14 | 427M | ~890MB | 高精度零样本检测 | open_clip |
| ViT-L-14-336px | 427M | ~890MB | 高分辨率零样本检测 | open_clip |

**技术细节：**
- **加载库**: `open_clip_torch`
- **加载API**: `open_clip.create_model_and_transforms()`
- **加载参数**:
  ```python
  model, _, preprocess = open_clip.create_model_and_transforms(
      'ViT-L-14',
      pretrained='openai',
      img_size=336  # 可选: 224, 336, 518
  )
  ```
- **预训练来源**: OpenAI
- **特征层**: ViT-B使用[2,5,8,11], ViT-L使用[5,11,17,23]

#### DINOv2 预训练权重 (用于MuSc)

| 模型名称 | 参数数量 | 模型大小 | 用途 | 加载方式 |
|---------|---------|---------|------|---------|
| dinov2_vits14_pretrain.pth | 22M | ~88MB | 轻量级特征提取 | torch.load |
| dinov2_vitb14_pretrain.pth | 86M | ~330MB | 标准特征提取 | torch.load |
| dinov2_vitl14_pretrain.pth | 300M | ~1.1GB | 高精度特征提取 | torch.load |

**技术细节：**
- **加载库**: `torch>=2.1.0`, `timm==0.9.12`
- **加载API**: `torch.load()`
- **下载URL**:
  - Small: `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth`
  - Base: `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth`
  - Large: `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth`

---

### 2.3 Dinomaly 异常检测模型

#### DINOv2 预训练权重 (用于Dinomaly)

| 模型名称 | 参数数量 | 模型大小 | 用途 | 加载方式 |
|---------|---------|---------|------|---------|
| dinov2_vits14_pretrain.pth | 22M | ~88MB | Dinomaly轻量级检测 | torch.load |
| dinov2_vitb14_pretrain.pth | 86M | ~330MB | Dinomaly标准检测 | torch.load |
| dinov2_vitl14_pretrain.pth | 300M | ~1.1GB | Dinomaly高精度检测 | torch.load |

**技术细节：**
- **加载库**: `torch>=2.1.0`, `timm==0.9.12`
- **加载代码位置**: `algorithms/Dinomaly/models/vit_encoder.py`
- **加载参数**:
  ```python
  state_dict = torch.load(ckpt_pth, map_location='cpu')
  if 'model' in state_dict:
      state_dict = state_dict['model']
  model.load_state_dict(state_dict, strict=False)
  ```

---

### 2.4 骨干网络模型

#### ResNet 系列

| 模型名称 | 参数数量 | 模型大小 | 用途 | 加载方式 |
|---------|---------|---------|------|---------|
| resnet18 | 11M | ~45MB | PatchCore/PaDiM轻量级骨干 | torch.load |
| wide_resnet50_2 | 68.9M | ~132MB | PatchCore/PaDiM标准骨干 | torch.load |

**技术细节：**
- **加载库**: `torch>=2.1.0`, `torchvision>=0.16.0`
- **下载URL**:
  - ResNet18: `https://download.pytorch.org/models/resnet18-f37072fd.pth`
  - Wide ResNet50-2: `https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth`
- **加载参数**:
  ```python
  state_dict = torch.load(model_path, map_location='cpu')
  model.load_state_dict(state_dict)
  ```

#### timm 模型库

| 模型名称 | 参数数量 | 模型大小 | 用途 | 加载方式 |
|---------|---------|---------|------|---------|
| resnet200 | 64.7M | ~250MB | MuSc深层骨干 | timm.create_model |
| efficientnet_b5 | 30M | ~118MB | MuSc高效骨干 | timm.create_model |
| efficientnet_b7 | 66M | ~256MB | MuSc高精度骨干 | timm.create_model |
| vit_small_patch8_224 | 22M | ~88MB | MuSc ViT骨干 | timm.create_model |
| vit_base_patch8_224 | 86M | ~330MB | MuSc ViT骨干 | timm.create_model |
| vit_large_patch8_224 | 300M | ~1.1GB | MuSc ViT骨干 | timm.create_model |

**技术细节：**
- **加载库**: `timm==0.9.12`
- **加载API**: `timm.create_model()`
- **加载参数**:
  ```python
  model = timm.create_model(
      'resnet200',
      pretrained=True,
      num_classes=0,  # 移除分类头
      global_pool=''  # 移除全局池化
  )
  ```

---

## 3. 依赖库版本要求

### 核心依赖

| 库名称 | 版本要求 | 用途 |
|-------|---------|------|
| torch | >=2.1.0 | 深度学习框架 |
| torchvision | >=0.16.0 | 图像处理 |
| torchaudio | >=2.1.0 | 音频处理 |
| transformers | >=4.35.0 | HuggingFace模型加载 |
| timm | ==0.9.12 | 图像模型库 |
| open_clip_torch | latest | CLIP模型加载 |
| huggingface_hub | >=0.19.0 | 模型下载 |

### 可选依赖

| 库名称 | 版本要求 | 用途 |
|-------|---------|------|
| mamba_ssm | >=2.0.0 | MambaAD算法 |
| causal_conv1d | >=1.5.0 | MambaAD算法 |

---

## 4. 模型加载路径配置

### 4.1 环境变量配置

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

### 4.2 本地模型路径映射

```yaml
# config/config.yaml
models:
  pretrained_dir: "pre_trained"
  
  subspacead:
    dinov2-small: "pre_trained/dinov2-small"
    dinov2-base: "pre_trained/dinov2-base"
    dinov2-large: "pre_trained/dinov2-large"
    dinov2-with-registers-large: "pre_trained/dinov2-with-registers-large"
  
  musc:
    ViT-B-32: "pre_trained/ViT-B-32.pt"
    ViT-B-16: "pre_trained/ViT-B-16.pt"
    ViT-L-14: "pre_trained/ViT-L-14.pt"
    dinov2_vits14: "pre_trained/dinov2_vits14_pretrain.pth"
    dinov2_vitb14: "pre_trained/dinov2_vitb14_pretrain.pth"
    dinov2_vitl14: "pre_trained/dinov2_vitl14_pretrain.pth"
  
  dinomaly:
    dinov2_vits14: "pre_trained/dinov2_vits14_pretrain.pth"
    dinov2_vitb14: "pre_trained/dinov2_vitb14_pretrain.pth"
    dinov2_vitl14: "pre_trained/dinov2_vitl14_pretrain.pth"
  
  resnet:
    resnet18: "pre_trained/resnet18-f37072fd.pth"
    wide_resnet50_2: "pre_trained/wide_resnet50_2-95faca4d.pth"
```

---

## 5. 模型存储目录结构

```
pre_trained/
├── huggingface/                          # HuggingFace模型缓存
│   ├── hub/
│   │   ├── models--facebook--dinov2-small/
│   │   ├── models--facebook--dinov2-base/
│   │   ├── models--facebook--dinov2-large/
│   │   └── ...
│   └── version.txt
│
├── dinov2-small/                         # DINOv2 Small模型文件
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
│
├── dinov2-base/                          # DINOv2 Base模型文件
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
│
├── dinov2-large/                         # DINOv2 Large模型文件
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
│
├── dinov2-with-registers-small/          # DINOv2 Small with Registers
├── dinov2-with-registers-base/           # DINOv2 Base with Registers
├── dinov2-with-registers-large/          # DINOv2 Large with Registers
├── dinov2-with-registers-giant/          # DINOv2 Giant with Registers
│
├── dinov2_vits14_pretrain.pth            # DINOv2 ViT-S/14预训练权重
├── dinov2_vitb14_pretrain.pth            # DINOv2 ViT-B/14预训练权重
├── dinov2_vitl14_pretrain.pth            # DINOv2 ViT-L/14预训练权重
│
├── ViT-B-32.pt                           # CLIP ViT-B/32模型
├── ViT-B-16.pt                           # CLIP ViT-B/16模型
├── ViT-L-14.pt                           # CLIP ViT-L/14模型
├── ViT-L-14-336px.pt                     # CLIP ViT-L/14@336px模型
│
├── resnet18-f37072fd.pth                 # ResNet-18预训练权重
├── wide_resnet50_2-95faca4d.pth          # Wide ResNet-50-2预训练权重
│
└── timm/                                 # timm模型缓存
    ├── resnet200/
    ├── efficientnet_b5/
    └── ...
```

---

## 6. 模型加载代码示例

### 6.1 SubspaceAD (DINOv2) 模型加载

```python
from transformers import AutoModel, AutoImageProcessor
import torch

# 配置本地模型路径
local_model_path = "pre_trained/dinov2-with-registers-large"

# 加载处理器和模型
processor = AutoImageProcessor.from_pretrained(
    local_model_path,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    local_model_path,
    local_files_only=True,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).eval()

# 移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### 6.2 MuSc (CLIP) 模型加载

```python
import open_clip
import torch

# 加载CLIP模型
model_name = "ViT-L-14"
model_path = "pre_trained/ViT-L-14.pt"

# 方法1: 从本地文件加载
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained=model_path  # 传入本地路径
)

# 方法2: 使用open_clip的缓存机制
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained='openai'
)

model = model.eval().cuda()
```

### 6.3 Dinomaly (DINOv2) 模型加载

```python
import torch
from algorithms.Dinomaly.models.vit_encoder import vit_encoder

# 加载DINOv2编码器
model = vit_encoder(
    encoder_name="dinov2reg_vit_large_14",
    img_size=600,
    patch_size=14,
    target_layers=[4, 6, 8, 10, 12, 14, 16, 18]
)

# 加载预训练权重
ckpt_path = "pre_trained/dinov2_vitl14_pretrain.pth"
state_dict = torch.load(ckpt_path, map_location='cpu')
if 'model' in state_dict:
    state_dict = state_dict['model']
model.load_state_dict(state_dict, strict=False)
model = model.eval().cuda()
```

### 6.4 ResNet 骨干网络加载

```python
import torch
import torchvision.models as models

# 加载ResNet18
model = models.resnet18(pretrained=False)
checkpoint = torch.load("pre_trained/resnet18-f37072fd.pth", map_location='cpu')
model.load_state_dict(checkpoint)

# 移除分类层，作为特征提取器
model.fc = torch.nn.Identity()
model = model.eval().cuda()
```

### 6.5 timm 模型加载

```python
import timm
import torch

# 加载timm模型
model = timm.create_model(
    'resnet200',
    pretrained=True,
    pretrained_cfg_overlay=dict(file="pre_trained/timm/resnet200/model.safetensors"),
    num_classes=0,
    global_pool=''
)
model = model.eval().cuda()
```

---

## 7. 模型完整性验证

### 7.1 文件大小验证

```python
import os

def verify_model_size(model_path, expected_size_mb, tolerance=0.1):
    """验证模型文件大小"""
    actual_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    size_diff = abs(actual_size_mb - expected_size_mb) / expected_size_mb
    
    if size_diff > tolerance:
        raise ValueError(
            f"Model size mismatch: expected ~{expected_size_mb}MB, "
            f"got {actual_size_mb:.1f}MB"
        )
    return True
```

### 7.2 MD5/SHA256哈希验证

```python
import hashlib

def calculate_file_hash(filepath, algorithm='md5'):
    """计算文件哈希值"""
    hash_obj = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def verify_model_hash(model_path, expected_hash, algorithm='md5'):
    """验证模型文件哈希"""
    actual_hash = calculate_file_hash(model_path, algorithm)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Model hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return True
```

### 7.3 模型结构验证

```python
import torch

def verify_model_structure(model_path, expected_keys=None):
    """验证模型结构"""
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 处理嵌套结构
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    actual_keys = set(state_dict.keys())
    
    if expected_keys and actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        raise ValueError(
            f"Model structure mismatch. Missing: {missing}, Unexpected: {unexpected}"
        )
    
    return True
```

---

## 8. 模型下载脚本使用

### 8.1 下载所有必需模型

```bash
python scripts/download_models.py --all
```

### 8.2 下载所有模型（包括可选）

```bash
python scripts/download_models.py --all --include-optional
```

### 8.3 下载特定模型

```bash
python scripts/download_models.py --model dinov2-large
```

### 8.4 验证已下载模型

```bash
python scripts/download_models.py --verify
```

### 8.5 导出模型配置

```bash
python scripts/download_models.py --export-config models_config.json
```

---

## 9. 总结

本项目共使用 **20+** 个预训练模型，主要分为以下几类：

1. **少样本检测模型**: DINOv2系列 (7个模型, ~5GB)
2. **零样本检测模型**: CLIP系列 (4个模型, ~2.5GB)
3. **Dinomaly模型**: DINOv2预训练权重 (3个模型, ~1.5GB)
4. **骨干网络**: ResNet系列 (2个模型, ~180MB)
5. **timm模型库**: 多个可选模型 (~2GB)

**总存储需求**: 约 **11GB**（必需模型约 **9GB**）

**推荐配置**:
- 最小配置: DINOv2 Base + CLIP ViT-B-16 + ResNet18 (~750MB)
- 标准配置: DINOv2 Large + CLIP ViT-L-14 + Wide ResNet50-2 (~2.5GB)
- 完整配置: 所有必需模型 (~9GB)
