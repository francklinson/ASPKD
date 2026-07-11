# 实现方案：已训练模型信息展示增强 + 算法描述丰富化

## 概述

用户要求：
1. 已训练模型列表信息显示更明确（当前只有 name/family/type/size/time 一行简略信息）
2. 增加筛选按钮（按算法族、算法名等筛选）
3. 算法族中关于已选模型的描述更丰富，从算法项目原文整理补充

## 现状分析

### 后端 (`backend/api/training.py`)
- `TrainedModel` 只有 7 个字段：name, path, size_mb, created_at, algorithm_family, model_type, model_size
- 缺少 `algorithm_name` 字段（虽然推断逻辑存在但未返回）
- 缺少 `category`（训练类别）、`data_source`（数据来源）字段
- `ALGORITHM_FAMILIES` 中各算法的 `description` 只有简短一句话

### 前端 (`frontend/training.html`)
- `loadModels()` 渲染模型列表只显示一行：`family · type · size · time`
- 没有筛选功能
- 算法详情面板 `updateAlgorithmInfo()` 只展示 5 个字段
- 族描述 `familyDesc` 只显示 `ALGORITHM_FAMILIES[key].description`

## 修改计划

### Step 1: 后端 — 扩展 TrainedModel 字段

**文件**: `backend/api/training.py`

1. 在 `TrainedModel` 中增加字段：
   - `algorithm_name: str = ""` — 具体算法名（如 patchcore, invad 等）
   - `category: str = ""` — 训练类别（从文件名推断，如 bottle）
   - `data_source: str = ""` — 数据来源（spk/mvtec）

2. 在 `get_trained_models()` 中补充推断逻辑：
   - 从文件名中解析 category（如 `dinomaly_dinov3_small_bottle_20250711_120000` → `bottle`）
   - 从文件名中解析 data_source

### Step 2: 后端 — 丰富 ALGORITHM_FAMILIES 算法描述

**文件**: `backend/api/training.py`

基于各算法项目 README/docstring 原文，更新每个算法的 `description` 字段，使其更详细。具体内容：

**Dinomaly 族**：
- 族描述：更新为包含论文标题和方法核心描述
- 各算法：补充 Context-Aware Recentering、Linear Attention 等关键技术点

**Dinomaly2 族**：
- 族描述：补充 "One Dinomaly2 Detect Them All" 论文信息，全频谱统一异常检测框架描述
- 各算法：补充 Linear Attention + Loose Constraint 等关键特性

**Anomalib 族**：
- 族描述：补充 "Intel 开源异常检测库" 更多细节
- 各算法 description 从 README 原文提取丰富描述（见下文详细内容）

**ADer 族**：
- 族描述：补充 ADer 工具箱论文信息
- 各算法 description 从模型源码 docstring 和 README 提取

具体算法描述更新内容：

```
PatchCore: "基于核心集的补丁特征嵌入方法（CVPR 2022）。将图像划分为 patch，通过预训练网络提取中间层特征并存入记忆库，推理时通过核心集子采样近似最近邻搜索计算异常分数。MVTec AD 图像级 AUROC 98.0%（WRN-50），无需梯度训练。"
CFA: "耦合超球面特征适应方法（Access 2022）。通过可学习的补丁描述符将正常特征映射到超球面，结合可扩展记忆库实现目标导向的异常定位。"
CS-Flow: "跨尺度全卷积归一化流模型。联合处理多尺度特征，通过跨尺度耦合块提升细粒度表示能力，支持同时进行异常检测和定位。"
DRAEM: "判别性重建异常检测方法（ICCV 2021）。由重建子网络和判别子网络组成，使用 Perlin 噪声生成模拟异常样本训练，结合 L2+SSIM 损失和 Focal Loss。"
DSR: "双空间重构异常检测。通过量化特征学习，使用编码器和双解码器架构，分别在特征空间和图像空间建模正常数据分布。"
Reverse Distillation: "反向蒸馏异常检测（CVPR 2022）。学生解码器从中间层反向学习教师特征提取器的特征表示，通过一类别瓶颈嵌入强制特征映射相似性，实现高精度定位。"
STFPM: "师生特征金字塔匹配（Knowledge Distillation）。教师-学生网络结构，通过多尺度特征匹配和层级知识融合实现不同尺寸异常检测。"
GANomaly: "基于条件 GAN 的异常检测（ACCV 2018）。编码器-解码器-编码器结构，通过比较潜在向量与重构向量的差异评估异常得分。"
SuperSimpleNet: "超简单网络异常检测（ICPR 2024）。特征提取+特征适配+特征级合成异常生成+分割检测模块，推理时跳过异常生成直接预测，支持无监督和监督学习。"
WinCLIP: "窗口级 CLIP 零样本/少样本异常检测（CVPR 2023）。利用预训练 CLIP 提取图像和文本嵌入，通过余弦相似度计算异常分数，多尺度滑动窗口实现像素级定位。"
GLASS: "统一异常合成策略（GLocal Anomaly Synthesis）。三分支训练：正常分支提取适应特征，全局异常合成（GAS）分支通过梯度上升合成近分布异常，局部异常合成（LAS）分支叠加纹理模拟远分布异常。"
INP-Former: "固有正常原型检测（CVPR 2025）。从测试图像中直接提取固有正常原型（INP），通过交叉注意力聚合预训练 ViT 特征，INP 引导解码器约束输出为正常模式。"
GeneralAD: "跨域通用异常检测。利用 ViT patch 结构，自监督构造伪异常样本（噪声注入、打乱、复制），注意力判别器逐 patch 评分。"
PatchFlow: "基于 Patch 特征的归一化流异常检测（2025）。结合局部邻域感知 patch 特征与归一化流，引入适配器模块对齐预训练表示与工业图像分布，瓶颈耦合结构降低计算复杂度。"
L2BT: "Learn to Be a Transformer 异常检测（IEEE Access）。将异常检测转化为 Transformer 学习任务，精确定位异常区域。"
AnomalyDINO: "基于 DINO 的少样本异常检测。利用自监督特征进行少样本设置下的异常检测和定位。"
AnomalyVFM: "零样本视觉基础模型异常检测（CVPR 2026）。将预训练 VFM 转化为零样本异常检测器，先通过 FLUX 生成合成图像训练。"
CFM: "跨模态融合异常检测。结合视觉和文本信息进行跨模态异常检测。"
FRE: "特征重建误差异常检测。基于自编码器重建预训练特征，通过重建误差评估异常。"
UniNet: "统一异常检测网络，融合多种检测范式的统一框架。"
DFM: "深度特征建模异常检测。基于 PCA 的特征降维与重建误差评估。"
DFKDE: "深度特征核密度估计。非参数化异常评分方法。"
EfficientAD: "轻量级异常检测。知识蒸馏 + 自编码器，推理延迟 <1ms/图。"
FastFlow: "快速 2D 归一化流异常检测。"
PaDiM: "参数化异常检测方法。多维度高斯分布建模正常特征。"

MambaAD: "基于状态空间模型的异常检测（arXiv 2024）。引入 Mamba 选择性扫描机制，在视觉 Transformer 和 CNN 中实现线性复杂度的长距离依赖建模，适合多类统一训练。"
InvAD: "逆生成式异常检测（arXiv 2024 / COCO-AD）。基于 StyleGAN2 架构学习正常数据分布的逆向映射，结合像素标准化和可学习风格映射层实现特征反转。"
ViTAD: "基于 Plain ViT 重建的异常检测（CVIU 2025）。使用视觉 Transformer 作为骨干，结合融合模块和可变形注意力机制，实现多类统一异常检测与高精度定位。"
UniAD: "统一异常检测框架（NeurIPS 2022）。单一模型处理多类别，使用多尺度特征金字塔和模块化卷积网络，结合注意力机制融合不同尺度特征。"
CFlow (ADer): "条件归一化流异常检测（WACV 2022）。ADer 框架实现版本，判别式预训练编码器 + 多尺度生成解码器，通过估计编码特征的似然性生成异常图。"
PyramidFlow: "金字塔级归一化流异常检测（CVPR 2023）。多尺度特征金字塔上的归一化流，逐层建模特征分布实现多粒度异常检测。"
SimpleNet: "简单网络异常检测（CVPR 2023）。特征空间判别器方法，轻量高效，特征适配+高斯噪声合成异常+判别评分。"
```

### Step 3: 前端 — 已训练模型列表增强 + 筛选按钮

**文件**: `frontend/training.html`

1. **增加筛选栏**（在模型列表上方）：
   - 算法族筛选按钮组（全部 / Dinomaly / Dinomaly2 / Anomalib / ADer）
   - 搜索框（按模型名称搜索）

2. **增强模型卡片展示**：
   - 模型名称突出显示
   - 算法族标签（彩色 badge）
   - 算法名称
   - 训练类别
   - 数据来源
   - 文件大小
   - 创建时间
   - 模型类型/大小（仅 Dinomaly 族显示）

3. **筛选逻辑**：
   - `modelFilters` 全局状态对象存储筛选条件
   - `filterModels()` 函数根据条件过滤模型列表
   - 筛选按钮点击时更新 `modelFilters` 并重新渲染

### Step 4: 前端 — 算法族描述和已选模型描述丰富化

**文件**: `frontend/training.html`

1. **族描述扩展**：`selectFamily()` 中的 `familyDesc` 显示更详细的族描述（含论文引用、关键特性）
2. **算法详情面板扩展**：`updateAlgorithmInfo()` 增加显示：
   - 论文出处（从 description 中解析或新增字段）
   - 更丰富的算法描述（多行文本）
   - 是否可训练状态

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `backend/api/training.py` | 1) TrainedModel 增加 algorithm_name/category/data_source 字段<br>2) get_trained_models() 补充推断逻辑<br>3) ALGORITHM_FAMILIES 各算法 description 丰富化 |
| `frontend/training.html` | 1) 模型列表增加筛选栏（算法族按钮+搜索框）<br>2) 模型卡片展示增强（badge/算法名/类别/来源等）<br>3) 算法详情面板描述丰富化 |

## 验证步骤

1. 启动后端 `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`
2. 访问训练页面，检查：
   - 已训练模型列表是否显示详细信息（算法名、类别、来源）
   - 筛选按钮是否正常工作
   - 各算法族的描述是否更丰富
   - 算法详情面板是否展示更多信息
