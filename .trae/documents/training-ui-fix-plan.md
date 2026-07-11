# 模型训练前端UI适配计划

## Context

需求设计.md 中有两个待办任务需要完成：
- **待办[1]**: 模型训练前端页面适配，支持所有注册模型，增加详细信息提示（性能、检测类型等）
- **待办[5]**: 前端 training.html 需添加算法族选择器

当前 training.html 仅支持3个算法族（Dinomaly/Anomalib/ADer），算法列表硬编码，无算法详情提示。后端已支持3族训练调度，但缺少 Dinomaly2。需要扩展到4个可训练算法族并添加算法详情。

## 可训练算法族

| 族 | 可训练 | 算法数 | 参数模式 |
|---|---|---|---|
| Dinomaly | 是 | 6 | encoder + size |
| Dinomaly2 | 是 | 6 | encoder + size |
| Anomalib | 是 | 22 | 算法选择 |
| ADer | 是 | 7 | 算法选择 |
| BaseASD | 否（需TensorFlow） | - | - |
| MuSc | 否（零样本） | - | - |
| SubspaceAD | 否（少样本） | - | - |

## 修改文件

1. **`backend/api/training.py`** — 扩展 ALGORITHM_FAMILIES（4族+详情），新增 Dinomaly2 训练调度器
2. **`frontend/training.html`** — 重写算法族选择器为动态渲染，新增算法详情面板

## 实施步骤

### 第一阶段：后端改造

#### Step 1: 扩展 ALGORITHM_FAMILIES 数据结构

在 `training.py` 中：

- 将 `ALGORITHM_FAMILIES` 从3族扩展为4族，增加 `dinomaly2`
- 为每个算法增加详情字段：`description`、`performance`、`gpu_memory`、`input_size`
- 增加 `param_schema` 字段标识参数UI模式（`encoder_size` 或 `algorithm_select`）
- 同步 Anomalib 算法列表（补齐 v2.5.0 新增算法：anomalyvfm/cfm/general_ad/glass/inp_former/l2bt/patchflow/anomaly_dino）

#### Step 2: 新增 Dinomaly2 训练调度器

- 在 `_dispatch_training()` 增加 `dinomaly2` 分支
- 新增 `_run_dinomaly2_training()` 函数，调用 `algorithms/Dinomaly2/dinomaly_2D.py`
- Dinomaly2 接受 `--data_path`、`--save_dir`、`--save_name`、`--backbone`、`--total_iters` 等参数，与 Dinomaly v1 类似
- backbone 映射：`dinov2_small` → `dinov2reg_vit_small_14`，`dinov2_base` → `dinov2reg_vit_base_14` 等

#### Step 3: 同步 Anomalib 算法列表

- `training.py` 的 Anomalib 算法列表对齐 `custom_detection.py` 中的 ALGORITHM_GROUPS
- 补齐 anomalyvfm, cfm, general_ad, glass, inp_former, l2bt, patchflow, anomaly_dino 等 v2.5.0 新增算法

### 第二阶段：前端改造

#### Step 4: 更新页面标题

- 将 header 中的 "启动 Dinomaly 训练任务" 改为 "启动模型训练任务"

#### Step 5: 重写算法族选择区域

- 移除3个硬编码按钮，改为 `<div id="algoFamilyGroup">` 动态容器
- 新增 `loadFamilies()` 从 `/api/training/families` 加载数据
- 新增 `renderFamilyButtons()` 只渲染 `trainable: true` 的族按钮

#### Step 6: 重写参数区域为动态渲染

- 移除3个硬编码参数区域（`dinomaly-algo-options`、`anomalib-algo-options`、`ader-algo-options`）
- 替换为统一的 `<div id="dynamicParamsContainer">`
- 新增 `renderParamsForFamily(familyKey)` 根据 `param_schema` 渲染不同UI：
  - `encoder_size`：编码器选择按钮 + 模型大小下拉 + 通用参数
  - `algorithm_select`：算法下拉列表 + 通用参数
- 通用参数（迭代次数、批次大小）在族切换时保留值

#### Step 7: 新增算法详情面板

- 在参数区域下方新增可折叠的算法详情面板
- 显示：检测类型、性能特征、显存需求、输入尺寸、说明
- 选择算法或切换编码器/大小时自动更新
- 新增 `updateAlgorithmInfo()` 函数
- 新增 `toggleAlgorithmInfo()` 展开/折叠控制

#### Step 8: 重写 startTraining()

- 适配动态参数结构，根据 `param_schema` 提取不同的参数组合

#### Step 9: 更新 DOMContentLoaded

- 增加 `loadFamilies()` 调用

### 第三阶段：CSS 样式

#### Step 10: 新增算法详情面板样式

- `.algorithm-info-panel` 容器样式
- `.algorithm-info-header` 可点击头部
- `.info-row` / `.info-label` 详情行样式

## 验证方式

1. 启动服务后访问 `/training` 页面
2. 验证4个算法族按钮正确显示
3. 验证切换算法族时参数区域动态更新
4. 验证算法详情面板显示正确信息
5. 验证 Dinomaly2 算法族可以选择并启动训练
6. 验证通用参数在族切换时保留
