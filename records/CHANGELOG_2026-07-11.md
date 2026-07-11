# 工作日志 — 2026-07-11

## 提交记录

### 75b94d3 — 训练验证脚本 + Anomalib 内存银行修复

**新增**: `tests/verify_training.py` (949行) — 全面训练验证脚本
**修改**: `algorithms/anomalib_adapter.py` — 内存银行逻辑修复

---

## 验证目标

验证全部注册算法在 `data/public_dataset/mvtec/capsule` 上的训练+推理可行性。

## 实现方案

### 测试脚本架构

```
tests/verify_training.py
  ├── 6 family handlers:
  │   ├── dinomaly    → 子进程训练 (dinomaly_train_evaluate)
  │   ├── anomalib    → Engine API 原地训练 (fit + test)
  │   ├── ader        → 子进程训练 (ADer/run.py)
  │   ├── musc        → 仅推理 (零样本)
  │   ├── subspacead  → 仅推理 (少样本 PCA)
  │   └── dinomaly2   → 仅推理
  ├── 环境变量自动设置:
  │   ├── HF_HUB_OFFLINE=1
  │   ├── HUGGINGFACE_HUB_CACHE=models/pre_trained/huggingface
  │   └── PYTHONPATH (子进程)
  └── 报告输出: records/verify_training_report.json
```

### CLI 参数

```bash
python tests/verify_training.py                  # 全部算法
python tests/verify_training.py --quick          # 快速模式
python tests/verify_training.py --families dinomaly  # 指定family
```

---

## 发现并修复的问题

### 1. Anomalib `_fit_memory_bank()` 错误 (`anomalib_adapter.py`)

| 问题 | 修复 |
|------|------|
| 错误: `from Anomalib.types import LearningType` | 正: `from Anomalib import LearningType` |
| 错误: `model.fit(dataloader)` — padim/patchcore 的 fit() 不接受参数 | 简化为 defer 到训练阶段 |
| 训练后 memory bank 未自动填充 | `fit()` 方法中增加 `model.fit()` 后处理步骤 |

### 2. HF Cache 路径 (`tests/verify_training.py`)

| 错误值 | 正确值 |
|--------|--------|
| `HUGGINGFACE_HUB_CACHE=models/pre_trained` | `models/pre_trained/huggingface` |
| `TRANSFORMERS_CACHE=models/pre_trained` | `models/pre_trained/huggingface` |

### 3. ADer 子进程配置路径

- 原有路径拼接错误: `PROJECT_ROOT/ADer/configs/...` 缺少 `algorithms/` 前缀
- 修复为使用 `os.path.join(ader_dir, "configs", ...)` + 绝对路径

### 4. 数据符号链接

```bash
ln -sf public_dataset/mvtec data/mvtec  # ADer configs 默认读 data/mvtec
```

---

## 验证结果 (快速模式)

| Family | 测试算法 | 训练 | 推理 | 状态 |
|--------|---------|------|------|------|
| Dinomaly | dinov3_small | ✅ 10 iters, 6s | ✅ good=0.318, anom=0.307 | **PASS** |
| Anomalib | patchcore | ❌ Memory bank empty | ❌ | FAIL |
| Anomalib | padim | ❌ Tensor dim mismatch | ❌ | FAIL |
| ADer | invad | ❌ 配置模块导入 | — | FAIL |
| MuSc | clip_b32_512 | N/A (零样本) | ✅ good=0.353, anom=0.282 | **PASS** |
| SubspaceAD | dinov2_small_672 | N/A (少样本) | ✅ good=0.456, anom=0.488 | **PASS** |
| Dinomaly2 | dinov2_small | N/A (无训练) | ✅ good=1.087, anom=1.086 | **PASS** |
| BaseASD ×5 | — | — | — | SKIP (Keras) |
| Stubs ×6 | — | — | — | SKIP |

**总结**: 4/7 family 可用, Anomalib 推理正常但训练有版本兼容性问题, ADer 训练待修

---

## Anomalib 训练失败根因分析

对 `draem`/`fastflow`/`ganomaly`/`stfpm` 的深层测试显示**所有 Anomalib 训练均失败**:

| 模型 | 错误 | 根因 |
|------|------|------|
| patchcore | Memory bank empty | 训练后未 subsample embedding |
| padim | Tensor dim mismatch (4096 vs 0) | anomaly_map 维度计算错误 |
| draem | too many values to unpack | lightning 回调 API 变更 |
| fastflow | too many values to unpack | 同上 |
| ganomaly | Expected 3D or 4D input | batch 维度处理问题 |
| stfpm | too many values to unpack | 同上 |
| reverse_distillation | Loss.forward() args mismatch | lightning v2.6+ API 变更 |

**共同根因**: 本地 Anomalib v2.5.0 代码与当前 PyTorch/Lightning 版本不兼容。需升级 Anomalib 源码或降级依赖。

---

## 已确认可用的推理 (前次验证)

| Family | 推理通过 |
|--------|---------|
| Dinomaly (6) | dinov3_small/base/large, dinov2_small/base/large |
| Anomalib (27) | 全部 (含新增 anomalyvfm/cfm/glass/inp_former) |
| MuSc (8) | 全部 CLIP + DINOv2 变体 |
| SubspaceAD (6) | 全部 |
| Dinomaly2 (6) | 全部 (随机初始化, 缺权重) |
| ADer (10) | fallback 模式可用 |
