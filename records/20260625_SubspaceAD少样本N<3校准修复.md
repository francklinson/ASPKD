# SubspaceAD 少样本检测 N<3 校准修复

## 日期
2026-06-25

## 问题描述

少样本检测仅上传 1~2 张参考图时（N<3），检测结果异常分数全为 1.0000，无法区分正常与异常。

## 根因分析

`_calibrate_normal_scores()` 方法中 N<3 的分支存在两个问题：

### 问题 1：dummy PCA params 维度错误（第一次报错）
第 248-252 行用维度错误的虚拟 PCA 参数调用 `_compute_image_score`：
- `mu = np.zeros(1)` 形状 `(1,)`，但实际特征维度是 1024
- `components = np.zeros((1, 1))` 形状 `(1, 1)`，`k=0`
- 导致 `pca_reconstruct` 中 `(N, 1024) @ (1, 0)` → **matmul 维度不匹配错误**

### 问题 2：启发式估算导致分数饱和（第二次修复前）
修复问题 1 后（删除 dummy 调用），剩余启发式公式存在本质缺陷：
- `_normal_score_mean = 0.01 × ref_norm`（任意因子）
- `_normal_score_std = 0.5 × 均值`
- `_score_upper_bound = 均值 + 4×标准差 = 0.03 × ref_norm`
- PCA 重建误差的量级远大于 `0.03 × ref_norm`，导致所有测试图的原始分数都远超 upper_bound → 全部映射到 1.0

## 修复方案

用参考图**自身的实际 PCA 重建误差**建立校准基线，替代任意估算：

### N=2：2 折交叉验证
- 图 A 用图 B 的 PCA 评分，图 B 用图 A 的 PCA 评分
- 得到真正的"跨图像未见样本"误差估计
- `widen = 2.0`，`upper = mean + 4×std`（与 N≥3 一致）

### N=1：自重建校准
- 在唯一参考图上拟合 PCA → 对该图自身的 patch tokens 评分
- 得到"图内补丁变异"的重建误差量级
- `widen = 3.0`，`upper = mean + 6×std`（更宽松的边界）

### 公共改进
- 参考图像 std 过低时，使用 `max(mean × 2.0, 0.1)` 作为保守默认值
- 防止 upper_bound ≤ lower_bound 的边界反转

## 与原项目对比

| 层面 | 原版 SubspaceAD `main.py` | 旧适配器 N<3 | 新适配器 N<3 |
|------|--------------------------|-------------|-------------|
| 分数校准 | 无（靠验证集） | dummy 占位 | PCA 实际重建误差 |
| LOO 逻辑 | 不存在 | N≥3 有 | N≥3 + N=2 都有 |
| N=1 处理 | 不存在 | 任意估算 | 自重建 + 宽边界 |

## 修改文件

| 文件 | 改动 |
|------|------|
| `algorithms/subspacead_adapter.py` | `_calibrate_normal_scores()` N<3 分支重写 |

## 提交记录
- `7463fbe` fix: SubspaceAD 少样本检测 N<3 校准修复
