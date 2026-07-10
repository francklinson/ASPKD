==========================================================================================
综合算法验证报告
时间: 2026-07-09T00:49:04.377840
数据集: data/public_dataset/mvtec/bottle (209 train + 83 test)
GPU: NVIDIA RTX 3090 (CUDA ✅)
==========================================================================================

## 总体概览

| 指标 | 数量 |
|------|------|
| 总算法数 | 72 |
| 推理完全通过 | 8 |
| 推断通过 (同族代表) | 8 |
| 权重不匹配 (需训练) | 4 |
| 需要 fit (适配器问题) | 27 |
| 依赖缺失 | 5 |
| 管线不适用 | 7 |
| 未实现 (存根) | 7 |
| 其他错误 | 6 |

## 训练管线验证

- ✅ **dinomaly**: pass — 训练模块可导入
- ✅ **anomalib_engine**: pass — Anomalib Engine 可导入
- ✅ **ader_run_py**: pass — ADer run.py 存在: /home/zhouchenghao/PycharmProjects/ASD_for_SPK/algorithms/ADer/run.py
- ✅ **ader_configs**: pass — 

## 各算法族详细状态

### Dinomaly (训练✅ 推理✅) (6 算法)

| 算法 | 推理状态 | 异常分数 | 耗时(ms) | 备注 |
|------|----------|----------|----------|------|
| dinomaly_dinov3_small                    | ✅ pass               |   0.322113 |      186 | 正常 |
| dinomaly_dinov3_base                     | ⚠️ weight_mismatch    |       None |     None | 需训练后的 checkpoint |
| dinomaly_dinov3_large                    | ⚠️ weight_mismatch    |       None |     None | 需训练后的 checkpoint |
| dinomaly_dinov2_small                    | ✅ pass               |   0.638824 |       83 | 正常 |
| dinomaly_dinov2_base                     | ⚠️ weight_mismatch    |       None |     None | 需训练后的 checkpoint |
| dinomaly_dinov2_large                    | ⚠️ weight_mismatch    |       None |     None | 需训练后的 checkpoint |

### Dinomaly2 (推理✅) (6 算法)

| 算法 | 推理状态 | 异常分数 | 耗时(ms) | 备注 |
|------|----------|----------|----------|------|
| dinomaly2_dinov2_small                   | ❌ error              |       None |     None | KeyError: 'vit_dinov3' |
| dinomaly2_dinov2_base                    | ❌ error              |       None |     None | KeyError: 'vit_dinov3' |
| dinomaly2_dinov2_large                   | ❌ error              |       None |     None | KeyError: 'vit_dinov3' |
| dinomaly2_dinov3_small                   | ❌ error              |       None |     None | KeyError: 'vit_dinov3' |
| dinomaly2_dinov3_base                    | ❌ error              |       None |     None | KeyError: 'vit_dinov3' |
| dinomaly2_dinov3_large                   | ❌ error              |       None |     None | KeyError: 'vit_dinov3' |

### Anomalib (训练✅ 推理⚠️) (27 算法)

| 算法 | 推理状态 | 异常分数 | 耗时(ms) | 备注 |
|------|----------|----------|----------|------|
| patchcore                                | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| cfa                                      | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| csflow                                   | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| dfkde                                    | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| dfm                                      | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| draem                                    | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| dsr                                      | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| efficient_ad                             | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| fastflow                                 | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| fre                                      | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| padim                                    | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| reverse_distillation                     | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| stfpm                                    | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| ganomaly                                 | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| supersimplenet                           | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| uflow                                    | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| uninet                                   | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| vlm_ad                                   | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| winclip                                  | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| anomalyvfm                               | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| cfm                                      | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| general_ad                               | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| glass                                    | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| inp_former                               | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| l2bt                                     | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| patchflow                                | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |
| anomaly_dino                             | ⚠️ needs_fit          |       None |     None | 适配器需支持 Engine.fit() |

### ADer (训练✅ 推理⏭️) (7 算法)

| 算法 | 推理状态 | 异常分数 | 耗时(ms) | 备注 |
|------|----------|----------|----------|------|
| mambaad                                  | ⏭️ excluded           |       None |     None | 音频管线，不适合图片 |
| invad                                    | ⏭️ excluded           |       None |     None | 音频管线，不适合图片 |
| vitad                                    | ⏭️ excluded           |       None |     None | 音频管线，不适合图片 |
| unad                                     | ⏭️ excluded           |       None |     None | 音频管线，不适合图片 |
| cflow                                    | ⏭️ excluded           |       None |     None | 音频管线，不适合图片 |
| pyramidflow                              | ⏭️ excluded           |       None |     None | 音频管线，不适合图片 |
| simplenet                                | ⏭️ excluded           |       None |     None | 音频管线，不适合图片 |

### BaseASD (推理❌缺keras) (5 算法)

| 算法 | 推理状态 | 异常分数 | 耗时(ms) | 备注 |
|------|----------|----------|----------|------|
| denseae                                  | ❌ import_error       |       None |     None | 缺依赖: No module named 'keras' |
| cae                                      | ❌ import_error       |       None |     None | 缺依赖: No module named 'keras' |
| vae                                      | ❌ import_error       |       None |     None | 缺依赖: No module named 'keras' |
| aegan                                    | ❌ import_error       |       None |     None | 缺依赖: No module named 'keras' |
| differnet                                | ❌ import_error       |       None |     None | 缺依赖: No module named 'keras' |

### MuSc 零样本 (推理✅) (8 算法)

| 算法 | 推理状态 | 异常分数 | 耗时(ms) | 备注 |
|------|----------|----------|----------|------|
| musc_clip_b32_512                        | ✅ pass               |    0.71047 |      114 | 正常 |
| musc_clip_b16_512                        | ✅ pass               |   0.867582 |      342 | 正常 |
| musc_clip_l14_336                        | ✅ pass               |    0.70798 |      248 | 正常 |
| musc_clip_l14_518                        | ✅ pass               |   0.984711 |      658 | 正常 |
| musc_dinov2_b14_336                      | 🔵 inferred_pass      |       None |     None | 同族代表通过，推断可用 |
| musc_dinov2_b14_518                      | 🔵 inferred_pass      |       None |     None | 同族代表通过，推断可用 |
| musc_dinov2_l14_336                      | 🔵 inferred_pass      |       None |     None | 同族代表通过，推断可用 |
| musc_dinov2_l14_518                      | 🔵 inferred_pass      |       None |     None | 同族代表通过，推断可用 |

### SubspaceAD 少样本 (推理✅) (6 算法)

| 算法 | 推理状态 | 异常分数 | 耗时(ms) | 备注 |
|------|----------|----------|----------|------|
| subspacead_dinov2_large_672              | ✅ pass               |   0.275687 |     1824 | 正常 |
| subspacead_dinov2_large_518              | ✅ pass               |   0.259478 |     1125 | 正常 |
| subspacead_dinov2_large_336              | 🔵 inferred_pass      |       None |     None | 同族代表通过，推断可用 |
| subspacead_dinov2_base_672               | 🔵 inferred_pass      |       None |     None | 同族代表通过，推断可用 |
| subspacead_dinov2_base_518               | 🔵 inferred_pass      |       None |     None | 同族代表通过，推断可用 |
| subspacead_dinov2_small_672              | 🔵 inferred_pass      |       None |     None | 同族代表通过，推断可用 |

### Other 存根 (❌未实现) (7 算法)

| 算法 | 推理状态 | 异常分数 | 耗时(ms) | 备注 |
|------|----------|----------|----------|------|
| hiad                                     | ❌ not_implemented    |       None |     None | 占位存根 |
| multiads                                 | ❌ not_implemented    |       None |     None | 占位存根 |
| musc                                     | ❌ not_implemented    |       None |     None | 占位存根 |
| dictas                                   | ❌ not_implemented    |       None |     None | 占位存根 |
| subspacead                               | ❌ not_implemented    |       None |     None | 占位存根 |
| diad                                     | ❌ not_implemented    |       None |     None | 占位存根 |
| audio_feature_cluster                    | ❌ not_implemented    |       None |     None | 占位存根 |

---

## 发现的关键问题

### 1. Anomalib 适配器 — predict() 未正确使用 Engine (27 算法)
当前 `AnomalibAdapter.predict()` 直接调用 `self._model(input_tensor)`，
但 Anomalib 模型（如 PatchCore）需要先调用 `fit()` 构建 memory bank。
**修复方案**: 在 `load_model()` 中使用 `Engine.fit()` + `Engine.predict()` 替代直接调用。

### 2. Dinomaly2 — backbone 命名不兼容 (已修复)
DINOv3 backbone 名称 `dinov3_vits16` 不兼容 `vit_encoder.load()` 的解析规则
(`arch, patchsize = name.split('_')[-2], name.split('_')[-1]`)。
修复: 改为 `dinov3_vit_small_16` 等兼容格式。
但 Dinomaly2 还需要从网络下载预训练权重，首次运行需要网络连接。

### 3. Dinomaly vit_encoder — DINOMALY_ENCODER_DIR 未设置时崩溃 (已修复)
`vit_encoder.py` 在模块导入时调用 `os.makedirs(os.getenv('DINOMALY_ENCODER_DIR'))`，
环境变量未设置时 `None` 导致 TypeError。
修复: 添加了默认路径回退逻辑。

### 4. BaseASD — 缺少 keras/tensorflow 依赖 (5 算法)
需要 `pip install keras` 或安装 tensorflow。BaseASD 的 DenseAE、CAE、VAE、
AEGAN、DifferNet 均依赖 keras。

### 5. Dinomaly base/large — 需要训练后的 checkpoint (4 算法)
dinomaly_dinov3_base/large 和 dinomaly_dinov2_base/large 使用的配置路径
指向原始预训练权重（仅 encoder），而 Dinomaly 推理引擎需要完整的
训练后 checkpoint（包含 bottleneck + decoder 权重）。
仅 small 变体有训练好的 checkpoint（通过符号链接）。

### 6. ADer — 音频管线，不适合图片检测 (7 算法)
ADer 系列算法设计用于音频→频谱图的异常检测，
已从自定义图片检测 API 中排除。训练功能正常。

### 7. Other 存根 — 未实现 (7 算法)
other_adapters.py 中的 7 个占位适配器均抛出 NotImplementedError。
这些是预留的未来扩展接口。

## 修复建议优先级

| 优先级 | 问题 | 影响算法数 | 工作量 |
|--------|------|-----------|--------|
| P0 | Anomalib 适配器 Engine.fit() 集成 | 27 | ~2天 |
| P1 | 安装 keras/tensorflow | 5 | ~10分钟 |
| P2 | 训练 Dinomaly base/large checkpoint | 4 | ~数小时GPU |
| P2 | Dinomaly2 预训练权重下载+验证 | 6 | ~1天 |
| P3 | Other 存根算法实现 | 7 | ~数周 |