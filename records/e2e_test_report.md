# 端到端测试报告

时间: 2026-07-15 00:08:05

```
======================================================================
端到端测试结果汇总 (2026-07-15 00:08:05)
======================================================================
总计: 23 项, 通过: 17, 失败: 6

── dinomaly ──
  [FAIL] Dinomaly DINOv3 Small | 推理(预训练)
       原因: failed: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
  [PASS] Dinomaly DINOv3 Small | 训练

── dinomaly2 ──
  [PASS] Dinomaly2 DINOv2 Small | 推理(预训练)
  [FAIL] Dinomaly2 DINOv2 Small | 训练
       原因: failed: 训练失败 (退出码: 1)

── anomalib ──
  [PASS] PatchCore | 推理(预训练)
  [PASS] PaDiM | 推理(预训练)
  [PASS] EfficientAD | 推理(预训练)
  [PASS] DFKDE | 推理(预训练)
  [PASS] FastFlow | 推理(预训练)
  [PASS] STFPM | 推理(预训练)
  [FAIL] U-Flow | 推理(预训练)
       原因: failed: Cannot send a request, as the client has been closed.
  [PASS] VLM-AD | 推理(预训练)
  [FAIL] AnomalyVFM | 推理(预训练)
       原因: failed: Cannot send a request, as the client has been closed.
  [PASS] PatchCore | 训练

── ader ──
  [PASS] MambaAD | 推理(预训练)
  [PASS] MambaAD | 训练

── trained ──
  [PASS] MAMBAADTrainer_configs_benchmark_mambaad_mambaad_256_100e_20260715-000707 | 推理(训练后)
  [PASS] anomalib_patchcore_bottle_20260715_000628 | 推理(训练后)
  [FAIL] dinomaly_dinov3_small_bottle_epoch_200_Wed Jul 15 00:06:18 2026.pth | 推理(训练后)
       原因: 不支持的算法: dinov3_small
  [FAIL] dinomaly_dinov3_small_bottle_epoch_200_Wed Jul 15 00:02:08 2026.pth | 推理(训练后)
       原因: 不支持的算法: dinov3_small
  [PASS] MAMBAADTrainer_configs_benchmark_mambaad_mambaad_256_100e_20260714-235206 | 推理(训练后)
  [PASS] MAMBAADTrainer_configs_benchmark_mambaad_mambaad_256_100e_20260714-013213 | 推理(训练后)
  [PASS] PyramidFlowTrainer_configs_benchmark_pyramidflow_pyramidflow_256_100e_20260714-013135 | 推理(训练后)

── 失败项汇总 ──
  dinomaly/dinomaly_dinov3_small [推理(预训练)]: failed: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
  anomalib/uflow [推理(预训练)]: failed: Cannot send a request, as the client has been closed.
  anomalib/anomalyvfm [推理(预训练)]: failed: Cannot send a request, as the client has been closed.
  dinomaly2/dinomaly2_dinov2_small [训练]: failed: 训练失败 (退出码: 1)
  trained/dinov3_small [推理(训练后)]: 不支持的算法: dinov3_small
  trained/dinov3_small [推理(训练后)]: 不支持的算法: dinov3_small

```
