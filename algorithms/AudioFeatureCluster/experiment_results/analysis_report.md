# 音频特征提取器对比分析报告

**生成时间**: 2026-03-25 23:04:15

## 1. 实验概述

- **数据目录**: `/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化/audio_database`
- **成功实验数**: 7/7
- **测试的特征提取器**: hubert, mfcc, mel, mert, wavlm, xlsr-wav2vec2, ast

## 2. 详细结果

### HUBERT

- 样本数: 328 (正常: 300, 异常: 28)
- 特征维度: 768
- 检测到异常: 33
- 正确检测异常: 20/28 (71.4%)
- 误报: 13, 漏报: 8
- 聚类分布: {0: 109, 1: 109, 2: 110}
- 处理时间: 5.59秒
- 输出图像: `/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化/experiment_results/anomaly_detection_result_hubert.png`

### MFCC

- 样本数: 328 (正常: 300, 异常: 28)
- 特征维度: 40
- 检测到异常: 33
- 正确检测异常: 17/28 (60.7%)
- 误报: 16, 漏报: 11
- 聚类分布: {0: 119, 1: 104, 2: 105}
- 处理时间: 1.44秒
- 输出图像: `/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化/experiment_results/anomaly_detection_result_mfcc.png`

### MEL

- 样本数: 328 (正常: 300, 异常: 28)
- 特征维度: 128
- 检测到异常: 33
- 正确检测异常: 16/28 (57.1%)
- 误报: 17, 漏报: 12
- 聚类分布: {0: 205, 1: 15, 2: 108}
- 处理时间: 1.23秒
- 输出图像: `/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化/experiment_results/anomaly_detection_result_mel.png`

### MERT

- 样本数: 325 (正常: 297, 异常: 28)
- 特征维度: 1024
- 检测到异常: 33
- 正确检测异常: 18/28 (64.3%)
- 误报: 15, 漏报: 10
- 聚类分布: {0: 108, 1: 109, 2: 108}
- 处理时间: 8.57秒
- 输出图像: `/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化/experiment_results/anomaly_detection_result_mert.png`

### WAVLM

- 样本数: 328 (正常: 300, 异常: 28)
- 特征维度: 768
- 检测到异常: 33
- 正确检测异常: 19/28 (67.9%)
- 误报: 14, 漏报: 9
- 聚类分布: {0: 109, 1: 110, 2: 109}
- 处理时间: 6.74秒
- 输出图像: `/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化/experiment_results/anomaly_detection_result_wavlm.png`

### XLSR-WAV2VEC2

- 样本数: 328 (正常: 300, 异常: 28)
- 特征维度: 1024
- 检测到异常: 33
- 正确检测异常: 17/28 (60.7%)
- 误报: 16, 漏报: 11
- 聚类分布: {0: 89, 1: 206, 2: 33}
- 处理时间: 8.48秒
- 输出图像: `/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化/experiment_results/anomaly_detection_result_xlsr-wav2vec2.png`

### AST

- 样本数: 328 (正常: 300, 异常: 28)
- 特征维度: 768
- 检测到异常: 33
- 正确检测异常: 20/28 (71.4%)
- 误报: 13, 漏报: 8
- 聚类分布: {0: 108, 1: 108, 2: 112}
- 处理时间: 11.82秒
- 输出图像: `/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化/experiment_results/anomaly_detection_result_ast.png`

## 3. 对比分析

### 3.1 处理速度

- **最快**: MEL (1.23秒)
- **最慢**: AST (11.82秒)

### 3.2 异常检测准确性

- **最佳检测准确率**: HUBERT (71.4%)
- **最低检测准确率**: MEL (57.1%)

检测准确率排名:
  1. HUBERT: 71.4% (正确 20/28)
  2. AST: 71.4% (正确 20/28)
  3. WAVLM: 67.9% (正确 19/28)
  4. MERT: 64.3% (正确 18/28)
  5. MFCC: 60.7% (正确 17/28)

### 3.3 特征维度

- **最高维度**: MERT (1024)
- **最低维度**: MFCC (40)

## 4. 结论与建议

### 4.1 方法特点总结

| 方法 | 类型 | 特点 | 适用场景 |
|------|------|------|----------|
| MFCC | 传统特征 | 计算快，维度低 | 实时应用，资源受限 |
| Mel Spectrogram | 传统特征 | 频谱信息丰富 | 音频分类，音乐分析 |
| HuBERT | 深度学习 | 语音表示强大 | 语音识别，说话人识别 |
| WavLM | 深度学习 | 鲁棒性高 | 噪声环境，远场语音 |
| XLSR-Wav2Vec2 | 深度学习 | 多语言支持 | 多语言场景 |
| MERT | 深度学习 | 音乐理解好 | 音乐分析，旋律识别 |
| AST | 深度学习 | 频谱+Transformer | 通用音频理解 |

### 4.2 实验建议

1. **根据应用场景选择**: 如果是语音任务，优先选择HuBERT/WavLM；如果是音乐任务，选择MERT
2. **考虑计算资源**: MFCC和Mel Spectrogram计算最快，适合实时应用
3. **数据质量**: 深度学习模型需要更多数据才能发挥优势
4. **异常定义**: 不同特征对'异常'的定义不同，需要根据具体任务调整阈值

---
*报告由自动对比实验系统生成*
