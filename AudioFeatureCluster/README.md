# 音频异常检测与特征提取对比实验

本项目实现了一个完整的音频异常检测系统，支持多种特征提取方法，并提供对比实验框架。

## 项目结构

```
聚类可视化/
├── audio_anomaly_detection.py  # 主程序：特征提取与聚类可视化
├── 01_download_audio.py    # 步骤1：下载开源人声素材
├── 02_augment_audio.py     # 步骤2：构造音频数据库
├── 03_compare_experiments.py # 步骤3：对比实验
├── run_all.py              # 一键运行完整流程
├── test_single.py          # 快速测试单个特征提取器
├── raw_audio/              # 原始音频文件
├── audio_database/         # 增强后的音频数据库
└── experiment_results/     # 实验结果
```

## 支持的音频特征提取方法

1. **MFCC** - 梅尔频率倒谱系数（传统方法，计算快速）
2. **Mel Spectrogram** - Mel频谱图（传统方法，频谱信息丰富）
3. **HuBERT** - Facebook语音表示学习模型
4. **WavLM** - Microsoft鲁棒语音预训练模型
5. **XLSR-Wav2Vec2** - 多语言Wav2Vec2模型
6. **MERT** - 音乐表示学习模型
7. **AST** - Audio Spectrogram Transformer

## 快速开始

### 方式1：一键运行完整流程

```bash
python run_all.py
```

### 方式2：分步执行

```bash
# 步骤1：下载开源人声素材（LibriSpeech）
python 01_download_audio.py

# 步骤2：构造音频数据库（添加多种处理方式）
python 02_augment_audio.py

# 步骤3：运行对比实验
python 03_compare_experiments.py
```

### 方式3：快速测试单个特征提取器

```bash
# 测试 MFCC（最快）
python test_single.py mfcc

# 测试其他提取器
python test_single.py hubert
python test_single.py mert
python test_single.py wavlm
```

## 音频数据库构造（适合异常检测场景）

### 设计原则

为模拟真实异常检测场景，数据集设计为：
- **正常样本 (~90%)**：大量样本聚集在聚类中心附近
- **异常样本 (~10%)**：少量样本明显偏离聚类中心

### 数据分布

每个说话人包含 **110个样本**：

| 类型 | 数量 | 占比 | 生成方式 |
|------|------|------|----------|
| **正常样本** | 100 | ~90% | 时域偏移(70) + 原始副本(5) + 音量变化(10) + 轻微噪声(10) + 轻微混响(5) |
| **异常样本** | 10 | ~10% | 变速 + 变调 + 重噪声 + 削波失真 + 电话音质 |

### 正常样本详情

1. **原始音频副本** (5个): 完全相同的原始音频
2. **时域偏移** (70个): 循环偏移0.1-2秒，保持内容相似性
3. **音量变化** (10个): ±1dB到±6dB的轻微调整
4. **轻微噪声** (10个): 高SNR(21-30dB)的白噪声
5. **轻微混响** (5个): 10%-20%的混响强度

### 异常样本详情

- 变速1.3x / 0.7x
- 变调±5半音 / +7半音
- 重噪声8dB / 5dB
- 削波失真0.4-0.5
- 电话音质限制

### 预期聚类效果

- **3个清晰的聚类中心**（对应3个说话人）
- **大量正常样本**形成密集的聚类区域
- **少量异常样本**明显偏离聚类中心

### 样本命名规则

- `*_normal.wav` - 正常样本
- `*_anomaly.wav` - 异常样本

### 可视化说明

聚类结果可视化保存在 `experiment_results/` 目录：

- **红色** = Speaker 01 的样本
- **蓝色** = Speaker 02 的样本  
- **绿色** = Speaker 03 的样本
- **黑色圆圈** = 算法检测到的异常点

预期效果：
- 三个颜色的密集聚类区域（每个说话人的100个正常样本）
- 部分样本被黑色圆圈标记为异常（通常是异常样本）

## 实验输出

运行对比实验后，将生成以下输出：

1. **单独可视化结果**
   - `anomaly_detection_result_mfcc.png`
   - `anomaly_detection_result_hubert.png`
   - ...

2. **对比报告**（在 `experiment_results/` 目录）
   - `comparison_table.txt` - 对比表格
   - `comparison_summary.png` - 可视化对比图
   - `comparison_results_*.json` - 详细结果JSON
   - `analysis_report.md` - 综合分析报告

## 使用方法示例

### 使用特定特征提取器

```python
from audio_anomaly_detection import Config, FeatureExtractorFactory

# 选择特征提取器
config = Config(feature_extractor_type="wavlm")
extractor = FeatureExtractorFactory.create_extractor(config)
```

### 自定义异常检测参数

```python
from audio_anomaly_detection import Config

config = Config(feature_extractor_type="mfcc")
config.n_clusters = 3                    # 聚类数量
config.outlier_threshold_percentile = 90  # 异常阈值百分位
config.segment_duration = 5               # 音频片段时长（秒）
```

## 依赖安装

```bash
pip install torch transformers librosa numpy matplotlib scikit-learn soundfile scipy
```

## 注意事项

1. **首次运行**需要下载预训练模型，需要网络连接
2. **GPU加速**：如果可用，会自动使用CUDA加速
3. **内存要求**：深度学习模型需要较多内存，建议至少8GB
4. **处理时间**：MFCC/Mel最快（秒级），深度学习模型较慢（分钟级）
