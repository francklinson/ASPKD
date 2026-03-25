# 音频异音检测工具库 (Audio Anomaly Detection for Speaker)

## 项目简介

本项目是一个面向扬声器(SPK)产品质量检测的音频异音检测工具库。核心思想是将音频信号转换为时频图(Spectrogram)，然后利用先进的图像异常检测算法来识别音频中的异常模式。

项目整合了30+种异常检测算法，涵盖基于重建、特征嵌入、流模型、知识蒸馏等多种技术路线，为工业音频质检场景提供一站式解决方案。

---

## 项目架构

```
ASD_for_SPK/
├── data/                          # 数据集目录
├── ref/                           # 参考音频片段(用于数据预处理对齐)
├── slice/                         # 音频切片输出目录
├── inference_dir/                 # 推理结果目录
├── results/                       # 训练结果目录
├── runs/                          # 训练日志目录
├── vis/                           # 可视化结果目录
├── data_prepocessing.py           # 音频预处理模块(核心)
├── draw_roc.py                    # ROC曲线绘制工具
├── check_env.py                   # 环境检测脚本
├── ADer/                          # ADer异常检测框架
├── Anomalib/                      # Anomalib异常检测库
├── BaseASD/                       # 基础自编码器系列
├── DiAD/                          # DiAD扩散模型
├── DictAS/                        # DictAS方法
├── Dinomaly/                      # Dinomaly方法
├── HiAD/                          # HiAD层次化异常检测
├── MuSc/                          # MuSc方法
├── MultiADS/                      # MultiADS多模态异常检测
├── SubspaceAD/                    # SubspaceAD子空间异常检测
└── 原始数据/                       # 原始音频数据
```

---

## 支持的算法

### 一、ADer框架
基于统一训练框架的异常检测算法集合：

| 算法 | 类型 | 说明 |
|------|------|------|
| **MambaAD** | 状态空间模型 | 基于Mamba架构的高效异常检测 |
| **InVad** | 生成模型 | 基于归一化流的异常检测 |
| **UniAD** | 统一框架 | 统一异常检测架构 |
| **ViTAD** | Transformer | 基于Vision Transformer的异常检测 |
| **DiAD** | 扩散模型 | 基于扩散模型的异常检测 |
| **CFlow** | 流模型 | 条件归一化流方法 |
| **PyramidFlow** | 流模型 | 金字塔流异常检测 |
| **SimpleNet** | 特征学习 | 简单高效的异常检测网络 |

### 二、Anomalib库
工业级异常检测算法库，支持20+种算法：

| 算法 | 类型 | 说明 |
|------|------|------|
| **PatchCore** | 特征嵌入 | 基于局部特征匹配的异常检测(推荐) |
| **CFA** | 特征学习 | 基于特征聚类的异常检测 |
| **CSFlow** | 流模型 | 全卷积流模型 |
| **DFKDE** | 密度估计 | 深度特征核密度估计 |
| **DFM** | 特征建模 | 深度特征建模 |
| **DRAEM** | 重建网络 | 基于重建的异常检测 |
| **DSR** | 扩散模型 | 基于扩散恢复的异常检测 |
| **EfficientAd** | 轻量模型 | 轻量级实时异常检测 |
| **FastFlow** | 流模型 | 快速归一化流 |
| **FRE** | 特征学习 | 特征重建误差 |
| **Dinomaly** | 特征学习 | 基于DINO特征的异常检测 |
| **PaDiM** | 特征嵌入 | 基于预训练特征分布建模 |
| **Reverse Distillation** | 知识蒸馏 | 逆向蒸馏师生网络 |
| **STFPM** | 知识蒸馏 | 学生-教师特征金字塔匹配 |
| **GANomaly** | 生成对抗 | 基于GAN的异常检测 |
| **SuperSimpleNet** | 轻量网络 | 超轻量异常检测网络 |
| **UFlow** | 流模型 | 无条件归一化流 |
| **UniNet** | 统一网络 | 统一异常检测网络 |
| **VLM-AD** | 视觉语言 | 基于视觉语言模型的异常检测 |
| **WinCLIP** | 视觉语言 | 基于CLIP的异常检测 |

### 三、BaseASD基础方法
基于自编码器的经典异常检测方法：

| 算法 | 说明 |
|------|------|
| **DenseAE** | 全连接自编码器 |
| **CAE** | 卷积自编码器 |
| **VAE** | 变分自编码器 |
| **AEGAN** | 自编码器+GAN |
| **DifferNet** | 基于流的差异网络 |

### 四、其他先进方法

| 算法 | 类型 | 说明 |
|------|------|------|
| **MuSc** | 表示学习 | 多尺度对比学习异常检测 |
| **HiAD** | 层次检测 | 层次化异常检测方法 |
| **MultiADS** | 多模态 | 多模态异常检测 |
| **SubspaceAD** | 子空间 | 子空间异常检测 |
| **DictAS** | 字典学习 | 基于字典学习的异常检测 |

---

## 数据预处理流程

音频异常检测的核心是将音频信号转换为适合深度学习模型处理的图像形式。

### 1. 音频定位 (`data_prepocessing.py`)

使用**MFCC + DTW**算法在完整音频中定位目标测试片段：

```python
from data_prepocessing import Preprocessor

# 初始化预处理器，指定参考音频
preprocessor = Preprocessor(ref_file="ref/渡口片段10s.wav", split_method='mfcc_dtw')

# 处理音频文件列表，提取目标片段并保存为时频图
preprocessor.process_audio(
    file_list=["path/to/audio1.wav", "path/to/audio2.wav"],
    save_dir="slice"
)
```

**定位算法对比**：
- **mfcc_dtw**: 基于MFCC特征和DTW动态时间规整，精度高(推荐)
- **corr**: 基于互相关，速度快但精度较低

### 2. 时频图生成

将音频片段转换为对数幅度频谱图(Log-frequency Spectrogram)：

```python
def plot_spectrogram(audio_path, output_path, offset=0.0, duration=None):
    # 加载音频
    y, sr = librosa.load(audio_path, offset=offset, duration=duration, sr=22050)
    # 幅值归一化
    y = librosa.util.normalize(y)
    # 计算STFT
    D = librosa.stft(y)
    # 转换为分贝标度
    DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # 保存为图像(600×600)
    plt.imsave(output_path, DB)
```

---

## 快速开始

### 环境配置

```bash
# 1. 检查环境
check_env.py

# 2. 安装依赖
pip install -r requirements.txt
```

**关键依赖**：
- PyTorch >= 2.1.0
- TensorFlow >= 2.9.0
- librosa (音频处理)
- OpenCV (图像处理)
- anomalib (异常检测库)

### 使用示例

#### 1. 数据预处理
```python
from data_prepocessing import Preprocessor

p = Preprocessor(ref_file="ref/渡口片段10s.wav", split_method='mfcc_dtw')
predict_file_list = ["原始数据/audio1.wav", "原始数据/audio2.wav"]
p.process_audio(file_list=predict_file_list, save_dir="slice")
```

#### 2. Anomalib方法训练与测试
```python
from Anomalib.data import MVTecAD
from Anomalib.models import Patchcore
from Anomalib.engine import Engine

# 定义数据集
datamodule = MVTecAD(root="data/spk", category="dk", train_batch_size=16)

# 定义模型
model = Patchcore()

# 定义训练引擎
engine = Engine(max_epochs=1000)

# 训练
engine.train(datamodule=datamodule, model=model)

# 预测
predictions = engine.predict(datamodule=datamodule, model=model, ckpt_path="path/to/ckpt")
```

#### 3. ADer方法使用
```python
from ADer import MambaAD, ViTAD, InVad, DiAD, UniAD, CFlow

# 创建模型实例
model = UniAD()
model.test()
```

#### 4. BaseASD方法使用
```python
from BaseASD.DenseAE.DenseAE_interface import DenseAEInterface
from BaseASD.ConvolutionalAE.CAE_interface import CAEInterface
from BaseASD.VAE.VAE_interface import VAEInterface
from BaseASD.AEGAN.AeGan_interface import AEGANInterface

# 创建接口实例
dae = DenseAEInterface()
cae = CAEInterface()
vae = VAEInterface()
aegan = AEGANInterface()

# 判断音频是否正常
result = dae.judge_is_normal("path/to/audio.wav")
```

---

## 数据集格式

项目使用类似MVTec AD的数据集结构：

```
data/spk/
├── category1/                    # 产品类别(如: dk, qzgy, N32等)
│   ├── train/
│   │   └── good/                 # 正常样本训练集
│   ├── test/
│   │   ├── good/                 # 正常样本测试集
│   │   └── bad/                  # 异常样本测试集
│   └── ground_truth/
│       └── bad/                  # 异常标注(可选)
├── category2/
│   └── ...
```

---

## 评估指标

项目支持以下异常检测评估指标：

- **AUROC**: 接收者操作特征曲线下面积(主要指标)
- **AP**: 平均精度
- **F1-Score**: F1分数
- **TPR/FPR**: 真正率/假正率
- **ACC**: 准确率
- **Precision/Recall**: 精确率/召回率

---

## 算法选择建议

| 场景 | 推荐算法 | 说明 |
|------|----------|------|
| 高精度需求 | PatchCore, UniAD, MambaAD | 性能最佳，适合质检场景 |
| 实时检测 | EfficientAd, SuperSimpleNet | 轻量级，推理速度快 |
| 少样本学习 | MuSc, DictAS | 支持Few-shot场景 |
| 无监督场景 | CFA, PaDiM, DFM | 无需异常样本训练 |
| 重建类异常 | DRAEM, FastFlow | 适合纹理/频谱异常 |

---

## 项目特色

1. **统一的音频预处理流程**：MFCC+DTW精准定位 + 时频图转换
2. **丰富的算法支持**：30+种SOTA异常检测算法
3. **模块化的设计**：各算法独立封装，易于扩展和替换
4. **工业级应用**：针对扬声器质检场景优化
5. **完善的评估体系**：支持多种指标和可视化

---

## 开发团队

- 项目用途：扬声器(SPK)产品质量检测
- 应用领域：工业质检、音频异常检测

---

## 参考论文

| 算法 | 论文 | 年份 |
|------|------|------|
| PatchCore | Towards Total Recall in Industrial Anomaly Detection | CVPR 2022 |
| CFlow | Real-Time Anomaly Detection and Localization | CVPR 2022 |
| DRAEM | Discriminative Reconstruction Anomaly Detection | CVPR 2021 |
| PaDiM | PaDiM: a Patch Distribution Modeling Framework | ICPR 2021 |
| STFPM | Student-Teacher Feature Pyramid Matching | CVPR 2021 |
| Reverse Distillation | Anomaly Detection via Reverse Distillation | CVPR 2022 |
| SimpleNet | SimpleNet: A Simple Network for Image Anomaly Detection | CVPR 2023 |
| EfficientAd | Accurate and Lightweight Anomaly Detection | CVPR 2023 |
| MambaAD | State Space Model for Anomaly Detection | 2024 |
| UniAD | Unified Anomaly Detection | 2024 |

---

## 许可证

本项目仅供研究和学习使用。
