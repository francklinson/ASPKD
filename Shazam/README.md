# Shazam 音频指纹识别模块

基于音频指纹技术的音乐/音频识别系统，使用 STFT + 局部峰值检测 + SHA1 哈希算法。

## 功能特性

- 🎵 **音频指纹生成**: 从音频文件提取唯一指纹
- 🔍 **快速识别**: 在数据库中匹配查询音频
- 📍 **精确定位**: 在长音频中定位片段位置
- 📦 **批量处理**: 支持批量添加和识别
- 🔌 **ASD集成**: 可作为 MFCC+DTW 的替代方案

## 快速开始

### 1. 初始化数据库

```bash
cd Shazam
python quickstart.py init
```

### 2. 添加参考音频

```bash
# 添加单个文件
python quickstart.py add ref/audio.wav --name "参考音频"

# 批量添加目录
python quickstart.py add ref/ --pattern "*.wav"
```

### 3. 识别音频

```bash
python quickstart.py recognize query.wav
```

### 4. 定位片段

```bash
python quickstart.py locate long_audio.wav --ref ref.wav
```

## Python API

### 基础用法

```python
from Shazam import AudioFingerprinter

# 创建指纹识别器
fingerprinter = AudioFingerprinter()

# 添加参考音频
fingerprinter.add_reference("ref.wav", name="参考音频")

# 识别查询音频
result = fingerprinter.recognize("query.wav")
print(f"匹配: {result.name}, 偏移: {result.offset}s, 置信度: {result.confidence}")

# 关闭连接
fingerprinter.close()
```

### 上下文管理器

```python
from Shazam import AudioFingerprinter

with AudioFingerprinter() as fp:
    fp.add_reference("ref.wav")
    result = fp.recognize("query.wav")
    # 自动关闭连接
```

### 音频定位

```python
from Shazam import AudioFingerprinter

with AudioFingerprinter() as fp:
    # 添加参考音频
    fp.add_reference("ref.wav", name="片段")

    # 在长音频中定位
    location = fp.locate("long_audio.wav", reference_name="片段")

    if location.found:
        print(f"位置: {location.start_time}~{location.end_time}s")
        print(f"置信度: {location.confidence}")
```

### 批量操作

```python
from Shazam import create_fingerprint_db, batch_recognize, batch_locate

# 批量创建指纹库
ids = create_fingerprint_db("dataset/key/", pattern="*.wav")

# 批量识别
results = batch_recognize(["q1.wav", "q2.wav", "q3.wav"])
for r in results:
    print(r.name, r.matched)

# 批量定位
positions = batch_locate(["long1.wav", "long2.wav"], reference_path="ref.wav")
```

## ASD 项目集成

### 替换 MFCC+DTW 定位

```python
from Shazam.asd_adapter import ShazamLocator, ShazamPreprocessor

# 方法1: 仅使用定位功能
locator = ShazamLocator(ref_audio="ref/渡口片段10s.wav")
segment = locator.locate("原始数据/test.wav")
print(f"片段位置: {segment.start_time}~{segment.end_time}s")

# 方法2: 完整预处理流程（定位+时频图生成）
preprocessor = ShazamPreprocessor(ref_file="ref/渡口片段10s.wav")
output_files = preprocessor.process_audio(
    file_list=["audio1.wav", "audio2.wav"],
    save_dir="slice/"
)
```

### 与原有代码对比

| 特性 | MFCC+DTW | Shazam |
|------|----------|--------|
| 定位精度 | 高 | 更高 |
| 抗噪能力 | 中等 | 强 |
| 速度 | 较慢 | 快（数据库索引）|
| 存储需求 | 低 | 中等（需存指纹）|
| 批量处理 | 支持 | 支持 |

## 配置说明

配置文件: `config/config.yaml`

```yaml
fingerprint:
  core:
    amp_min: 5              # 能量阈值
    near_num: 20            # 近邻点数量
    neighborhood: 15        # 局部最大值检测范围
    stft:
      sr: 16000             # 采样率
      n_fft: 4096           # FFT窗口
      hop_length: 1024      # 步长
  database:
    host: 'localhost'
    port: 3306
    user: 'music'
    password: 'xxx'
    database: 'music_recognition'
```

## 命令行工具

```bash
# 查看帮助
python quickstart.py --help

# 列出所有命令
python quickstart.py init     # 初始化数据库
python quickstart.py add      # 添加参考音频
python quickstart.py recognize # 识别音频
python quickstart.py locate   # 定位片段
python quickstart.py list     # 列出参考音频
python quickstart.py delete   # 删除参考音频
python quickstart.py batch    # 批量识别
```

## 数据库结构

### music 表
| 字段 | 说明 |
|------|------|
| music_id | 歌曲ID |
| music_name | 歌曲名称 |
| music_path | 文件路径 |

### finger_prints 表
| 字段 | 说明 |
|------|------|
| id_fp | 指纹ID |
| music_id_fk | 外键（关联music表）|
| hash | SHA1哈希值 |
| offset | 时间偏移 |

## 注意事项

1. **MySQL 依赖**: 确保 MySQL 服务已启动
2. **采样率**: 默认使用 16kHz，与 ASD 项目的 22.05kHz 不同
3. **首次使用**: 需要先执行 `init` 创建数据库表
4. **参考音频**: 添加到指纹库后才能用于定位

## 项目结构

```
Shazam/
├── __init__.py          # 包入口，导出主要类
├── api.py               # 核心 API 实现
├── asd_adapter.py       # ASD 项目适配器
├── quickstart.py        # 命令行工具
├── example_usage.py     # 使用示例
├── core/                # 核心算法
│   ├── IMusicProcessor.py
│   ├── STFTMusicProcessor.py
│   └── ...
├── database/            # 数据库操作
│   └── MySQLConnector.py
└── config/              # 配置文件
    └── config.yaml
```
