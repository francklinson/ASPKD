# PreciseSegmentLocator - 精确片段定位器

基于 Shazam 音频指纹算法优化的长音频多片段检测与定位系统。

## 核心思路

### 1. 问题背景

传统的 `LongAudioAnalyzer` 使用**滑动窗口**方式检测长音频：
- 窗口大小 10s，步长 5s
- 每个窗口独立查询数据库
- 多次查询导致性能瓶颈

**缺点**：
- 同一音频片段被多个窗口重复处理
- 数据库查询次数多（N 个窗口 = N 次查询）
- 无法精确定位片段起始点（只能返回窗口位置）

### 2. 优化策略

`PreciseSegmentLocator` 采用**全局指纹提取 + 批量查询**策略：

```
┌─────────────────────────────────────────────────────────────┐
│  长音频 (60s)                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. 全局指纹提取 (一次)                                │   │
│  │     - 提取完整频谱图                                   │   │
│  │     - 检测全局峰值点                                   │   │
│  │     - 生成完整 hash 列表 (~3000个)                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  2. 批量数据库查询 (一次)                              │   │
│  │     - 批量查询所有 hash                               │   │
│  │     - 获取匹配结果                                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  3. 时间对齐聚类                                       │   │
│  │     - offset_diff = db_offset - query_offset         │   │
│  │     - 相同 offset_diff 的匹配点聚类                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  4. 多片段定位                                         │   │
│  │     - 找出所有超过阈值的匹配点                         │   │
│  │     - 支持同一音频包含多个参考音频                     │   │
│  │     - 精确定位每个片段起始时间                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3. 核心算法

#### 3.1 指纹提取（与 Shazam 兼容）

```python
# STFT 参数
sr = 16000          # 采样率
n_fft = 4096        # FFT 窗口大小
hop_length = 1024   # 帧移

# 峰值检测
- 使用 maximum_filter 找局部最大值
- 生成二进制结构元素 (cross shape)
- 过滤低能量峰值 (amp_min=5)

# Hash 生成
hash = SHA1(f"{f1}|{f2}|{t_delta}")
```

#### 3.2 时间对齐投票

```python
# 核心公式
offset_diff = db_offset - query_offset

# 相同的 (music_id, offset_diff) 表示同一匹配位置
vote_map[(music_id, offset_diff)] += 1
```

**负偏移处理**（重要）：
- `offset_diff = -129` 表示参考音频从查询音频的 8.26s 处开始
- 实际起始时间 = `-offset_diff * hop_length / sr = 8.26s`

#### 3.3 双阈值过滤

```python
# 条件1: 绝对匹配数（防误报）
confidence >= 10

# 条件2: 匹配比例（适应不同长度参考音频）
match_ratio = confidence / total_hashes >= 0.01
```

### 4. 性能对比

| 指标 | LongAudioAnalyzer (滑动窗口) | PreciseSegmentLocator (全局) |
|------|------------------------------|------------------------------|
| 扫描方式 | 滑动窗口 (10s, 步长5s) | 全局提取 |
| 数据库查询 | 5-10 次 | 1 次批量 |
| 处理 60s 音频 | ~2.5s | ~0.8s |
| 定位精度 | 窗口级别 (±5s) | Hash 级别 (±0.064s) |
| 多片段检测 | 单一片段 | 支持多片段 |

### 5. 使用方式

```python
from core.precise_segment_locator.locator import PreciseSegmentLocator, SegmentLocatorConfig

# 创建定位器
config = SegmentLocatorConfig(
    threshold=10,           # 最小匹配 hash 数
    min_match_ratio=0.01,   # 最小匹配比例
    segment_duration=10.0   # 切分片段时长
)
locator = PreciseSegmentLocator(config)

# 添加参考音频
locator.add_reference(music_id=33, music_name="渡口")
locator.add_reference(music_id=34, music_name="青藏高原")

# 分析长音频
result = locator.locate_segments("audio.wav")

for seg in result.segments:
    print(f"{seg.music_name}: {seg.start_time:.2f}s - {seg.end_time:.2f}s")
```

### 6. 文件结构

```
core/precise_segment_locator/
├── __init__.py          # 模块初始化
├── locator.py           # 核心定位器实现
├── adapter.py           # LongAudioAnalyzer 兼容适配器
└── README.md            # 本文件
```

### 7. 关键注意事项

#### 7.1 负偏移处理

```python
# 当 locate 返回负偏移时：
if location.start_time < 0:
    # 表示参考音频在查询音频的 |offset| 秒处开始
    start_time = -location.start_time  # 取绝对值
```

**这是 Shazam 算法的特性**，必须在代码注释和文档中明确标注。

#### 7.2 坐标顺序

指纹提取时坐标顺序必须与 Shazam 一致：
```python
j, i = np.where(local_max)  # j=频率索引, i=时间索引
peaks = list(zip(i, j))     # 返回 (时间, 频率)
```

### 8. 适配器模式

`PreciseSegmentLocatorAdapter` 提供与 `LongAudioAnalyzer` 兼容的接口：

```python
from core.precise_segment_locator.adapter import PreciseSegmentLocatorAdapter
from core.long_audio_analyzer import AnalyzerConfig

# 配置兼容
config = AnalyzerConfig(
    window_size=10.0,
    match_threshold=10
)

# 直接替换 LongAudioAnalyzer
analyzer = PreciseSegmentLocatorAdapter(config, db_connector)
result = analyzer.analyze("audio.wav")
```

## 依赖

- numpy
- librosa
- scipy
- hashlib
- MySQL Connector (core.shazam.database.connector)
