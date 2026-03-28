# 统一算法调用接口

本项目提供了一个统一的算法调用接口，可以方便地切换和使用不同的异常声音检测(ASD)算法。

## 目录结构

```
algorithms/
├── __init__.py                    # 统一接口入口
├── factory.py                     # 工厂函数实现
├── unified_usage_example.py       # 使用示例
├── dinomaly_adapter.py           # Dinomaly算法适配器
├── ader_adapter.py               # ADer算法适配器
├── anomalib_adapter.py           # Anomalib算法适配器
├── baseasd_adapter.py            # BaseASD算法适配器
├── other_adapters.py             # 其他算法适配器
├── Dinomaly/                     # Dinomaly算法实现
├── ADer/                         # ADer算法库
├── Anomalib/                     # Anomalib算法库
├── BaseASD/                      # BaseASD算法库
└── ...                           # 其他算法目录
```

## 快速开始

### 1. 创建检测器

使用工厂函数创建检测器实例：

```python
from algorithms import create_detector

# 创建Dinomaly检测器（自动加载配置）
detector = create_detector("dinomaly_dinov3_small")

# 创建其他算法
detector = create_detector("mambaad")          # ADer的MambaAD
detector = create_detector("patchcore")        # Anomalib的PatchCore
detector = create_detector("denseae")          # BaseASD的DenseAE
```

### 2. 加载模型

```python
# 加载模型权重
detector.load_model()
print(f"模型加载成功，运行设备: {detector.device}")
```

### 3. 执行推理

**单张图像推理：**

```python
result = detector.predict("path/to/image.png")

print(f"是否异常: {result.is_anomaly}")
print(f"异常分数: {result.anomaly_score:.4f}")
print(f"推理时间: {result.inference_time:.2f} ms")

# 如果有热力图
if result.metadata and 'heatmap_path' in result.metadata:
    print(f"热力图路径: {result.metadata['heatmap_path']}")
```

**批量推理：**

```python
image_paths = [
    "path/to/image1.png",
    "path/to/image2.png",
    "path/to/image3.png"
]

results = detector.predict_batch(image_paths)

for path, result in zip(image_paths, results):
    print(f"{os.path.basename(path)}: 分数={result.anomaly_score:.4f}, 异常={result.is_anomaly}")
```

### 4. 释放资源

```python
detector.release()
```

## 可用算法列表

### Dinomaly系列（基于DINO特征）
- `dinomaly_dinov3_small` - DINOv3 Small模型
- `dinomaly_dinov3_base` - DINOv3 Base模型
- `dinomaly_dinov3_large` - DINOv3 Large模型
- `dinomaly_dinov2_small` - DINOv2 Small模型
- `dinomaly_dinov2_base` - DINOv2 Base模型
- `dinomaly_dinov2_large` - DINOv2 Large模型

### ADer系列
- `mambaad` - 基于状态空间模型的MambaAD
- `invad` - 基于归一化流的InVad
- `vitad` - 基于Vision Transformer的ViTAD
- `unad` - 统一异常检测框架UniAD
- `cflow` - 基于归一化流的CFlow
- `pyramidflow` - 金字塔流PyramidFlow
- `simplenet` - 简洁高效的SimpleNet

### Anomalib系列
- `patchcore` - 基于特征嵌入的PatchCore
- `efficient_ad` - 轻量级EfficientAD
- `padim` - 基于特征嵌入的PaDiM

### BaseASD系列
- `denseae` - 密集自编码器DenseAE
- `cae` - 卷积自编码器CAE
- `vae` - 变分自编码器VAE

## 配置文件

算法配置存储在 `config/algorithms.yaml`：

```yaml
algorithms:
  dinomaly_dinov3_small:
    name: "Dinomaly DINOv3 Small"
    type: "feature_based"
    thresholds:
      default: 0.033

models:
  dinomaly:
    dinov3_small: "path/to/dinomaly_dinov3_small.pth"
    dinov2_small: "path/to/dinomaly_dinov2_small.pth"
  mambaad:
    default: "path/to/mambaad_best.pth"
  patchcore:
    default: "path/to/patchcore_best.pth"
```

## 高级用法

### 动态调整阈值

```python
# 获取当前阈值
current_threshold = detector.threshold
print(f"当前阈值: {current_threshold}")

# 设置新阈值
detector.set_threshold(0.1)
result = detector.predict("image.png")
```

### 指定运行设备

```python
# 自动选择（优先GPU）
detector = create_detector("dinomaly_dinov3_small", device="auto")

# 强制使用CPU
detector = create_detector("dinomaly_dinov3_small", device="cpu")

# 使用指定GPU
detector = create_detector("dinomaly_dinov3_small", device="cuda:0")
```

### 获取算法信息

```python
from algorithms import list_available_algorithms, get_algorithm_info

# 获取所有可用算法
algorithms = list_available_algorithms()
print(f"可用算法: {algorithms}")

# 获取算法详细信息
info = get_algorithm_info("dinomaly_dinov3_small")
print(f"算法名称: {info['name']}")
print(f"算法类型: {info['type']}")
```

### 自定义模型路径

```python
# 从配置自动读取
detector = create_detector("dinomaly_dinov3_small")

# 手动指定模型路径
detector = create_detector(
    "dinomaly_dinov3_small",
    model_path="/path/to/custom/model.pth"
)
```

## 运行示例

运行完整示例：

```bash
cd /home/zhouchenghao/PycharmProjects/ASD_for_SPK
python algorithms/unified_usage_example.py
```

示例提供了6个演示：
1. 基础使用 - 单张图像检测
2. 批量推理
3. 获取可用算法列表
4. 多算法对比
5. 动态调整阈值
6. 指定运行设备

## 添加新算法

要添加新的算法，需要：

1. **创建适配器**（如 `newalgo_adapter.py`）：

```python
from core import BaseDetector, DetectionResult, register_algorithm

@register_algorithm("new_algo_name")
class NewAlgoAdapter(BaseDetector):
    def load_model(self):
        # 加载模型逻辑
        pass
    
    def predict(self, image_path):
        # 推理逻辑
        return DetectionResult(
            is_anomaly=False,
            anomaly_score=0.0,
            anomaly_map=None,
            inference_time=0.0
        )
```

2. **在 `__init__.py` 中导入**：

```python
try:
    from . import newalgo_adapter
except Exception as e:
    print(f"[algorithms] newalgo_adapter 导入失败: {e}")
```

3. **在 `config/algorithms.yaml` 中添加配置**：

```yaml
algorithms:
  new_algo_name:
    name: "New Algorithm"
    type: "custom"
    thresholds:
      default: 0.5

models:
  new_algo_name:
    default: "path/to/model.pth"
```

## 接口规范

所有算法适配器必须实现 `BaseDetector` 基类：

```python
class BaseDetector(ABC):
    def __init__(self, model_path: str, device: str = 'auto', threshold: float = 0.5)
    
    @abstractmethod
    def load_model(self) -> None                    # 加载模型
    
    @abstractmethod
    def predict(self, image_path: str) -> DetectionResult  # 单张推理
    
    def predict_batch(self, image_paths: list) -> list    # 批量推理（可选）
    
    def set_threshold(self, threshold: float) -> None     # 设置阈值
    
    def release(self) -> None                      # 释放资源
```

## 常见问题

### Q: 出现 `TypeError: expected str, bytes or os.PathLike object, not NoneType`

A: 这是因为环境变量未设置。确保在 `config/asd_gui_config.yaml` 中正确配置了环境变量：

```yaml
environments:
  DINOMALY_ENCODER_DIR: "/path/to/encoder/dir"
```

### Q: 如何添加自己的模型？

A: 在 `config/algorithms.yaml` 的 `models` 部分添加模型路径：

```yaml
models:
  your_algorithm:
    default: "path/to/your/model.pth"
```

### Q: 如何修改默认阈值？

A: 在 `config/algorithms.yaml` 中修改对应算法的阈值：

```yaml
algorithms:
  dinomaly_dinov3_small:
    thresholds:
      default: 0.05  # 修改为你需要的值
```

### Q: 支持哪些图像格式？

A: 支持所有PIL库支持的格式，包括PNG、JPG、BMP等。

## 依赖要求

```txt
torch>=1.9.0
torchvision>=0.10.0
PIL
numpy
pyyaml
```

不同算法可能有额外依赖，请参考各算法目录下的 `requirements.txt`。

## 许可证

本项目遵循各自算法库的许可证。具体请参考各算法目录下的LICENSE文件。
