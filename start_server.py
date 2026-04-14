#!/usr/bin/env python3
"""
服务器启动脚本
使用虚拟环境运行
环境配置已迁移到 start_server.sh
"""
import os
import sys

# ========== 首先设置 CUDA 环境变量（必须在导入 torch 之前）==========
# 项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(project_root, "config", "config.yaml")

if os.path.exists(config_path):
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        env_config = config.get('environments', {})
        for key, value in env_config.items():
            if value and key not in os.environ:
                os.environ[key] = str(value)
                print(f"[StartServer] Set environment variable: {key}={value}")
    except Exception as e:
        print(f"[StartServer] Warning: Failed to load environment variables from config: {e}")

# 确保 CUDA 设备设置正确
# 默认暴露所有 GPU，用户可通过前端选择具体使用哪张
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    # 不设置则 PyTorch 能看到所有 GPU，用户在前端选择
    print("[StartServer] CUDA_VISIBLE_DEVICES not set, exposing all GPUs")

# 确保项目根目录在路径中
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Python: {sys.executable}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

# 测试导入
try:
    import librosa
    print(f"✓ librosa imported: {librosa.__version__}")
except ImportError as e:
    print(f"✗ librosa import failed: {e}")
    sys.exit(1)

try:
    import watchdog
    print(f"✓ watchdog imported")
except ImportError as e:
    print(f"✗ watchdog import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 启动 FastAPI
import uvicorn

# 从环境变量获取配置
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8004))

print("\n" + "=" * 60)
print("🎵 音频异常检测后端服务")
print("=" * 60)
print(f"服务地址: http://{HOST}:{PORT}")
print(f"API 文档: http://{HOST}:{PORT}/docs")
print("=" * 60 + "\n")

uvicorn.run(
    "backend.main:app",
    host=HOST,
    port=PORT,
    reload=False,
    log_level="info"
)
