#!/usr/bin/env python3
"""
服务器启动脚本
使用虚拟环境运行
环境配置已迁移到 start_server.sh
"""
import os
import sys

# 项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

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
