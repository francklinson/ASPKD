#!/usr/bin/env python3
"""
服务器启动脚本
使用虚拟环境运行
"""
import os
import sys

# 确保使用正确的虚拟环境路径
project_root = os.path.dirname(os.path.abspath(__file__))
venv_site_packages = os.path.join(project_root, ".venv", "lib", "python3.12", "site-packages")

# 设置环境变量
if os.path.exists(venv_site_packages):
    os.environ['PYTHONPATH'] = venv_site_packages + ':' + os.environ.get('PYTHONPATH', '')
    sys.path.insert(0, venv_site_packages)
    sys.path.insert(0, project_root)

print(f"Python: {sys.executable}")
print(f"Project root: {project_root}")
print(f"Site-packages: {venv_site_packages}")
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

print("\n" + "=" * 60)
print("🎵 音频异常检测后端服务")
print("=" * 60)
print("服务地址: http://0.0.0.0:8000")
print("API 文档: http://0.0.0.0:8000/docs")
print("=" * 60 + "\n")

uvicorn.run(
    "backend.main:app",
    host="0.0.0.0",
    port=8000,
    reload=False,
    log_level="info"
)
