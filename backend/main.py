"""
FastAPI 后端主入口
完全前后端分离架构 - 音频异常检测服务
"""
import os
import sys

# ========== 首先设置 CUDA 环境变量（必须在导入 torch 之前）==========
# 读取配置文件中的环境变量设置
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
                print(f"[Main] Set environment variable: {key}={value}")
    except Exception as e:
        print(f"[Main] Warning: Failed to load environment variables from config: {e}")

# 确保 CUDA 设备设置正确
# 默认暴露所有 GPU，用户可通过前端选择具体使用哪张
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    # 不设置则 PyTorch 能看到所有 GPU，用户在前端选择
    print("[Main] CUDA_VISIBLE_DEVICES not set, exposing all GPUs")

# ========== 设置 Python 路径 ==========
# 首先确保虚拟环境路径正确
venv_site_packages = os.path.join(project_root, ".venv", "lib", "python3.12", "site-packages")
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

# 添加项目根目录到路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ========== 导入路由和核心组件 ==========
# 注意：这些导入必须在 CUDA 环境变量设置之后
print(f"[Main] CUDA_VISIBLE_DEVICES before imports: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")

# 检查 torch 是否已经被导入
if 'torch' in sys.modules:
    print(f"[Main] WARNING: torch already imported before route imports!")
else:
    print(f"[Main] torch not yet imported - good")

from backend.api import detection, monitor, tasks, reference_audio, feature_cluster, zero_shot, few_shot, client_monitor
from backend.core.websocket import websocket_manager
from backend.core.task_manager import task_manager

print(f"[Main] CUDA_VISIBLE_DEVICES after imports: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
if 'torch' in sys.modules:
    print(f"[Main] WARNING: torch was imported during route imports!")
    import torch
    print(f"[Main] torch.cuda.is_available() at import time: {torch.cuda.is_available()}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print("[Backend] 启动音频异常检测服务...")
    
    # 初始化任务管理器
    await task_manager.initialize()
    
    yield
    
    # 关闭时执行
    print("[Backend] 关闭服务，清理资源...")
    await task_manager.cleanup()


app = FastAPI(
    title="音频异常检测 API",
    description="前后端分离架构的音频异常检测服务",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(detection.router, prefix="/api/detection", tags=["检测"])
app.include_router(monitor.router, prefix="/api/monitor", tags=["监控"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["任务管理"])
app.include_router(reference_audio.router, prefix="/api/reference-audio", tags=["参考音频库"])
app.include_router(feature_cluster.router, prefix="/api/cluster", tags=["特征聚类"])
app.include_router(zero_shot.router, prefix="/api/zero-shot", tags=["零样本检测"])
app.include_router(few_shot.router, prefix="/api/few-shot", tags=["少样本检测"])
app.include_router(client_monitor.router, prefix="/api/client", tags=["客户端管理"])

# WebSocket 路由 - 使用标准装饰器方式
@app.websocket("/ws/progress/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket 端点 - 实时推送任务进度"""
    await websocket_manager.handle_connection(websocket, task_id)

# 静态文件服务（前端文件）
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Favicon 路由
@app.get("/favicon.ico")
async def favicon():
    """返回 favicon"""
    favicon_path = os.path.join(frontend_path, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")

# 热力图静态文件服务
visualize_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualize")
if os.path.exists(visualize_path):
    app.mount("/visualize", StaticFiles(directory=visualize_path), name="visualize")


@app.get("/")
async def root():
    """根路径返回登录页面"""
    login_path = os.path.join(frontend_path, "login.html")
    if os.path.exists(login_path):
        return FileResponse(login_path)
    # 如果没有登录页面，返回主页面
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "音频异常检测 API 服务运行中", "docs": "/docs"}


@app.get("/main")
async def main_page():
    """主页面入口"""
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "主页面未找到"}


@app.get("/login")
async def login_page():
    """登录页面入口"""
    login_path = os.path.join(frontend_path, "login.html")
    if os.path.exists(login_path):
        return FileResponse(login_path)
    return {"error": "登录页面未找到"}


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "audio-anomaly-detection-api",
        "version": "2.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info"
    )
