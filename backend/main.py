"""
FastAPI 后端主入口
完全前后端分离架构 - 音频异常检测服务
"""
import os
import sys

# 首先确保虚拟环境路径正确
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

from backend.api import detection, monitor, tasks, reference_audio, feature_cluster
from backend.core.websocket import websocket_manager
from backend.core.task_manager import task_manager


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
