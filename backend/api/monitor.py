"""
目录监控 API
"""
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.core.monitor_service import monitor_service

router = APIRouter()


class MonitorConfig(BaseModel):
    """监控配置"""
    directory: str = Field(..., description="监控目录路径")
    interval: int = Field(default=30, ge=5, le=300, description="检测间隔（秒）")
    algorithm: str = Field(default="dinomaly_dinov3_small", description="检测算法")
    device: str = Field(default="auto", description="运行设备")
    detect_existing: bool = Field(default=False, description="是否检测已有文件")
    file_extensions: list = Field(
        default=[".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"],
        description="监控的文件扩展名"
    )


class MonitorStatus(BaseModel):
    """监控状态"""
    is_running: bool
    directory: Optional[str]
    interval: int
    algorithm: Optional[str]
    device: Optional[str]
    total_processed: int
    anomaly_count: int
    start_time: Optional[str]


@router.post("/start")
async def start_monitoring(config: MonitorConfig):
    """启动目录监控"""
    try:
        success = await monitor_service.start(
            directory=config.directory,
            interval=config.interval,
            algorithm=config.algorithm,
            device=config.device,
            detect_existing=config.detect_existing,
            file_extensions=config.file_extensions
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="监控启动失败，可能已经在运行")
        
        return {
            "status": "started",
            "message": f"开始监控目录: {config.directory}",
            "config": config.dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@router.post("/stop")
async def stop_monitoring():
    """停止目录监控"""
    success = await monitor_service.stop()
    if not success:
        raise HTTPException(status_code=400, detail="监控未在运行")
    
    return {"status": "stopped", "message": "监控已停止"}


@router.get("/status", response_model=MonitorStatus)
async def get_monitor_status():
    """获取监控状态"""
    status = monitor_service.get_status()
    return MonitorStatus(**status)


@router.get("/results")
async def get_monitor_results(limit: int = 100, offset: int = 0):
    """
    获取监控检测结果
    
    - **limit**: 返回结果数量
    - **offset**: 偏移量
    """
    results = monitor_service.get_results(limit=limit, offset=offset)
    return {
        "results": results,
        "total": monitor_service.total_processed,
        "anomaly_count": monitor_service.anomaly_count
    }


@router.post("/clear")
async def clear_results():
    """清空监控结果"""
    monitor_service.clear_results()
    return {"status": "cleared", "message": "结果已清空"}


@router.post("/cleanup")
async def cleanup_temp_files(max_age_hours: int = 24):
    """清理临时文件"""
    deleted_count = await monitor_service.cleanup_temp_files(max_age_hours)
    return {
        "status": "cleaned",
        "deleted_files": deleted_count,
        "message": f"清理了 {deleted_count} 个临时文件"
    }
