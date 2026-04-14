"""
目录监控 API
"""
import os
import zipfile
import shutil
from datetime import datetime
from typing import Optional
from urllib.parse import quote
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.core.local_monitor_service import monitor_service

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
    # 参考音频选择
    reference_audios: list = Field(
        default=[],
        description="选择的参考音频路径列表，为空则使用自动匹配"
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
        from backend.core.task_manager import task_manager
        
        # 检查是否有运行中的离线检测任务
        running_count = task_manager.get_running_count()
        queued_count = task_manager.get_queued_count()
        
        if running_count > 0 or queued_count > 0:
            # 获取当前正在运行的任务信息
            current_task = None
            for task in task_manager.tasks.values():
                if task.status in ["running", "pending"]:
                    current_task = task
                    break
            
            if current_task:
                # 检查算法是否匹配
                if current_task.algorithm != config.algorithm:
                    raise HTTPException(
                        status_code=409, 
                        detail=f"存在运行中的离线检测任务使用算法 '{current_task.algorithm}'，监控必须使用相同算法。请等待任务完成或取消任务后再启动监控。"
                    )
                # 检查设备是否匹配
                if current_task.device != config.device:
                    raise HTTPException(
                        status_code=409,
                        detail=f"存在运行中的离线检测任务使用设备 '{current_task.device}'，监控必须使用相同设备。请等待任务完成或取消任务后再启动监控。"
                    )
        
        success = await monitor_service.start(
            directory=config.directory,
            interval=config.interval,
            algorithm=config.algorithm,
            device=config.device,
            detect_existing=config.detect_existing,
            file_extensions=config.file_extensions,
            reference_audios=config.reference_audios
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="监控启动失败，可能已经在运行")
        
        return {
            "status": "started",
            "message": f"开始监控目录: {config.directory}",
            "config": config.dict()
        }
    
    except HTTPException:
        raise
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


@router.get("/detection-context")
async def get_detection_context():
    """
    获取当前检测上下文信息
    用于监控启动前检查是否可以使用指定的算法和设备
    """
    from backend.core.task_manager import task_manager
    
    running_count = task_manager.get_running_count()
    queued_count = task_manager.get_queued_count()
    
    context = {
        "has_running_task": running_count > 0 or queued_count > 0,
        "running_count": running_count,
        "queued_count": queued_count,
        "current_algorithm": None,
        "current_device": None,
        "current_task_id": None
    }
    
    # 获取当前运行中的任务信息
    for task in task_manager.tasks.values():
        if task.status in ["running", "pending"]:
            context["current_algorithm"] = task.algorithm
            context["current_device"] = task.device
            context["current_task_id"] = task.id
            break
    
    return context


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


class UpdateReferenceAudiosRequest(BaseModel):
    """更新参考音频请求"""
    reference_audios: list = Field(
        default=[],
        description="新的参考音频路径列表"
    )


@router.post("/update-references")
async def update_reference_audios(request: UpdateReferenceAudiosRequest):
    """
    动态更新参考音频（运行时生效）
    
    - **reference_audios**: 新的参考音频路径列表
    """
    result = await monitor_service.update_reference_audios(request.reference_audios)
    return result


class ScanRequest(BaseModel):
    """扫描目录请求"""
    directory: str = Field(..., description="目录路径")
    file_extensions: list = Field(
        default=[".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"],
        description="文件扩展名列表"
    )


@router.post("/scan")
async def scan_directory(request: ScanRequest):
    """
    扫描目录中的音频文件
    
    - **directory**: 目录路径
    - **file_extensions**: 文件扩展名列表
    """
    import os
    
    directory = request.directory
    file_extensions = request.file_extensions
    
    if not os.path.exists(directory):
        raise HTTPException(status_code=400, detail="目录不存在")
    
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail="路径不是目录")
    
    extensions = set(ext.lower() for ext in (file_extensions or [".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"]))
    
    audio_files = []
    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[1].lower()
                if ext in extensions:
                    audio_files.append({
                        "name": filename,
                        "path": filepath,
                        "size": os.path.getsize(filepath)
                    })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"扫描目录失败: {str(e)}")
    
    return {
        "directory": directory,
        "file_count": len(audio_files),
        "files": audio_files
    }


@router.get("/export")
async def export_monitor_results():
    """
    导出实时监控结果为压缩包（包含Excel表格和热力图）
    """
    results = monitor_service.get_results(limit=1000, offset=0)
    
    if not results:
        raise HTTPException(status_code=400, detail="没有检测结果可导出")
    
    # 创建导出目录
    export_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_dir = os.path.join("exports", f"monitor_{export_id}")
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        # 1. 创建 Excel 文件
        try:
            import pandas as pd
            
            excel_data = []
            for r in results:
                excel_data.append({
                    "时间": r.get("timestamp", ""),
                    "文件名": r.get("filename", ""),
                    "异常分数": r.get("anomaly_score", 0),
                    "检测结果": r.get("status", ""),
                    "是否异常": "是" if r.get("is_anomaly") else "否"
                })
            
            df = pd.DataFrame(excel_data)
            excel_path = os.path.join(export_dir, "监控结果.xlsx")
            df.to_excel(excel_path, index=False, engine='openpyxl')
        except ImportError:
            # 如果没有 pandas，使用 CSV 格式
            import csv
            csv_path = os.path.join(export_dir, "监控结果.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["时间", "文件名", "异常分数", "检测结果", "是否异常"])
                for r in results:
                    writer.writerow([
                        r.get("timestamp", ""),
                        r.get("filename", ""),
                        r.get("anomaly_score", 0),
                        r.get("status", ""),
                        "是" if r.get("is_anomaly") else "否"
                    ])
        
        # 2. 收集热力图叠加原图
        overlay_dir = os.path.join(export_dir, "热力图叠加原图")
        os.makedirs(overlay_dir, exist_ok=True)
        
        for r in results:
            overlay_path = r.get("overlay_path")
            if overlay_path and os.path.exists(overlay_path):
                # 复制叠加图到导出目录
                filename = os.path.basename(overlay_path)
                dest_path = os.path.join(overlay_dir, filename)
                shutil.copy2(overlay_path, dest_path)
        
        # 3. 打包成 zip
        zip_filename = f"监控结果_{export_id}.zip"
        zip_path = os.path.join("exports", zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)
        
        # 4. 清理临时目录
        shutil.rmtree(export_dir)
        
        # 使用纯英文文件名避免编码问题
        # 在Content-Disposition中使用时间戳格式
        safe_filename = f"monitor_{export_id}.zip"
        
        return FileResponse(
            path=zip_path,
            filename=safe_filename,
            media_type='application/zip'
        )
        
    except Exception as e:
        # 清理临时目录
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")
