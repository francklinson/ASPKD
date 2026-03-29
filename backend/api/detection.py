"""
检测相关 API
"""
import os
import uuid
import zipfile
import shutil
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.task_manager import task_manager
from backend.core.websocket import websocket_manager

router = APIRouter()


class DetectionRequest(BaseModel):
    """检测请求"""
    algorithm: str = "dinomaly_dinov3_small"
    device: str = "auto"
    save_results: bool = True


class DetectionResponse(BaseModel):
    """检测响应"""
    task_id: str
    status: str
    message: str
    queue_position: Optional[int] = None


class DetectionResult(BaseModel):
    """检测结果"""
    task_id: str
    status: str
    progress: float
    results: Optional[List[dict]] = None
    error: Optional[str] = None


@router.post("/upload", response_model=DetectionResponse)
async def upload_and_detect(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    algorithm: str = Form("dinomaly_dinov3_small"),
    device: str = Form("auto"),
    save_results: bool = Form(True)
):
    """
    上传音频文件并创建检测任务
    
    - **files**: 音频文件列表 (支持 wav, mp3, flac 等格式)
    - **algorithm**: 检测算法名称
    - **device**: 运行设备 (auto, cpu, cuda:0 等)
    - **save_results**: 是否保存结果文件
    """
    # 验证文件
    if not files:
        raise HTTPException(status_code=400, detail="未上传文件")
    
    allowed_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件格式: {file.filename}, 仅支持 {allowed_extensions}"
            )
    
    # 创建任务
    task_id = await task_manager.create_task(
        files=files,
        algorithm=algorithm,
        device=device,
        save_results=save_results
    )
    
    # 获取队列位置
    queue_position = task_manager.get_queue_position(task_id)
    
    return DetectionResponse(
        task_id=task_id,
        status="queued",
        message=f"任务已创建，正在排队处理",
        queue_position=queue_position
    )


@router.post("/batch", response_model=DetectionResponse)
async def detect_batch(
    request: DetectionRequest,
    file_paths: List[str]
):
    """
    对服务器上的文件进行批量检测（用于目录监控）
    
    - **file_paths**: 服务器上的文件路径列表
    - **request**: 检测配置
    """
    # 验证文件存在
    for path in file_paths:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"文件不存在: {path}")
    
    task_id = await task_manager.create_batch_task(
        file_paths=file_paths,
        algorithm=request.algorithm,
        device=request.device,
        save_results=request.save_results
    )
    
    return DetectionResponse(
        task_id=task_id,
        status="queued",
        message="批量检测任务已创建",
        queue_position=task_manager.get_queue_position(task_id)
    )


@router.get("/result/{task_id}", response_model=DetectionResult)
async def get_result(task_id: str):
    """获取检测结果"""
    result = task_manager.get_task_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return DetectionResult(**result)


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """取消正在进行的任务"""
    success = await task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="任务不存在或已完成")
    
    return {"status": "cancelled", "task_id": task_id}


@router.get("/algorithms")
async def get_available_algorithms():
    """获取可用算法列表"""
    algorithms = [
        {
            "id": "dinomaly_dinov3_small",
            "name": "Dinomaly DINOv3 Small",
            "description": "基于DINOv3轻量级模型",
            "type": "feature_based"
        },
        {
            "id": "dinomaly_dinov3_large",
            "name": "Dinomaly DINOv3 Large",
            "description": "基于DINOv3大模型",
            "type": "feature_based"
        },
        {
            "id": "dinomaly_dinov2_small",
            "name": "Dinomaly DINOv2 Small",
            "description": "基于DINOv2轻量级模型",
            "type": "feature_based"
        },
        {
            "id": "dinomaly_dinov2_large",
            "name": "Dinomaly DINOv2 Large",
            "description": "基于DINOv2大模型",
            "type": "feature_based"
        },
    ]
    
    return {"algorithms": algorithms}


@router.get("/devices")
async def get_available_devices():
    """获取可用设备列表"""
    import torch
    
    devices = [
        {"id": "auto", "name": "自动选择 (GPU优先)", "type": "auto"},
        {"id": "cpu", "name": "CPU (纯CPU运行)", "type": "cpu"}
    ]
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            devices.append({
                "id": f"cuda:{i}",
                "name": f"GPU {i}: {gpu_name}",
                "type": "cuda"
            })
    
    return {"devices": devices}


@router.get("/export/{task_id}")
async def export_task_results(task_id: str):
    """
    导出检测结果为压缩包（包含Excel表格和热力图）
    
    - **task_id**: 任务ID
    """
    result = task_manager.get_task_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if result["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成，无法导出")
    
    results = result.get("results", [])
    if not results:
        raise HTTPException(status_code=400, detail="没有检测结果可导出")
    
    # 创建导出目录
    export_dir = os.path.join("exports", task_id)
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        # 1. 创建 Excel 文件
        try:
            import pandas as pd
            
            excel_data = []
            for r in results:
                excel_data.append({
                    "文件名": r.get("filename", ""),
                    "异常分数": r.get("anomaly_score", 0),
                    "检测结果": r.get("status", ""),
                    "是否异常": "是" if r.get("is_anomaly") else "否",
                    "热力图路径": r.get("heatmap_path", "")
                })
            
            df = pd.DataFrame(excel_data)
            excel_path = os.path.join(export_dir, "检测结果.xlsx")
            df.to_excel(excel_path, index=False, engine='openpyxl')
        except ImportError:
            # 如果没有 pandas，使用 CSV 格式
            import csv
            csv_path = os.path.join(export_dir, "检测结果.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["文件名", "异常分数", "检测结果", "是否异常", "热力图路径"])
                for r in results:
                    writer.writerow([
                        r.get("filename", ""),
                        r.get("anomaly_score", 0),
                        r.get("status", ""),
                        "是" if r.get("is_anomaly") else "否",
                        r.get("heatmap_path", "")
                    ])
        
        # 2. 收集热力图
        heatmap_dir = os.path.join(export_dir, "热力图")
        os.makedirs(heatmap_dir, exist_ok=True)
        
        for r in results:
            heatmap_path = r.get("heatmap_path")
            if heatmap_path and os.path.exists(heatmap_path):
                # 复制热力图到导出目录
                filename = os.path.basename(heatmap_path)
                dest_path = os.path.join(heatmap_dir, filename)
                shutil.copy2(heatmap_path, dest_path)
        
        # 3. 打包成 zip
        zip_filename = f"检测结果_{task_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join("exports", zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)
        
        # 4. 清理临时目录
        shutil.rmtree(export_dir)
        
        return FileResponse(
            path=zip_path,
            filename=zip_filename,
            media_type='application/zip',
            content_disposition=f'attachment; filename="{zip_filename}"'
        )
        
    except Exception as e:
        # 清理临时目录
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")
