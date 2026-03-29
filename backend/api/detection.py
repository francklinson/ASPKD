"""
检测相关 API
"""
import os
import uuid
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel

from backend.core.task_manager import task_manager

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
