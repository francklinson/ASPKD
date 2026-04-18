"""
检测相关 API
"""
import os
import uuid
import zipfile
import shutil
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import quote
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.core.websocket import websocket_manager

router = APIRouter()

# 设备列表缓存
_devices_cache: Dict[str, Any] = {"data": None, "timestamp": 0, "ttl": 10}  # 10秒缓存


def _get_cached_devices() -> Optional[Dict[str, Any]]:
    """获取缓存的设备列表"""
    global _devices_cache
    current_time = time.time()
    if _devices_cache["data"] is not None and (current_time - _devices_cache["timestamp"]) < _devices_cache["ttl"]:
        return _devices_cache["data"]
    return None


def _set_cached_devices(data: Dict[str, Any]) -> None:
    """设置设备列表缓存"""
    global _devices_cache
    _devices_cache["data"] = data
    _devices_cache["timestamp"] = time.time()

# task_manager 在函数内部延迟导入，避免启动时触发 torch 初始化


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
    save_results: bool = Form(True),
    reference_audio: Optional[str] = Form(None)
):
    """
    上传音频文件并创建检测任务
    
    - **files**: 音频文件列表 (支持 wav, mp3, flac 等格式)
    - **algorithm**: 检测算法名称
    - **device**: 运行设备 (auto, cpu, cuda:0 等)
    - **save_results**: 是否保存结果文件
    - **reference_audio**: 参考音频文件路径（可选，从参考音频库中选择）
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
    
    # 注意：参考音频验证已在 preprocessing.py 中处理
    # Shazam 模式只需要数据库中有指纹，不需要实际文件存在
    # 非 Shazam 模式会在创建预处理器时检查文件是否存在
    
    # 延迟导入 task_manager，避免启动时触发 torch 初始化
    from backend.core.task_manager import task_manager
    
    # 创建任务
    task_id = await task_manager.create_task(
        files=files,
        algorithm=algorithm,
        device=device,
        save_results=save_results,
        reference_audio=reference_audio
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
    
    from backend.core.task_manager import task_manager
    
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
    from backend.core.task_manager import task_manager
    result = task_manager.get_task_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return DetectionResult(**result)


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """取消正在进行的任务"""
    from backend.core.task_manager import task_manager
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


@router.get("/reference-audios")
async def get_available_reference_audios():
    """
    获取可用的参考音频列表

    返回参考音频库中所有可用的参考音频，供离线检测时选择使用
    """
    # 从配置文件读取默认参考音频
    import yaml
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        default_ref = config.get('preprocessing', {}).get('ref_file', '')
    except Exception:
        default_ref = ''
    
    try:
        from core.shazam import AudioFingerprinter

        with AudioFingerprinter() as fp:
            references = fp.get_all_references()

        # 转换为前端友好的格式
        ref_list = []
        for ref in references:
            ref_list.append({
                "id": ref.get("music_id"),
                "name": ref.get("name"),
                "path": ref.get("path"),
                "hash_count": ref.get("hash_count", 0)
            })

        return {
            "references": ref_list,
            "total": len(ref_list),
            "default": default_ref  # 从配置文件读取的默认参考音频路径
        }
    except Exception as e:
        print(f"[Detection API] 获取参考音频列表失败: {e}")
        # 如果无法连接数据库，返回默认配置中的参考音频
        return {
            "references": [],
            "total": 0,
            "default": default_ref,
            "error": f"无法获取参考音频库: {str(e)}"
        }


@router.get("/devices")
async def get_available_devices():
    """获取可用设备列表（包含显存信息）"""
    import sys
    import os

    # 检查缓存
    cached = _get_cached_devices()
    if cached is not None:
        return cached

    # 仅在调试模式下输出详细日志
    debug_mode = os.environ.get('ASD_DEBUG', 'false').lower() == 'true'

    if debug_mode:
        # 检查 torch 是否已经在 sys.modules 中
        if 'torch' in sys.modules:
            print(f"[Devices] WARNING: torch already in sys.modules before explicit import!")
            import torch
            print(f"[Devices] torch.cuda.is_available() before explicit import: {torch.cuda.is_available()}")
        else:
            print(f"[Devices] torch not in sys.modules before import - good")
        print(f"[Devices] CUDA_VISIBLE_DEVICES before import torch: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")

    import torch

    if debug_mode:
        print(f"[Devices] CUDA_VISIBLE_DEVICES after import torch: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
        print(f"[Devices] ========== 开始查询GPU设备 ==========")
        print(f"[Devices] torch 版本: {torch.__version__}")

    devices = [
        {"id": "auto", "name": "自动选择 (GPU优先)", "type": "auto", "info": "自动选择可用显存最多的GPU"}
    ]

    # CPU 信息
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        mem = psutil.virtual_memory()
        devices.append({
            "id": "cpu",
            "name": f"CPU ({cpu_count}核)",
            "type": "cpu",
            "info": f"系统内存: {mem.total / (1024**3):.1f}GB, 可用: {mem.available / (1024**3):.1f}GB"
        })
        if debug_mode:
            print(f"[Devices] CPU 信息: {cpu_count}核, 内存 {mem.total / (1024**3):.1f}GB")
    except Exception as e:
        if debug_mode:
            print(f"[Devices] CPU 信息获取失败: {e}")
        devices.append({"id": "cpu", "name": "CPU", "type": "cpu", "info": "纯CPU运行"})

    # GPU 信息
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        try:
            gpu_count = torch.cuda.device_count()
            gpu_list = []

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)

                # 获取显存信息
                try:
                    torch.cuda.set_device(i)
                    props = torch.cuda.get_device_properties(i)
                    total_mem = props.total_memory / (1024**3)  # GB
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    free_mem = total_mem - allocated

                    if debug_mode:
                        print(f"[Devices] GPU {i}: {gpu_name}, 显存: 总={total_mem:.2f}GB, 已用={allocated:.2f}GB, 可用={free_mem:.2f}GB")

                    gpu_info = {
                        "id": f"cuda:{i}",
                        "name": f"GPU {i}: {gpu_name}",
                        "type": "cuda",
                        "info": f"总显存: {total_mem:.1f}GB | 已用: {allocated:.1f}GB | 可用: {free_mem:.1f}GB",
                        "total_memory_gb": round(total_mem, 2),
                        "free_memory_gb": round(free_mem, 2),
                        "usage_percent": round((allocated / total_mem) * 100, 1) if total_mem > 0 else 0
                    }
                except Exception as e:
                    if debug_mode:
                        print(f"[Devices] GPU {i} 显存信息获取失败: {e}")
                    gpu_info = {
                        "id": f"cuda:{i}",
                        "name": f"GPU {i}: {gpu_name}",
                        "type": "cuda",
                        "info": f"显存信息获取失败: {str(e)}"
                    }

                gpu_list.append(gpu_info)

            # 按可用显存排序（可用显存多的在前）
            gpu_list.sort(key=lambda x: x.get("free_memory_gb", 0), reverse=True)

            # 标记推荐使用的 GPU（仅添加 recommended 字段，显示逻辑由前端处理）
            for i, gpu in enumerate(gpu_list):
                if i == 0 and gpu.get("free_memory_gb", 0) > 1.0:  # 至少有1GB可用
                    gpu["recommended"] = True
                devices.append(gpu)

            if debug_mode:
                print(f"[Devices] 成功查询 {len(gpu_list)} 个 GPU")
        except Exception as e:
            if debug_mode:
                print(f"[Devices] GPU 查询过程出错: {e}")
    else:
        if debug_mode:
            print(f"[Devices] CUDA 不可用")

    gpu_count = torch.cuda.device_count() if cuda_available else 0
    result = {"devices": devices, "gpu_count": gpu_count}

    # 缓存结果
    _set_cached_devices(result)

    if debug_mode:
        print(f"[Devices] 返回设备列表: {len(devices)} 个设备, GPU 数量: {gpu_count}")
        print(f"[Devices] ========== GPU设备查询结束 ==========")

    return result


@router.get("/audio/{audio_path:path}")
async def get_audio_file(audio_path: str):
    """
    获取音频切片文件（用于试听）
    
    - **audio_path**: 音频文件的相对路径（如: slice/audio/xxx.wav）
    """
    import os
    from fastapi.responses import FileResponse
    
    # 安全检查：确保路径不包含 .. 等危险字符
    if ".." in audio_path or audio_path.startswith("/"):
        raise HTTPException(status_code=400, detail="无效的路径")
    
    # 构建完整路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    full_path = os.path.join(project_root, audio_path)
    
    # 检查文件是否存在
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"音频文件不存在: {audio_path}")
    
    # 检查是否是文件
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=400, detail="无效的文件路径")
    
    # 返回文件
    return FileResponse(
        path=full_path,
        media_type='audio/wav',
        filename=os.path.basename(full_path)
    )


@router.get("/export/{task_id}")
async def export_task_results(task_id: str):
    """
    导出检测结果为压缩包（包含Excel表格和热力图）
    
    - **task_id**: 任务ID
    """
    from backend.core.task_manager import task_manager
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
                    "是否异常": "是" if r.get("is_anomaly") else "否"
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
                writer.writerow(["文件名", "异常分数", "检测结果", "是否异常"])
                for r in results:
                    writer.writerow([
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
        
        # 使用纯英文文件名避免编码问题
        export_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"result_{export_id}.zip"
        
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
