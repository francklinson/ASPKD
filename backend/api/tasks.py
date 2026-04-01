"""
任务管理 API
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException

from backend.core.task_manager import task_manager

router = APIRouter()


@router.get("/list")
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    获取任务列表
    
    - **status**: 按状态筛选 (pending, running, completed, failed, cancelled)
    - **limit**: 返回数量
    - **offset**: 偏移量
    """
    tasks = task_manager.list_tasks(
        status=status,
        limit=limit,
        offset=offset
    )
    
    return {
        "tasks": tasks,
        "total": task_manager.get_task_count(),
        "running": task_manager.get_running_count(),
        "queued": task_manager.get_queued_count()
    }


@router.get("/stats")
async def get_task_stats():
    """获取任务统计信息"""
    stats = task_manager.get_stats()
    return stats


@router.post("/cleanup")
async def cleanup_old_tasks(keep_days: int = 7, clear_all: bool = False, include_files: bool = True):
    """
    清理旧任务记录及关联文件
    
    - **keep_days**: 保留最近几天的任务（默认7天）
    - **clear_all**: 是否清理所有已完成/失败/取消的任务（优先于keep_days）
    - **include_files**: 是否同时清理上传的文件、热力图、导出文件（默认True）
    """
    import os
    import shutil
    from datetime import datetime
    
    file_stats = {"uploads": 0, "visualize": 0, "exports": 0, "errors": []}
    
    if clear_all:
        removed_count = await task_manager.clear_all_tasks()
    else:
        removed_count = await task_manager.cleanup_old_tasks(keep_days)
    
    # 清理物理文件
    if include_files:
        # 1. 清理 uploads/ 目录（上传的音频文件）
        uploads_dir = "uploads"
        if os.path.exists(uploads_dir):
            try:
                for item in os.listdir(uploads_dir):
                    item_path = os.path.join(uploads_dir, item)
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        file_stats["uploads"] += 1
                    except Exception as e:
                        file_stats["errors"].append(f"uploads/{item}: {str(e)}")
            except Exception as e:
                file_stats["errors"].append(f"uploads: {str(e)}")
        
        # 2. 清理 visualize/ 目录（热力图）
        visualize_dir = "visualize"
        if os.path.exists(visualize_dir):
            try:
                for item in os.listdir(visualize_dir):
                    item_path = os.path.join(visualize_dir, item)
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        file_stats["visualize"] += 1
                    except Exception as e:
                        file_stats["errors"].append(f"visualize/{item}: {str(e)}")
            except Exception as e:
                file_stats["errors"].append(f"visualize: {str(e)}")
        
        # 3. 清理 exports/ 目录（导出的Excel/Zip文件）
        exports_dir = "exports"
        if os.path.exists(exports_dir):
            try:
                for item in os.listdir(exports_dir):
                    item_path = os.path.join(exports_dir, item)
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        file_stats["exports"] += 1
                    except Exception as e:
                        file_stats["errors"].append(f"exports/{item}: {str(e)}")
            except Exception as e:
                file_stats["errors"].append(f"exports: {str(e)}")
        
        # 4. 清理 slice/ 目录（临时切片文件）
        slice_dir = "slice"
        if os.path.exists(slice_dir):
            try:
                for item in os.listdir(slice_dir):
                    item_path = os.path.join(slice_dir, item)
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        file_stats["slice"] += 1
                    except Exception as e:
                        file_stats["errors"].append(f"slice/{item}: {str(e)}")
            except Exception as e:
                file_stats["errors"].append(f"slice: {str(e)}")
    
    return {
        "status": "cleaned",
        "removed_count": removed_count,
        "file_stats": file_stats,
        "message": f"清理了 {removed_count} 个任务记录，{file_stats.get('uploads', 0)} 个上传文件，{file_stats.get('slice', 0)} 个临时切片，{file_stats.get('visualize', 0)} 个热力图，{file_stats.get('exports', 0)} 个导出文件"
    }


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """删除任务记录"""
    success = task_manager.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {"status": "deleted", "task_id": task_id}
