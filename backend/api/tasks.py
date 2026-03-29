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
async def cleanup_old_tasks(keep_days: int = 7):
    """
    清理旧任务记录
    
    - **keep_days**: 保留最近几天的任务
    """
    removed_count = await task_manager.cleanup_old_tasks(keep_days)
    return {
        "status": "cleaned",
        "removed_count": removed_count,
        "message": f"清理了 {removed_count} 个旧任务"
    }


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """删除任务记录"""
    success = task_manager.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {"status": "deleted", "task_id": task_id}
