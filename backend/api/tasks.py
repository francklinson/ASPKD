"""
任务管理 API
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException

router = APIRouter()

# task_manager 在函数内部延迟导入，避免启动时触发 torch 初始化


@router.get("/all")
async def list_all_tasks(
    task_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    获取所有类型的任务列表（统一接口）

    - **task_type**: 按类型筛选 (offline_audio, online, custom_detection, training)
    - **status**: 按状态筛选 (pending, running, completed, failed, cancelled)
    - **limit**: 返回数量
    - **offset**: 偏移量
    """
    all_tasks = []

    # 1. 离线音频检测任务
    if not task_type or task_type == "offline_audio":
        try:
            from backend.core.task_manager import task_manager
            for tid, t in task_manager.tasks.items():
                all_tasks.append({
                    "id": t.id,
                    "type": "offline_audio",
                    "status": t.status,
                    "algorithm": t.algorithm,
                    "algorithm_family": "",
                    "progress": t.progress,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "started_at": t.started_at.isoformat() if t.started_at else None,
                    "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                    "file_count": len(t.files),
                    "error": t.error,
                    "client_name": None,
                    "category": None,
                })
        except Exception:
            pass

    # 2. 在线检测任务
    if not task_type or task_type == "online":
        try:
            from backend.core.client_monitor_service import client_detection_service
            results = client_detection_service.get_results(limit=1000, offset=0)
            for r in results:
                ts = r.get("timestamp", "")
                all_tasks.append({
                    "id": f"online_{r.get('result_id', '')}",
                    "type": "online",
                    "status": "completed",
                    "algorithm": r.get("algorithm", ""),
                    "algorithm_family": "",
                    "progress": 100.0,
                    "created_at": ts,
                    "started_at": ts,
                    "completed_at": ts,
                    "file_count": 1,
                    "error": None,
                    "client_name": r.get("client_name"),
                    "category": None,
                })
        except Exception:
            pass

    # 3. 自定义图片检测任务
    if not task_type or task_type == "custom_detection":
        try:
            from backend.api.custom_detection import CUSTOM_TASKS
            for tid, t in CUSTOM_TASKS.items():
                # 统一状态映射
                status_val = t.get("status", "queued")
                if status_val == "processing":
                    status_val = "running"
                elif status_val == "queued":
                    status_val = "pending"
                results_list = t.get("results", [])
                all_tasks.append({
                    "id": tid,
                    "type": "custom_detection",
                    "status": status_val,
                    "algorithm": t.get("algorithm", ""),
                    "algorithm_family": t.get("algorithm_family", ""),
                    "progress": t.get("progress", 0.0),
                    "created_at": t.get("created_at"),
                    "started_at": t.get("started_at"),
                    "completed_at": t.get("completed_at"),
                    "file_count": t.get("file_count", len(results_list)),
                    "error": t.get("error"),
                    "client_name": None,
                    "category": None,
                })
        except Exception:
            pass

    # 4. 模型训练任务
    if not task_type or task_type == "training":
        try:
            from backend.api.training import TRAINING_TASKS
            for tid, t in TRAINING_TASKS.items():
                # 统一状态映射
                status_val = t.get("status", "pending")
                if status_val == "processing":
                    status_val = "running"
                # 训练任务 progress 是字符串，用 progress_pct 数字字段
                progress_val = t.get("progress_pct", 0.0)
                if isinstance(progress_val, str):
                    try:
                        progress_val = float(progress_val)
                    except (ValueError, TypeError):
                        progress_val = 0.0
                all_tasks.append({
                    "id": tid,
                    "type": "training",
                    "status": status_val,
                    "algorithm": t.get("algorithm_name", ""),
                    "algorithm_family": t.get("algorithm_family", ""),
                    "progress": progress_val,
                    "created_at": t.get("created_at"),
                    "started_at": t.get("started_at"),
                    "completed_at": t.get("completed_at"),
                    "file_count": 0,
                    "error": t.get("error") or (t.get("progress") if isinstance(t.get("progress"), str) and "失败" in t.get("progress", "") else None),
                    "client_name": None,
                    "category": ", ".join(t.get("categories", [])),
                })
        except Exception:
            pass

    # 按状态筛选
    if status:
        all_tasks = [t for t in all_tasks if t["status"] == status]

    # 按时间倒序
    all_tasks.sort(key=lambda x: x.get("created_at") or "", reverse=True)

    total = len(all_tasks)
    page = all_tasks[offset:offset + limit]

    return {"tasks": page, "total": total}


@router.get("/stats/all")
async def get_all_task_stats():
    """获取所有类型任务的统计信息"""
    stats = {
        "total": 0, "running": 0, "pending": 0, "completed": 0, "failed": 0, "cancelled": 0,
        "by_type": {
            "offline_audio": {"total": 0, "running": 0, "pending": 0, "completed": 0, "failed": 0},
            "online": {"total": 0, "running": 0, "pending": 0, "completed": 0, "failed": 0},
            "custom_detection": {"total": 0, "running": 0, "pending": 0, "completed": 0, "failed": 0},
            "training": {"total": 0, "running": 0, "pending": 0, "completed": 0, "failed": 0},
        }
    }

    # 离线音频
    try:
        from backend.core.task_manager import task_manager
        ts = task_manager.get_stats()
        for k in ["total", "running", "pending", "completed", "failed"]:
            stats["by_type"]["offline_audio"][k] = ts.get(k, 0)
            stats[k] += ts.get(k, 0)
        stats["total"] += 0  # already counted
    except Exception:
        pass

    # 在线检测
    try:
        from backend.core.client_monitor_service import client_detection_service
        online_count = client_detection_service.get_results_count()
        stats["by_type"]["online"]["total"] = online_count
        stats["by_type"]["online"]["completed"] = online_count
        stats["total"] += online_count
        stats["completed"] += online_count
    except Exception:
        pass

    # 自定义图片检测
    try:
        from backend.api.custom_detection import CUSTOM_TASKS
        for t in CUSTOM_TASKS.values():
            s = t.get("status", "queued")
            if s == "processing": s = "running"
            elif s == "queued": s = "pending"
            stats["by_type"]["custom_detection"]["total"] += 1
            if s in stats["by_type"]["custom_detection"]:
                stats["by_type"]["custom_detection"][s] += 1
            stats["total"] += 1
            if s in stats:
                stats[s] += 1
    except Exception:
        pass

    # 训练任务
    try:
        from backend.api.training import TRAINING_TASKS
        for t in TRAINING_TASKS.values():
            s = t.get("status", "pending")
            if s == "processing": s = "running"
            stats["by_type"]["training"]["total"] += 1
            if s in stats["by_type"]["training"]:
                stats["by_type"]["training"][s] += 1
            stats["total"] += 1
            if s in stats:
                stats[s] += 1
    except Exception:
        pass

    return stats


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
    from backend.core.task_manager import task_manager
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
    from backend.core.task_manager import task_manager
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
    from backend.core.task_manager import task_manager

    # 动态检测项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    file_stats = {"uploads": 0, "visualize": 0, "exports": 0, "errors": []}

    if clear_all:
        removed_count = await task_manager.clear_all_tasks()
    else:
        removed_count = await task_manager.cleanup_old_tasks(keep_days)

    # 清理物理文件
    if include_files:
        # 1. 清理 uploads/ 目录（上传的音频文件）
        uploads_dir = os.path.join(project_root, "data", "uploads")
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

        # 2. 清理 data/output/vis/ 目录（热力图）
        visualize_dir = os.path.join(project_root, "data", "output", "vis")
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
                        file_stats["errors"].append(f"data/output/vis/{item}: {str(e)}")
            except Exception as e:
                file_stats["errors"].append(f"data/output/vis: {str(e)}")

        # 3. 清理 data/output/exports/ 目录（导出的Excel/Zip文件）
        exports_dir = os.path.join(project_root, "data", "output", "exports")
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
                        file_stats["errors"].append(f"data/output/exports/{item}: {str(e)}")
            except Exception as e:
                file_stats["errors"].append(f"data/output/exports: {str(e)}")

        # 4. 清理 data/output/slices/ 目录（临时切片文件）
        slice_dir = os.path.join(project_root, "data", "output", "slices")
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
                        file_stats["errors"].append(f"data/output/slices/{item}: {str(e)}")
            except Exception as e:
                file_stats["errors"].append(f"data/output/slices: {str(e)}")
    
    return {
        "status": "cleaned",
        "removed_count": removed_count,
        "file_stats": file_stats,
        "message": f"清理了 {removed_count} 个任务记录，{file_stats.get('uploads', 0)} 个上传文件，{file_stats.get('slice', 0)} 个临时切片，{file_stats.get('visualize', 0)} 个热力图，{file_stats.get('exports', 0)} 个导出文件"
    }


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """删除任务记录"""
    from backend.core.task_manager import task_manager
    success = task_manager.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {"status": "deleted", "task_id": task_id}
