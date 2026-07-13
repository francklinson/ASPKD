"""
系统状态 API
提供算法可用性、服务健康等系统级信息
"""
from typing import Dict, Any, List
from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/system/algorithm-availability")
async def get_algorithm_availability() -> Dict[str, Any]:
    """
    获取所有算法的可用性状态。

    返回每个算法的推理/训练可用性、缺失的库和模型文件、原因说明。
    结果由服务启动时的 check_all_algorithms() 缓存，不会重复检查。
    """
    from backend.algorithm_availability import (
        get_all_availability, is_cache_populated, get_available_algorithms,
    )

    if not is_cache_populated():
        return {
            "status": "not_checked",
            "message": "算法可用性检查尚未执行，请等待服务完全启动",
            "algorithms": {},
            "summary": {},
        }

    results = get_all_availability()
    available_ids = get_available_algorithms()

    algorithms = {}
    for alg_id, avail in sorted(results.items()):
        algorithms[alg_id] = {
            "family": avail.family,
            "inference_available": avail.inference_available,
            "training_available": avail.training_available,
            "missing_libraries": avail.missing_libraries,
            "missing_model_files": avail.missing_model_files,
            "reasons": avail.reasons,
        }

    return {
        "status": "checked",
        "algorithms": algorithms,
        "available_algorithms": available_ids,
        "summary": {
            "total": len(results),
            "available_for_inference": sum(1 for r in results.values() if r.inference_available),
            "available_for_training": sum(1 for r in results.values() if r.training_available),
        },
    }


@router.get("/system/health")
async def system_health() -> Dict[str, Any]:
    """系统健康检查（含算法可用性摘要）"""
    from backend.algorithm_availability import (
        is_cache_populated, get_all_availability,
    )

    health = {
        "status": "ok",
        "algorithms_checked": is_cache_populated(),
    }

    if is_cache_populated():
        results = get_all_availability()
        health["algorithms_summary"] = {
            "total": len(results),
            "available_for_inference": sum(1 for r in results.values() if r.inference_available),
            "available_for_training": sum(1 for r in results.values() if r.training_available),
        }

    return health


@router.get("/system/gpus")
async def list_gpus() -> Dict[str, Any]:
    """获取可用 GPU 列表及显存信息"""
    import torch
    if not torch.cuda.is_available():
        return {"available": False, "gpus": []}

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_total = props.total_memory / (1024 ** 3)
        mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        mem_free = mem_total - mem_allocated
        gpus.append({
            "id": i,
            "name": props.name,
            "memory_total_gb": round(mem_total, 1),
            "memory_free_gb": round(mem_free, 1),
            "memory_used_gb": round(mem_allocated, 1),
        })
    return {"available": True, "gpus": gpus}
