"""
系统状态 API
提供算法可用性、服务健康等系统级信息
"""
from typing import Dict, Any
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
