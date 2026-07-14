#!/usr/bin/env python3
"""训练验证 — 测试各族代表性算法训练是否可用（10 iters, MVTec bottle）"""
import os, sys, time, json, subprocess, signal, gc
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "pre_trained" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "models" / "pre_trained" / "huggingface"))
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT / "algorithms" / "Anomalib") + ":" + os.environ.get("PYTHONPATH", ""))

REPORT_PATH = PROJECT_ROOT / "records" / "verify_training_now.json"
MVTEC_BOTTLE = PROJECT_ROOT / "data" / "public_dataset" / "mvtec" / "bottle"
TRAINING_TIMEOUT = 180  # 3 min per algorithm


def clear_gpu():
    try:
        import torch
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass


def run_training_dinomaly(alg_name, model_type, model_size):
    """Run Dinomaly training via subprocess"""
    script = PROJECT_ROOT / "algorithms" / "Dinomaly" / "dinomaly_train_evaluate.py"
    cmd = [
        sys.executable, str(script),
        "--data_path", str(MVTEC_BOTTLE),
        "--category", "bottle",
        "--model_type", model_type,
        "--model_size", model_size,
        "--total_iters", "10",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TRAINING_TIMEOUT, cwd=str(PROJECT_ROOT))
        success = result.returncode == 0
        error = result.stderr[-500:] if not success else ""
        return {"success": success, "returncode": result.returncode, "error": error}
    except subprocess.TimeoutExpired:
        return {"success": False, "returncode": -1, "error": "超时"}


def run_training_dinomaly2(alg_name, model_type, model_size):
    """Run Dinomaly2 training via subprocess"""
    script = PROJECT_ROOT / "algorithms" / "Dinomaly2" / "dinomaly_2D.py"
    # Map to backbone names
    backbone_map = {
        ("dinov2", "small"): "dinov2reg_vit_small_14",
        ("dinov2", "base"): "dinov2reg_vit_base_14",
        ("dinov2", "large"): "dinov2reg_vit_large_14",
        ("dinov3", "small"): "dinov3_vit_s16",
        ("dinov3", "base"): "dinov3_vit_b16",
        ("dinov3", "large"): "dinov3_vit_l16",
    }
    backbone = backbone_map.get((model_type, model_size), "dinov2reg_vit_small_14")
    extra_args = []
    if model_type == "dinov3":
        extra_args = ["--use_get_intermediate"]
    cmd = [
        sys.executable, str(script),
        "--data_path", str(MVTEC_BOTTLE.parent),  # parent = mvtec/
        "--category", "bottle",
        "--backbone", backbone,
        "--iters", "10",
        "--batch_size", "2",
    ] + extra_args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TRAINING_TIMEOUT, cwd=str(PROJECT_ROOT))
        success = result.returncode == 0
        error = result.stderr[-500:] if not success else ""
        return {"success": success, "returncode": result.returncode, "error": error}
    except subprocess.TimeoutExpired:
        return {"success": False, "returncode": -1, "error": "超时"}


def run_training_ader(alg_name):
    """Run ADer training via backend API"""
    import asyncio
    from backend.api.training import TrainingRequest, start_training, TRAINING_TASKS

    request = TrainingRequest(
        categories=["bottle"],
        data_source="mvtec",
        algorithm_family="ader",
        algorithm_name=alg_name,
        model_type="default",
        model_size="default",
        total_iters=10,
        batch_size=2,
        gpu_id=0,
    )

    try:
        resp = asyncio.run(start_training(request))
        task_id = resp.get("task_id")
        if not resp.get("success"):
            return {"success": False, "error": f"启动失败: {resp.get('message', '')}",
                    "test": "training", "algorithm": alg_name}

        # Wait for completion
        timeout = 300
        elapsed = 0
        while elapsed < timeout:
            task = TRAINING_TASKS.get(task_id, {})
            status = task.get("status", "")
            if status == "completed":
                return {"success": True, "error": "", "test": "training", "algorithm": alg_name}
            elif status == "failed":
                log = (task.get("log", "") or "")[-500:]
                progress = task.get("progress", "")
                return {"success": False, "error": f"{progress}\n{log}",
                        "test": "training", "algorithm": alg_name}
            import asyncio as a
            a.sleep(2)
            elapsed += 2
        # Timeout
        task = TRAINING_TASKS.get(task_id, {})
        proc = task.get("process")
        if proc and proc.poll() is None:
            proc.terminate()
        task["status"] = "failed"
        return {"success": False, "error": "超时", "test": "training", "algorithm": alg_name}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "test": "training", "algorithm": alg_name}


def main():
    print("=" * 60)
    print("训练验证 — 代表性算法")
    print(f"时间: {datetime.now().isoformat()}")
    print("=" * 60)

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")

    results = {}
    test_suite = [
        # (family, algorithm, test_func, args)
        ("Dinomaly", "dinomaly_dinov3_small", run_training_dinomaly, ("dinov3", "small")),
        ("Dinomaly2", "dinomaly2_dinov2_small", run_training_dinomaly2, ("dinov2", "small")),
        # ADer: test each
        ("ADer", "simplenet", None, None),  # special: via API
    ]

    # Test Dinomaly
    print("\n--- Dinomaly: dinomaly_dinov3_small ---")
    clear_gpu()
    r = run_training_dinomaly("dinomaly_dinov3_small", "dinov3", "small")
    results["dinomaly_dinov3_small"] = {"test": "training", "algorithm": "dinomaly_dinov3_small",
                                         "family": "Dinomaly", "success": r["success"],
                                         "error": r.get("error", ""), "details": r}
    print(f"  {'✅' if r['success'] else '❌'} {'通过' if r['success'] else r['error'][:120]}")

    # Test Dinomaly2
    print("\n--- Dinomaly2: dinomaly2_dinov2_small ---")
    clear_gpu()
    r = run_training_dinomaly2("dinomaly2_dinov2_small", "dinov2", "small")
    results["dinomaly2_dinov2_small"] = {"test": "training", "algorithm": "dinomaly2_dinov2_small",
                                          "family": "Dinomaly2", "success": r["success"],
                                          "error": r.get("error", ""), "details": r}
    print(f"  {'✅' if r['success'] else '❌'} {'通过' if r['success'] else r['error'][:120]}")

    # Test ADer algorithms
    ader_algs = ["simplenet", "invad", "vitad", "unad", "cflow", "pyramidflow",
                 "mambaad", "destseg", "realnet", "rdpp"]
    for alg in ader_algs:
        print(f"\n--- ADer: {alg} ---")
        clear_gpu()
        r = run_training_ader(alg)
        r["family"] = "ADer"
        results[alg] = r
        status = "✅ 通过" if r["success"] else f"❌ {r.get('error', '')[:120]}"
        print(f"  {status}")

    # Save results
    passed = sum(1 for v in results.values() if v["success"])
    failed = sum(1 for v in results.values() if not v["success"])
    report = {"timestamp": datetime.now().isoformat(), "summary": {"total": len(results), "passed": passed, "failed": failed}, "results": results}
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"训练验证完成: {passed} 通过, {failed} 失败 (共 {len(results)})")
    for alg, r in results.items():
        if not r["success"]:
            print(f"  ❌ {alg}: {r.get('error', '')[:150]}")
    print(f"报告: {REPORT_PATH}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
