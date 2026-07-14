#!/usr/bin/env python3
"""
快速算法验证 — 推理 + 训练
直接调用 detector API，逐个测试，记录结果。
"""
import os
import sys
import time
import json
import traceback
import gc
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "algorithms"))
sys.path.insert(0, str(PROJECT_ROOT / "algorithms" / "Dinomaly2"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "pre_trained" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "models" / "pre_trained" / "huggingface"))

TEST_IMG = str(PROJECT_ROOT / "data" / "public_dataset" / "mvtec" / "bottle" / "test" / "good" / "000.png")
REPORT_DIR = PROJECT_ROOT / "records"
REPORT_PATH = REPORT_DIR / "verify_now_report.json"


def clear_gpu():
    try:
        import torch
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass


def test_inference(alg_name: str) -> dict:
    """直接测试推理"""
    result = {"algorithm": alg_name, "test": "inference", "success": False}
    detector = None
    t0 = time.time()
    try:
        from algorithms import create_detector
        clear_gpu()
        detector = create_detector(alg_name)
        detector.load_model()
        pred = detector.predict(TEST_IMG)
        result["success"] = True
        result["score"] = round(float(getattr(pred, 'anomaly_score', -1)), 6)
        result["has_heatmap"] = getattr(pred, 'anomaly_map', None) is not None
        result["is_anomaly"] = bool(getattr(pred, 'is_anomaly', False))
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        if detector:
            try: detector.release()
            except: pass
        clear_gpu()
    result["time_s"] = round(time.time() - t0, 2)
    return result


# 算法定义
INFERENCE_ALGORITHMS = {
    "Dinomaly": [
        "dinomaly_dinov3_small", "dinomaly_dinov3_base", "dinomaly_dinov3_large",
        "dinomaly_dinov2_small", "dinomaly_dinov2_base", "dinomaly_dinov2_large",
    ],
    "Dinomaly2": [
        "dinomaly2_dinov2_small", "dinomaly2_dinov2_base", "dinomaly2_dinov2_large",
        "dinomaly2_dinov3_small", "dinomaly2_dinov3_base", "dinomaly2_dinov3_large",
    ],
    "Anomalib": [
        # 需要预训练权重的（从saved/mvtec/加载已训练模型）
        "patchcore", "csflow", "draem", "dsr", "fre",
        "reverse_distillation", "ganomaly", "supersimplenet", "uninet",
        # checkpoint-free（直接从config构建）
        "padim", "stfpm", "fastflow", "efficient_ad", "uflow",
        "cfm", "vlm_ad",
        # new in v2.5
        "general_ad", "glass", "inp_former", "l2bt", "patchflow",
        # memory bank dependent (need training)
        "cfa", "dfkde", "dfm", "winclip", "anomalyvfm", "anomaly_dino",
    ],
    "ADer": [
        "mambaad", "invad", "vitad", "unad", "cflow",
        "pyramidflow", "simplenet", "destseg", "realnet", "rdpp",
    ],
    "MuSc": [
        "musc_clip_b32_512", "musc_clip_b16_512", "musc_clip_l14_336", "musc_clip_l14_518",
        "musc_dinov2_b14_336", "musc_dinov2_b14_518", "musc_dinov2_l14_336", "musc_dinov2_l14_518",
    ],
    "SubspaceAD": [
        "subspacead_dinov2_large_672", "subspacead_dinov2_large_518", "subspacead_dinov2_large_336",
        "subspacead_dinov2_base_672", "subspacead_dinov2_base_518", "subspacead_dinov2_small_672",
    ],
}


def main():
    print("=" * 70)
    print("算法可用性验证 (推理)")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"测试图片: {TEST_IMG}")
    print("=" * 70)

    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, 显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")
    else:
        print("WARNING: CUDA not available!")

    all_results = {}
    total = sum(len(v) for v in INFERENCE_ALGORITHMS.values())

    i = 0
    for family, algorithms in INFERENCE_ALGORITHMS.items():
        print(f"\n{'='*60}")
        print(f"📍 {family} ({len(algorithms)} 算法)")
        print(f"{'='*60}")

        for alg in algorithms:
            i += 1
            print(f"  [{i}/{total}] {alg}...", end=" ", flush=True)
            r = test_inference(alg)
            all_results[alg] = r
            status = "✅" if r["success"] else "❌"
            score = r.get("score", "N/A")
            err = r.get("error", "")[:100]
            print(f"{status} score={score} time={r['time_s']}s {err}")

            # 增量保存
            passed = sum(1 for v in all_results.values() if v["success"])
            failed = sum(1 for v in all_results.values() if not v["success"])
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": {"total": len(all_results), "passed": passed, "failed": failed, "tested": i},
                "results": all_results,
            }
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            with open(REPORT_PATH, "w") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

    # 总结
    passed = sum(1 for v in all_results.values() if v["success"])
    failed = sum(1 for v in all_results.values() if not v["success"])
    print(f"\n{'='*70}")
    print(f"推理验证完成: {passed} 通过, {failed} 失败 (共 {len(all_results)})")

    failures = {k: v for k, v in all_results.items() if not v["success"]}
    if failures:
        print(f"\n失败列表:")
        for alg, r in failures.items():
            print(f"  ❌ {alg}: {r.get('error', 'unknown')[:150]}")

    print(f"\n报告: {REPORT_PATH}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
