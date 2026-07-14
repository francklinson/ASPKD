#!/usr/bin/env python3
"""续测 ADer / MuSc / SubspaceAD 推理"""
import os, sys, time, json, gc, traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "algorithms"))
sys.path.insert(0, str(PROJECT_ROOT / "algorithms" / "Dinomaly2"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

TEST_IMG = str(PROJECT_ROOT / "data" / "public_dataset" / "mvtec" / "bottle" / "test" / "good" / "000.png")
REPORT_PATH = PROJECT_ROOT / "records" / "verify_now_report.json"


def clear_gpu():
    try:
        import torch
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass


def test_one(alg_name):
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
        result["score"] = round(float(getattr(pred, "anomaly_score", -1)), 6)
        result["has_heatmap"] = getattr(pred, "anomaly_map", None) is not None
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        if detector:
            try:
                detector.release()
            except:
                pass
        clear_gpu()
    result["time_s"] = round(time.time() - t0, 2)
    return result


# Load existing results
existing = {}
if REPORT_PATH.exists():
    with open(REPORT_PATH) as f:
        existing = json.load(f).get("results", {})

remaining = {
    "ADer": ["invad", "vitad", "unad", "cflow", "pyramidflow", "simplenet", "destseg", "realnet", "rdpp"],
    "MuSc": ["musc_clip_b32_512", "musc_clip_b16_512", "musc_clip_l14_336", "musc_clip_l14_518",
             "musc_dinov2_b14_336", "musc_dinov2_b14_518", "musc_dinov2_l14_336", "musc_dinov2_l14_518"],
    "SubspaceAD": ["subspacead_dinov2_large_672", "subspacead_dinov2_large_518", "subspacead_dinov2_large_336",
                   "subspacead_dinov2_base_672", "subspacead_dinov2_base_518", "subspacead_dinov2_small_672"],
}

# Skip already tested (mambaad was already tested)
already_tested = set(existing.keys())
print(f"Already tested: {len(already_tested)} algorithms")

for family, algorithms in remaining.items():
    print(f"\n=== {family} ===")
    for alg in algorithms:
        if alg in already_tested:
            print(f"  SKIP {alg} (already tested)")
            continue
        print(f"  {alg}...", end=" ", flush=True)
        r = test_one(alg)
        existing[alg] = r
        status = "OK" if r["success"] else "FAIL"
        score = r.get("score", "N/A")
        err = r.get("error", "")[:100]
        print(f"{status} score={score} time={r['time_s']}s {err}")

        # Save incrementally
        passed = sum(1 for v in existing.values() if v["success"])
        failed = sum(1 for v in existing.values() if not v["success"])
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": {"total": len(existing), "passed": passed, "failed": failed},
            "results": existing,
        }
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

passed = sum(1 for v in existing.values() if v["success"])
failed = sum(1 for v in existing.values() if not v["success"])
print(f"\nDone! Total: {len(existing)}, Passed: {passed}, Failed: {failed}")
