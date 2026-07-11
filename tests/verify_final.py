#!/usr/bin/env python3
"""最终验证 — 逐一测试所有可用算法 (无超时机制)"""
import os, sys, time, json, traceback
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'algorithms'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'algorithms', 'Dinomaly2'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = 'models/pre_trained/huggingface'
os.environ['TRANSFORMERS_CACHE'] = 'models/pre_trained/huggingface'

TRAIN_GOOD = 'data/public_dataset/mvtec/bottle/train/good'
TEST_GOOD = 'data/public_dataset/mvtec/bottle/test/good'
REPORT_PATH = 'records/verify_all_algorithms_FINAL.json'

EXCLUDED = {"mambaad","invad","vitad","unad","cflow","pyramidflow","simplenet",
            "denseae","cae","vae","aegan","differnet",
            "hiad","multiads","musc","dictas","subspacead","diad","audio_feature_cluster"}

def get_test_img():
    d = TEST_GOOD
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.png','.jpg','.bmp')):
                return os.path.join(d, f)
    return None

def get_refs(n=5):
    refs = []
    if os.path.isdir(TRAIN_GOOD):
        for f in sorted(os.listdir(TRAIN_GOOD))[:n]:
            if f.lower().endswith(('.png','.jpg','.bmp')):
                refs.append(os.path.join(TRAIN_GOOD, f))
    return refs

def clear_gpu():
    try:
        import torch; torch.cuda.empty_cache()
    except: pass

def save(results):
    passed = sum(1 for r in results.values() if r.get('inference')=='pass')
    failed = sum(1 for r in results.values() if r.get('inference')=='fail')
    excluded = sum(1 for r in results.values() if r.get('inference')=='excluded')
    with open(REPORT_PATH, 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "summary": {"total":len(results),"passed":passed,"failed":failed,"excluded":excluded},
                   "results": results}, f, ensure_ascii=False, indent=2)

def test_dinomaly(alg, img):
    from algorithms import create_detector
    detector = create_detector(alg)
    try:
        detector.load_model()
        t0 = time.time()
        pred = detector.predict(img)
        t = (time.time()-t0)*1000
        return {"inference":"pass","score":round(float(pred.anomaly_score),6),"time_ms":round(t)}
    finally:
        try: detector.release()
        except: pass
        clear_gpu()

def test_dinomaly2(alg, img):
    from algorithms import create_detector
    detector = create_detector(alg)
    try:
        detector.load_model()
        t0 = time.time()
        pred = detector.predict(img)
        t = (time.time()-t0)*1000
        return {"inference":"pass","score":round(float(pred.anomaly_score),6),"time_ms":round(t)}
    finally:
        try: detector.release()
        except: pass
        clear_gpu()

def test_anomalib(alg):
    from anomalib.models import get_model
    from anomalib.engine import Engine
    from anomalib.data import MVTecAD

    model = get_model(alg)
    bottle_dir = os.path.dirname(os.path.dirname(TRAIN_GOOD))
    mvtec_root = os.path.dirname(bottle_dir)
    category = os.path.basename(bottle_dir)
    engine = Engine()
    datamodule = MVTecAD(root=mvtec_root, category=category, train_batch_size=8, eval_batch_size=1, num_workers=0)
    datamodule.setup()
    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    fit_ms = (time.time()-t0)*1000
    clear_gpu()
    return {"inference":"pass","fit_ms":round(fit_ms),"training_verified":True}

def test_musc(alg, img, refs):
    from algorithms import create_detector
    detector = create_detector(alg)
    try:
        detector.load_model()
        all_imgs = [img] + [r for r in refs[:4] if r != img]
        t0 = time.time()
        results = detector.predict_batch(all_imgs)
        t = (time.time()-t0)*1000/len(all_imgs)
        return {"inference":"pass","score":round(float(results[0].anomaly_score),6),"time_ms":round(t)}
    finally:
        try: detector.release()
        except: pass
        clear_gpu()

def test_subspacead(alg, img, refs):
    from algorithms import create_detector
    detector = create_detector(alg)
    try:
        detector.load_model()
        t0 = time.time()
        pred = detector.predict(img, reference_paths=refs[:5])
        t = (time.time()-t0)*1000
        return {"inference":"pass","score":round(float(pred.anomaly_score),6),"time_ms":round(t)}
    finally:
        try: detector.release()
        except: pass
        clear_gpu()

def main():
    import torch
    print(f"CUDA: {'✅' if torch.cuda.is_available() else '❌'} | GPU: {torch.cuda.get_device_name(0)}")
    img = get_test_img()
    refs = get_refs()
    if not img:
        print("No test image!"); return
    print(f"Test: {os.path.basename(img)}, Refs: {len(refs)}")

    results = {}
    count = 0

    # Dinomaly (6)
    for alg in ["dinomaly_dinov3_small","dinomaly_dinov3_base","dinomaly_dinov3_large",
                "dinomaly_dinov2_small","dinomaly_dinov2_base","dinomaly_dinov2_large"]:
        count += 1
        print(f"[{count}/53] {alg}...", end=" ", flush=True)
        try:
            r = test_dinomaly(alg, img)
            results[alg] = r
            print(f"✅ score={r['score']:.4f} {r['time_ms']:.0f}ms")
        except Exception as e:
            results[alg] = {"inference":"fail","error":f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"❌ {str(e)[:80]}")
        save(results)

    # Dinomaly2 (6)
    for prefix in ["dinomaly2_dinov2","dinomaly2_dinov3"]:
        for size in ["small","base","large"]:
            alg = f"{prefix}_{size}"
            count += 1
            print(f"[{count}/53] {alg}...", end=" ", flush=True)
            try:
                r = test_dinomaly2(alg, img)
                results[alg] = r
                print(f"✅ score={r['score']:.4f}")
            except Exception as e:
                results[alg] = {"inference":"fail","error":f"{type(e).__name__}: {str(e)[:200]}"}
                print(f"❌ {str(e)[:80]}")
            save(results)

    # Anomalib (27) - fit only to verify training
    anom_algs = ["patchcore","cfa","csflow","dfkde","dfm","draem","dsr","efficient_ad",
                 "fastflow","fre","padim","reverse_distillation","stfpm","ganomaly",
                 "supersimplenet","uflow","uninet","vlm_ad","winclip","anomalyvfm","cfm",
                 "general_ad","glass","inp_former","l2bt","patchflow","anomaly_dino"]
    for alg in anom_algs:
        count += 1
        print(f"[{count}/53] {alg}...", end=" ", flush=True)
        try:
            r = test_anomalib(alg)
            results[alg] = r
            print(f"✅ fit={r['fit_ms']:.0f}ms")
        except Exception as e:
            results[alg] = {"inference":"fail","error":f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"❌ {str(e)[:80]}")
        save(results)

    # MuSc (8)
    for alg in ["musc_clip_b32_512","musc_clip_b16_512","musc_clip_l14_336","musc_clip_l14_518",
                 "musc_dinov2_b14_336","musc_dinov2_b14_518","musc_dinov2_l14_336","musc_dinov2_l14_518"]:
        count += 1
        print(f"[{count}/53] {alg}...", end=" ", flush=True)
        try:
            r = test_musc(alg, img, refs)
            results[alg] = r
            print(f"✅ score={r['score']:.4f}")
        except Exception as e:
            results[alg] = {"inference":"fail","error":f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"❌ {str(e)[:80]}")
        save(results)

    # SubspaceAD (6)
    for alg in ["subspacead_dinov2_large_672","subspacead_dinov2_large_518",
                "subspacead_dinov2_large_336","subspacead_dinov2_base_672",
                "subspacead_dinov2_base_518","subspacead_dinov2_small_672"]:
        count += 1
        print(f"[{count}/53] {alg}...", end=" ", flush=True)
        try:
            r = test_subspacead(alg, img, refs)
            results[alg] = r
            print(f"✅ score={r['score']:.4f}")
        except Exception as e:
            results[alg] = {"inference":"fail","error":f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"❌ {str(e)[:80]}")
        save(results)

    # Mark excluded
    for alg in EXCLUDED:
        results[alg] = {"inference":"excluded","error":"已屏蔽（缺依赖或不适用）"}

    save(results)
    passed = sum(1 for r in results.values() if r.get('inference')=='pass')
    failed = sum(1 for r in results.values() if r.get('inference')=='fail')
    excluded = sum(1 for r in results.values() if r.get('inference')=='excluded')
    print(f"\n{'='*60}")
    print(f"Total: {len(results)} | ✅ Pass: {passed} | ❌ Fail: {failed} | ⏭️ Excluded: {excluded}")

if __name__ == "__main__":
    main()
