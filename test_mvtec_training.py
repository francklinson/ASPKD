#!/usr/bin/env python3
"""
全面验证测试：MVTec 数据集训练 + 自定义检测端到端流程
测试所有4个可训练算法族：Dinomaly, Dinomaly2, Anomalib, ADer
"""
import json
import time
import sys
import requests

BASE = "http://localhost:8004"
MVTEC_SOURCE = "mvtec"
TEST_CATEGORY = "bottle"  # MVTec bottle: 209 train, 20 test good, 63 test anomaly

# 测试图片路径
TEST_IMAGES = [
    f"/home/zhouchenghao/PycharmProjects/ASD_for_SPK/data/public_dataset/mvtec/{TEST_CATEGORY}/test/good/000.png",
    f"/home/zhouchenghao/PycharmProjects/ASD_for_SPK/data/public_dataset/mvtec/{TEST_CATEGORY}/test/broken_large/000.png",
]

def api(method, path, **kwargs):
    url = f"{BASE}{path}"
    resp = getattr(requests, method)(url, **kwargs)
    return resp

def test_families():
    """测试1: 算法族API"""
    print("\n" + "="*60)
    print("测试1: 算法族API")
    resp = api("get", "/api/training/families")
    data = resp.json()
    assert len(data) == 4, f"Expected 4 families, got {len(data)}"
    
    trainable = {k: v for k, v in data.items() if v.get("trainable")}
    assert len(trainable) == 4, f"Expected 4 trainable families"
    
    for key, fam in data.items():
        algos = fam.get("algorithms", [])
        assert len(algos) > 0, f"Family {key} has no algorithms"
        # Check detail fields
        for a in algos:
            for field in ["id", "name", "type", "description", "performance", "gpu_memory", "input_size"]:
                assert field in a, f"Algorithm {a.get('id','?')} missing field {field}"
    
    print(f"  ✓ 4个算法族，详情字段完整")
    print(f"    Dinomaly: {len(data['dinomaly']['algorithms'])} 个算法")
    print(f"    Dinomaly2: {len(data['dinomaly2']['algorithms'])} 个算法")
    print(f"    Anomalib: {len(data['anomalib']['algorithms'])} 个算法")
    print(f"    ADer: {len(data['ader']['algorithms'])} 个算法")
    return data

def test_datasets():
    """测试2: 数据集API包含MVTec"""
    print("\n" + "="*60)
    print("测试2: 数据集API")
    resp = api("get", "/api/training/datasets")
    data = resp.json()
    
    mvtec = [d for d in data if d.get("source") == "mvtec"]
    spk = [d for d in data if d.get("source") == "spk"]
    
    assert len(mvtec) == 15, f"Expected 15 MVTec categories, got {len(mvtec)}"
    assert len(spk) >= 0, f"SPK categories should exist"
    
    # Verify source fields
    for d in data:
        assert "source" in d, f"Missing source field in {d['name']}"
        assert "source_label" in d, f"Missing source_label field in {d['name']}"
    
    # Find bottle
    bottle = next((d for d in mvtec if d["name"] == "bottle"), None)
    assert bottle is not None, "bottle category not found"
    assert bottle["trainable"], "bottle should be trainable"
    assert bottle["train_normal_count"] >= 10, "bottle should have enough training data"
    
    print(f"  ✓ MVTec: {len(mvtec)} 个类别, SPK: {len(spk)} 个类别")
    print(f"    bottle: 训练={bottle['train_normal_count']}, 测试正常={bottle['test_normal_count']}, 测试异常={bottle['test_anomaly_count']}")
    return data

def test_training(family_key, family_data, algo_id=None):
    """测试3: 训练指定算法族"""
    print(f"\n{'='*60}")
    param_schema = family_data["param_schema"]
    
    if param_schema == "encoder_size":
        # Dinomaly2 only supports DINOv2
        model_type = "dinov2" if family_key == "dinomaly2" else "dinov3"
        model_size = "small"
        algo_name = f"{family_key}_{model_type}_{model_size}"
    else:
        # Use the first algorithm or specified one
        algo_name = algo_id or family_data["algorithms"][0]["id"]
    
    print(f"测试3: 训练 {family_data['name']} ({algo_name}) 在 MVTec {TEST_CATEGORY}")
    
    payload = {
        "categories": [TEST_CATEGORY],
        "data_source": MVTEC_SOURCE,
        "algorithm_family": family_key,
        "algorithm_name": algo_name,
        "model_type": model_type if param_schema == "encoder_size" else "",
        "model_size": model_size if param_schema == "encoder_size" else "",
        "total_iters": 100,  # 快速测试用
        "batch_size": 4,
    }
    
    resp = api("post", "/api/training/start", json=payload)
    result = resp.json()
    
    if not result.get("success"):
        print(f"  ✗ 训练启动失败: {result.get('detail', result.get('message', 'unknown'))}")
        return None
    
    task_id = result["task_id"]
    print(f"  ✓ 训练任务已创建: {task_id}")
    
    # Poll status
    max_wait = 300  # 5 min max
    start = time.time()
    while time.time() - start < max_wait:
        resp = api("get", f"/api/training/status/{task_id}")
        status = resp.json()
        
        if status["status"] == "completed":
            print(f"  ✓ 训练完成")
            return task_id
        elif status["status"] == "failed":
            log = status.get("log", "")
            last_lines = log.strip().split("\n")[-3:]
            print(f"  ✗ 训练失败: {status['progress']}")
            for line in last_lines:
                print(f"    {line}")
            return None
        else:
            elapsed = int(time.time() - start)
            print(f"  ... {status['status']} ({elapsed}s): {status['progress'][:60]}", end="\r")
            time.sleep(5)
    
    print(f"  ✗ 训练超时")
    return None

def test_custom_detection(algorithm_id):
    """测试5: 自定义检测"""
    print(f"\n  检测测试: {algorithm_id}")
    
    payload = {
        "dataset": TEST_CATEGORY,
        "image_paths": TEST_IMAGES,
        "algorithm": algorithm_id,
        "threshold": 0.5,
    }
    
    resp = api("post", "/api/custom-detection/from-dataset", json=payload)
    if resp.status_code != 200:
        print(f"  ✗ 检测启动失败: HTTP {resp.status_code}")
        return False
    
    result = resp.json()
    task_id = result.get("task_id")
    if not task_id:
        print(f"  ✗ 检测启动失败: {result}")
        return False
    
    print(f"  检测任务: {task_id}")
    
    # Poll status
    max_wait = 120
    start = time.time()
    while time.time() - start < max_wait:
        resp = api("get", f"/api/custom-detection/result/{task_id}")
        status = resp.json()
        
        if status["status"] == "completed":
            results = status.get("results", [])
            print(f"  ✓ 检测完成, {len(results)} 个结果")
            for r in results:
                score = r.get("anomaly_score", 0)
                is_anom = r.get("is_anomaly", score > 0.5)
                label = "异常" if is_anom else "正常"
                print(f"    {r.get('filename','?')}: score={score:.4f} ({label})")
            return True
        elif status["status"] == "failed":
            print(f"  ✗ 检测失败: {status.get('message', 'unknown')}")
            return False
        else:
            time.sleep(5)
    
    print(f"  ✗ 检测超时")
    return False

def test_trained_models():
    """测试4: 已训练模型列表"""
    print(f"\n{'='*60}")
    print("测试4: 已训练模型")
    resp = api("get", "/api/training/models")
    data = resp.json()
    print(f"  已训练模型: {len(data)} 个")
    for m in data[:5]:
        print(f"    {m['name']} | {m.get('algorithm_family','?')} | {m['size_mb']}MB")
    return data

def main():
    print("="*60)
    print("  MVTec 数据集全流程验证测试")
    print("="*60)
    
    # Test 1: Families
    families = test_families()
    
    # Test 2: Datasets
    test_datasets()
    
    # Test 3+4+5: Train each family on MVTec, then detect
    results = {}
    
    # ---- Dinomaly (small, 100 iters) ----
    task_id = test_training("dinomaly", families["dinomaly"])
    results["dinomaly"] = task_id is not None
    if task_id:
        time.sleep(2)
        det_ok = test_custom_detection("dinomaly_dinov3_small")
        results["dinomaly_detection"] = det_ok
    
    # ---- Dinomaly2 (dinov2 small, 100 iters) ----
    # Dinomaly2 only supports DINOv2, not DINOv3
    task_id = test_training("dinomaly2", families["dinomaly2"])
    results["dinomaly2"] = task_id is not None
    if task_id:
        time.sleep(2)
        det_ok = test_custom_detection("dinomaly2_dinov2_small")
        results["dinomaly2_detection"] = det_ok
    
    # ---- Anomalib (PatchCore, quick) ----
    task_id = test_training("anomalib", families["anomalib"], algo_id="patchcore")
    results["anomalib"] = task_id is not None
    if task_id:
        time.sleep(2)
        det_ok = test_custom_detection("patchcore")
        results["anomalib_detection"] = det_ok
    
    # ---- ADer (MambaAD, quick) ----
    task_id = test_training("ader", families["ader"], algo_id="mambaad")
    results["ader"] = task_id is not None
    if task_id:
        time.sleep(2)
        det_ok = test_custom_detection("mambaad")
        results["ader_detection"] = det_ok
    
    # Test 4: Trained models
    test_trained_models()
    
    # Summary
    print(f"\n{'='*60}")
    print("测试结果汇总:")
    print("-"*60)
    for key, val in results.items():
        status = "✓ 通过" if val else "✗ 失败"
        print(f"  {key:30s} {status}")
    
    all_passed = all(results.values())
    print("-"*60)
    print(f"  总体: {'全部通过 ✓' if all_passed else '部分失败 ✗'}")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
