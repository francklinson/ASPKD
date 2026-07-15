#!/usr/bin/env python3
"""
端到端测试脚本 — 验证前端入口的算法是否都可以训练和推理
通过 HTTP API 调用后端接口，模拟前端操作流程
"""
import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime

BASE_URL = os.environ.get("API_URL", "http://localhost:8004")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 测试用的 MVTec 图片路径
TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "public_dataset" / "mvtec" / "bottle" / "test" / "broken_large"


def get_test_images(count=3):
    """获取测试图片的绝对路径列表"""
    paths = []
    search_dirs = [
        TEST_IMAGE_DIR,
        PROJECT_ROOT / "data" / "public_dataset" / "mvtec" / "bottle" / "test" / "broken_large",
        PROJECT_ROOT / "data" / "public_dataset" / "mvtec" / "bottle" / "test",
        PROJECT_ROOT / "data" / "public_dataset" / "mvtec" / "bottle" / "train" / "good",
    ]
    for d in search_dirs:
        if d.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                for f in sorted(d.rglob(ext)):
                    paths.append(str(f))
                    if len(paths) >= count:
                        return paths
    return paths


def get_families():
    """获取所有算法族"""
    resp = requests.get(f"{BASE_URL}/api/training/families")
    resp.raise_for_status()
    return resp.json()


def get_trainable_algorithms(families):
    """提取所有可训练算法"""
    result = []
    for family_key, family in families.items():
        if not family.get("trainable", False):
            continue
        for algo in family.get("algorithms", []):
            algo_id = algo.get("id", "") if isinstance(algo, dict) else algo
            trainable = algo.get("trainable", True) if isinstance(algo, dict) else True
            available = algo.get("available", True) if isinstance(algo, dict) else True
            if trainable and available:
                result.append((family_key, algo_id, algo.get("name", algo_id) if isinstance(algo, dict) else algo_id))
    return result


def get_inference_algorithms(families):
    """提取所有可推理算法（含不可训练但可推理的）"""
    result = []
    for family_key, family in families.items():
        for algo in family.get("algorithms", []):
            if isinstance(algo, dict):
                algo_id = algo["id"]
                available = algo.get("available", True)
                if available:
                    result.append((family_key, algo_id, algo.get("name", algo_id)))
    return result


def start_training(family, algo_name, categories, gpu_id=0, data_source="mvtec", max_epochs=2):
    """启动训练任务 — 使用正确的 TrainingRequest 格式"""
    payload = {
        "algorithm_family": family,
        "algorithm_name": algo_name,
        "categories": categories,
        "data_source": data_source,
        "gpu_id": gpu_id,
        "total_iters": max_epochs * 100,
    }
    resp = requests.post(f"{BASE_URL}/api/training/start", json=payload)
    return resp


def wait_for_training(task_id, timeout=600, poll_interval=5):
    """等待训练完成"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        resp = requests.get(f"{BASE_URL}/api/training/status/{task_id}")
        if resp.status_code != 200:
            time.sleep(poll_interval)
            continue
        data = resp.json()
        status = data.get("status", "unknown")
        progress = data.get("progress", "")
        metrics = data.get("metrics", {})
        loss = metrics.get("loss_history", [])[-1] if metrics.get("loss_history") else "N/A"

        if status in ("completed", "failed", "stopped"):
            return data

        elapsed = int(time.time() - start_time)
        print(f"    [{elapsed}s] 进度: {progress}, 最新loss: {loss}    ", end="\r")
        time.sleep(poll_interval)

    return {"status": "timeout", "task_id": task_id}


def run_inference_from_dataset(algorithm, image_paths, model_path=None, threshold=0.5):
    """通过 from-dataset 接口执行推理"""
    payload = {
        "dataset": "bottle",
        "image_paths": image_paths,
        "algorithm": algorithm,
        "threshold": threshold,
    }
    if model_path:
        payload["model_path"] = model_path

    resp = requests.post(f"{BASE_URL}/api/custom-detection/from-dataset", json=payload)
    return resp


def wait_for_inference(task_id, timeout=300, poll_interval=3):
    """等待推理完成"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        resp = requests.get(f"{BASE_URL}/api/custom-detection/result/{task_id}")
        if resp.status_code != 200:
            time.sleep(poll_interval)
            continue
        data = resp.json()
        status = data.get("status", "unknown")
        if status in ("completed", "failed", "stopped"):
            return data
        time.sleep(poll_interval)
    return {"status": "timeout", "task_id": task_id}


def get_trained_models():
    """获取已训练模型列表"""
    resp = requests.get(f"{BASE_URL}/api/training/models")
    if resp.status_code == 200:
        return resp.json()
    return []


class TestResults:
    def __init__(self):
        self.results = []

    def add(self, family, algo_id, algo_name, test_type, success, detail=""):
        self.results.append({
            "family": family,
            "algo_id": algo_id,
            "algo_name": algo_name,
            "test_type": test_type,
            "success": success,
            "detail": detail,
        })

    def summary(self):
        total = len(self.results)
        ok = sum(1 for r in self.results if r["success"])
        fail = total - ok
        lines = [
            "=" * 70,
            f"端到端测试结果汇总 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
            "=" * 70,
            f"总计: {total} 项, 通过: {ok}, 失败: {fail}",
            "",
        ]

        by_family = {}
        for r in self.results:
            by_family.setdefault(r["family"], []).append(r)

        for family, items in by_family.items():
            lines.append(f"── {family} ──")
            for r in items:
                icon = "PASS" if r["success"] else "FAIL"
                lines.append(f"  [{icon}] {r['algo_name']} | {r['test_type']}")
                if not r["success"] and r["detail"]:
                    lines.append(f"       原因: {r['detail'][:200]}")
            lines.append("")

        failures = [r for r in self.results if not r["success"]]
        if failures:
            lines.append("── 失败项汇总 ──")
            for r in failures:
                lines.append(f"  {r['family']}/{r['algo_id']} [{r['test_type']}]: {r['detail'][:150]}")
            lines.append("")

        return "\n".join(lines)


def main():
    results = TestResults()

    print("=" * 70)
    print("端到端测试开始")
    print("=" * 70)

    # ── 0. 准备测试图片 ──
    print("\n[准备] 获取测试图片...")
    test_images = get_test_images(3)
    if not test_images:
        print("  FAIL: 找不到测试图片，跳过推理测试")
    else:
        print(f"  OK: 找到 {len(test_images)} 张测试图片")
        for img in test_images:
            print(f"    - {img}")

    # ── 1. 获取算法信息 ──
    print("\n[1/4] 获取算法族信息...")
    families = get_families()
    trainable = get_trainable_algorithms(families)
    all_inference = get_inference_algorithms(families)
    print(f"  可训练算法: {len(trainable)} 个")
    print(f"  可推理算法: {len(all_inference)} 个")

    # ── 2. 测试推理（预训练模型）──
    print("\n[2/4] 测试推理（预训练模型）...")
    if test_images:
        # 每族选1-2个代表测试推理
        test_set = []
        seen = set()
        for family, algo_id, name in trainable:
            if family not in seen:
                seen.add(family)
                test_set.append((family, algo_id, name))
        # 补充 Anomalib 不可训练但可推理的算法
        for family, algo_id, name in all_inference:
            if family == "anomalib" and (family, algo_id) not in [(t[0], t[1]) for t in test_set]:
                if len(test_set) < 12:
                    test_set.append((family, algo_id, name))

        for family, algo_id, name in test_set:
            print(f"  推理: {name} ({algo_id})...", end=" ", flush=True)
            try:
                resp = run_inference_from_dataset(algo_id, test_images[:1])
                if resp.status_code == 200:
                    data = resp.json()
                    task_id = data.get("task_id")
                    if task_id:
                        result = wait_for_inference(task_id, timeout=120)
                        status = result.get("status", "unknown")
                        if status == "completed":
                            det_results = result.get("results", [])
                            score = det_results[0].get("anomaly_score", "N/A") if det_results else "N/A"
                            print(f"PASS (score={score})")
                            results.add(family, algo_id, name, "推理(预训练)", True, f"score={score}")
                        else:
                            error = result.get("error", result.get("message", status))
                            print(f"FAIL ({status}: {str(error)[:80]})")
                            results.add(family, algo_id, name, "推理(预训练)", False, f"{status}: {error}")
                    else:
                        print(f"FAIL (无task_id)")
                        results.add(family, algo_id, name, "推理(预训练)", False, "无task_id")
                else:
                    try:
                        detail = resp.json().get("detail", resp.text[:200])
                    except Exception:
                        detail = resp.text[:200]
                    print(f"FAIL ({resp.status_code}: {str(detail)[:80]})")
                    results.add(family, algo_id, name, "推理(预训练)", False, str(detail)[:200])
            except Exception as e:
                print(f"FAIL ({e})")
                results.add(family, algo_id, name, "推理(预训练)", False, str(e)[:200])
            time.sleep(1)  # 避免过快请求
    else:
        print("  跳过（无测试图片）")

    # ── 3. 测试训练（每族1个代表，快速训练） ──
    print("\n[3/4] 测试训练（每族选1个代表算法，快速训练）...")
    train_tests = []
    seen_families = set()
    for family, algo_id, name in trainable:
        if family not in seen_families:
            seen_families.add(family)
            train_tests.append((family, algo_id, name))

    test_category = "bottle"

    for family, algo_id, name in train_tests:
        print(f"\n  训练: {name} ({algo_id}) [family={family}]")
        try:
            resp = start_training(family, algo_id, [test_category], max_epochs=2)
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text[:200])
                except Exception:
                    detail = resp.text[:200]
                print(f"    FAIL: 启动失败 - {detail}")
                results.add(family, algo_id, name, "训练", False, f"启动失败: {detail}")
                continue

            task_data = resp.json()
            task_id = task_data.get("task_id")
            if not task_id:
                print(f"    FAIL: 无task_id")
                results.add(family, algo_id, name, "训练", False, "无task_id")
                continue

            print(f"    task_id={task_id}, 等待完成...")
            result = wait_for_training(task_id, timeout=600)
            status = result.get("status", "unknown")
            progress = result.get("progress", "")

            if status == "completed":
                model_path = result.get("model_path", "")
                metrics = result.get("metrics", {})
                best_loss = metrics.get("best_loss", "N/A")
                has_loss = len(metrics.get("loss_history", [])) > 0
                print(f"    PASS: 训练完成! model={model_path}, best_loss={best_loss}, 有loss曲线={has_loss}")
                results.add(family, algo_id, name, "训练", True, f"model={model_path}, loss={best_loss}, loss曲线={has_loss}")
            else:
                error = result.get("error", progress)
                print(f"    FAIL: 训练{status}: {error}")
                results.add(family, algo_id, name, "训练", False, f"{status}: {error}")

        except Exception as e:
            print(f"    FAIL: 异常 - {e}")
            results.add(family, algo_id, name, "训练", False, str(e)[:200])

    # ── 4. 训练后推理测试 ──
    print("\n[4/4] 测试训练后推理...")
    if test_images:
        trained_models = get_trained_models()
        print(f"  已训练模型: {len(trained_models)} 个")
        tested = 0
        for model_info in trained_models[:8]:
            model_name = model_info.get("name", "")
            model_path = model_info.get("path", "")
            algo_name = model_info.get("algorithm_name", "")
            if not algo_name:
                continue
            print(f"  推理: {model_name} (algo={algo_name})...", end=" ", flush=True)
            try:
                resp = run_inference_from_dataset(algo_name, test_images[:1], model_path=model_path)
                if resp.status_code == 200:
                    data = resp.json()
                    task_id = data.get("task_id")
                    if task_id:
                        result = wait_for_inference(task_id, timeout=120)
                        status = result.get("status", "unknown")
                        if status == "completed":
                            det_results = result.get("results", [])
                            score = det_results[0].get("anomaly_score", "N/A") if det_results else "N/A"
                            print(f"PASS (score={score})")
                            results.add("trained", algo_name, model_name, "推理(训练后)", True, f"score={score}")
                        else:
                            error = result.get("error", result.get("message", status))
                            print(f"FAIL ({status}: {str(error)[:80]})")
                            results.add("trained", algo_name, model_name, "推理(训练后)", False, f"{status}: {error}")
                    else:
                        print(f"FAIL (无task_id)")
                        results.add("trained", algo_name, model_name, "推理(训练后)", False, "无task_id")
                else:
                    try:
                        detail = resp.json().get("detail", resp.text[:200])
                    except Exception:
                        detail = resp.text[:200]
                    print(f"FAIL ({resp.status_code}: {str(detail)[:80]})")
                    results.add("trained", algo_name, model_name, "推理(训练后)", False, str(detail)[:200])
                tested += 1
            except Exception as e:
                print(f"FAIL ({e})")
                results.add("trained", algo_name, model_name, "推理(训练后)", False, str(e)[:200])
            time.sleep(1)
        if tested == 0:
            print("  跳过（无有算法ID的已训练模型）")
    else:
        print("  跳过（无测试图片）")

    # ── 汇总 ──
    summary = results.summary()
    print("\n" + summary)

    report_path = PROJECT_ROOT / "records" / "e2e_test_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 端到端测试报告\n\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("```\n" + summary + "\n```\n")
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
