#!/usr/bin/env python3
"""
算法可用性验证脚本

验证标记为"训练+推理可用"的算法是否真的可以进行训练和推理。
针对 Dinomaly / Dinomaly2 / Anomalib / ADer 四个算法族。

用法:
    python scripts/verify_algorithm_availability.py [--quick] [--family FAMILY] [--skip-training] [--skip-inference]

参数:
    --quick          快速模式: 每族只测试1个代表性算法
    --family         只测试指定族 (dinomaly/dinomaly2/anomalib/ader)
    --skip-training  跳过训练测试
    --skip-inference 跳过推理测试
"""

import os
import sys
import time
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "algorithms"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


# ============================================================================
# 测试数据准备
# ============================================================================

TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "spk"
TEST_MVTEC_DIR = PROJECT_ROOT / "data" / "public_dataset" / "mvtec"


def get_test_image():
    """获取一张测试图片路径"""
    # 优先使用 SPK 数据
    for category in sorted(TEST_IMAGE_DIR.iterdir()):
        test_good = category / "test" / "good"
        if test_good.exists():
            for f in sorted(test_good.iterdir()):
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                    return str(f)
    # 回退到 MVTec
    if TEST_MVTEC_DIR.exists():
        for category in sorted(TEST_MVTEC_DIR.iterdir()):
            test_good = category / "test" / "good"
            if test_good.exists():
                for f in sorted(test_good.iterdir()):
                    if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                        return str(f)
    return None


def get_test_category():
    """获取一个可用的测试类别名（优先用有足够训练样本的数据集）"""
    # 优先用 MVTec（训练样本充足）
    if TEST_MVTEC_DIR.exists():
        for d in sorted(TEST_MVTEC_DIR.iterdir()):
            if d.is_dir() and (d / "train" / "good").exists():
                count = len(list((d / "train" / "good").iterdir()))
                if count >= 10:
                    return d.name
    if TEST_IMAGE_DIR.exists():
        for d in sorted(TEST_IMAGE_DIR.iterdir()):
            if d.is_dir() and (d / "train" / "good").exists():
                count = len(list((d / "train" / "good").iterdir()))
                if count >= 10:
                    return d.name
    return None


def get_data_source():
    """获取数据源名称（优先用有足够训练样本的数据集）"""
    test_cat = get_test_category()
    if test_cat:
        if TEST_MVTEC_DIR.exists() and (TEST_MVTEC_DIR / test_cat).exists():
            return "mvtec"
    if TEST_IMAGE_DIR.exists() and any(TEST_IMAGE_DIR.iterdir()):
        return "spk"
    return None


# ============================================================================
# 算法族定义
# ============================================================================

ALGORITHM_FAMILIES = {
    "dinomaly": {
        "algorithms": [
            "dinomaly_dinov3_small",
            "dinomaly_dinov3_base",
            "dinomaly_dinov3_large",
            "dinomaly_dinov2_small",
            "dinomaly_dinov2_base",
            "dinomaly_dinov2_large",
        ],
        "quick_pick": "dinomaly_dinov3_small",
    },
    "dinomaly2": {
        "algorithms": [
            "dinomaly2_dinov2_small",
            "dinomaly2_dinov2_base",
            "dinomaly2_dinov2_large",
            "dinomaly2_dinov3_small",
            "dinomaly2_dinov3_base",
            "dinomaly2_dinov3_large",
        ],
        "quick_pick": "dinomaly2_dinov2_small",
    },
    "anomalib": {
        "algorithms": [
            "patchcore", "cfa", "csflow", "dfkde", "dfm", "draem", "dsr",
            "fre", "reverse_distillation", "ganomaly", "supersimplenet",
            "uninet", "winclip",
            "anomalyvfm", "general_ad", "glass", "inp_former",
            "l2bt", "patchflow", "anomaly_dino",
        ],
        "quick_pick": "patchcore",
        # 不可训练的 anomalib 算法 (不在此列表中)
        "non_trainable": ["padim", "dfkde", "efficient_ad", "fastflow", "stfpm", "uflow", "cfm", "vlm_ad"],
    },
    "ader": {
        "algorithms": [
            "mambaad", "invad", "vitad", "unad", "cflow",
            "pyramidflow", "simplenet", "destseg", "realnet", "rdpp",
        ],
        "quick_pick": "simplenet",
    },
}


# ============================================================================
# 推理测试
# ============================================================================

def test_inference(algorithm_name: str, test_image: str) -> dict:
    """测试算法推理是否可用"""
    result = {
        "algorithm": algorithm_name,
        "test": "inference",
        "success": False,
        "error": None,
        "duration_s": 0,
        "details": {},
    }

    t0 = time.time()
    try:
        from algorithms import create_detector

        # 创建检测器
        detector = create_detector(algorithm_name)
        result["details"]["detector_created"] = True

        # 加载模型
        detector.load_model()
        result["details"]["model_loaded"] = True

        # 推理
        det_result = detector.predict(test_image)
        result["details"]["prediction_made"] = True
        result["details"]["score"] = getattr(det_result, 'anomaly_score', None)
        result["details"]["has_anomaly_map"] = getattr(det_result, 'anomaly_map', None) is not None
        result["success"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["details"]["traceback"] = traceback.format_exc()

    result["duration_s"] = round(time.time() - t0, 2)
    return result


# ============================================================================
# 训练测试
# ============================================================================

def test_training(algorithm_name: str, family: str, category: str, data_source: str) -> dict:
    """测试算法训练是否可用（通过 API 层调用）"""
    result = {
        "algorithm": algorithm_name,
        "family": family,
        "test": "training",
        "success": False,
        "error": None,
        "duration_s": 0,
        "details": {},
    }

    t0 = time.time()
    try:
        import asyncio
        # 直接调用后端训练逻辑，绕过 HTTP 层
        from backend.api.training import (
            TrainingRequest, TrainingConfig, start_training,
            TRAINING_TASKS, _save_training_tasks
        )

        # 解析 algorithm_name 到 family/algorithm_name/model_type/model_size
        model_type = "dinov3"
        model_size = "small"
        algo_name = algorithm_name

        if family in ("dinomaly", "dinomaly2"):
            if "dinov2" in algorithm_name:
                model_type = "dinov2"
            elif "dinov3" in algorithm_name:
                model_type = "dinov3"
            if "base" in algorithm_name:
                model_size = "base"
            elif "large" in algorithm_name:
                model_size = "large"
            algo_name = algorithm_name  # 保持完整名

        request = TrainingRequest(
            categories=[category],
            data_source=data_source,
            algorithm_family=family,
            algorithm_name=algo_name,
            model_type=model_type,
            model_size=model_size,
            total_iters=10,  # 最小迭代次数，仅验证可启动
            batch_size=4,
            gpu_id=0,
        )

        # start_training 是 async 函数，需要用 asyncio.run 调用
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 如果已在事件循环中（不应出现在脚本中），用新线程
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                resp = pool.submit(asyncio.run, start_training(request)).result()
        else:
            resp = asyncio.run(start_training(request))

        task_id = resp.get("task_id")
        result["details"]["task_id"] = task_id
        result["details"]["start_response"] = resp.get("message", "")

        if not resp.get("success"):
            result["error"] = f"训练启动失败: {resp.get('message', 'unknown')}"
            result["duration_s"] = round(time.time() - t0, 2)
            return result

        # 等待训练完成或超时
        timeout = 300  # 5分钟超时
        check_interval = 2
        elapsed = 0
        while elapsed < timeout:
            task = TRAINING_TASKS.get(task_id, {})
            status = task.get("status", "unknown")
            if status in ("completed", "failed"):
                result["details"]["final_status"] = status
                result["details"]["progress"] = task.get("progress", "")
                if status == "completed":
                    result["success"] = True
                else:
                    # 获取失败日志
                    log_snippet = (task.get("log") or "")[-500:]
                    result["error"] = f"训练失败: {task.get('progress', '')}\n日志末尾:\n{log_snippet}"
                break
            time.sleep(check_interval)
            elapsed += check_interval
        else:
            result["error"] = f"训练超时 ({timeout}s)"
            # 终止卡住的进程
            task = TRAINING_TASKS.get(task_id, {})
            proc = task.get("process")
            if proc and proc.poll() is None:
                proc.terminate()
            task["status"] = "failed"
            task["progress"] = "验证脚本超时终止"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["details"]["traceback"] = traceback.format_exc()

    result["duration_s"] = round(time.time() - t0, 2)
    return result


# ============================================================================
# 依赖检查
# ============================================================================

def check_dependencies(algorithm_name: str, family: str) -> dict:
    """检查算法的依赖是否满足"""
    result = {
        "algorithm": algorithm_name,
        "family": family,
        "test": "dependencies",
        "success": True,
        "missing": [],
        "details": {},
    }

    # 家族级依赖
    family_deps = {
        "dinomaly": ["torch", "timm"],
        "dinomaly2": ["torch", "timm"],
        "anomalib": ["torch"],
        "ader": ["torch", "timm", "fvcore", "tensorboardX"],
    }

    # 算法级额外依赖
    algo_deps = {
        "mambaad": ["mamba_ssm", "causal_conv1d"],
        "winclip": ["open_clip"],
    }

    import importlib.util
    deps_to_check = family_deps.get(family, [])
    if algorithm_name in algo_deps:
        deps_to_check.extend(algo_deps[algorithm_name])

    for dep in deps_to_check:
        spec = importlib.util.find_spec(dep)
        if spec is None:
            result["missing"].append(dep)
            result["success"] = False

    # Anomalib 特殊检查: 本地源码
    if family == "anomalib":
        anomalib_dir = PROJECT_ROOT / "algorithms" / "anomalib"
        if not (anomalib_dir / "__init__.py").exists():
            result["missing"].append("anomalib (本地源码)")
            result["success"] = False

    # ADer 特殊检查: meta.json
    if family == "ader":
        meta_path = PROJECT_ROOT / "data" / "spk" / "meta.json"
        if not meta_path.exists():
            result["details"]["meta_json_missing"] = True
            # 不算失败，因为会自动生成

    result["details"]["checked_deps"] = deps_to_check
    return result


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="验证算法可用性")
    parser.add_argument("--quick", action="store_true", help="快速模式：每族只测1个")
    parser.add_argument("--family", choices=["dinomaly", "dinomaly2", "anomalib", "ader"], help="只测试指定族")
    parser.add_argument("--skip-training", action="store_true", help="跳过训练测试")
    parser.add_argument("--skip-inference", action="store_true", help="跳过推理测试")
    parser.add_argument("--only-deps", action="store_true", help="仅检查依赖")
    args = parser.parse_args()

    print("=" * 70)
    print("算法可用性验证")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"项目根目录: {PROJECT_ROOT}")
    print("=" * 70)

    # 准备测试数据
    test_image = get_test_image()
    test_category = get_test_category()
    data_source = get_data_source()

    print(f"\n测试数据:")
    print(f"  数据源: {data_source}")
    print(f"  测试类别: {test_category}")
    print(f"  测试图片: {test_image}")

    if not test_image and not args.skip_inference:
        print("\n[错误] 没有找到测试图片，无法进行推理测试")
        args.skip_inference = True

    if not test_category and not args.skip_training:
        print("\n[错误] 没有找到测试类别，无法进行训练测试")
        args.skip_training = True

    # 选择要测试的算法
    families_to_test = {}
    if args.family:
        families_to_test[args.family] = ALGORITHM_FAMILIES[args.family]
    else:
        families_to_test = ALGORITHM_FAMILIES

    # 收集所有测试结果
    all_results = []
    summary = {
        "dependency_pass": 0, "dependency_fail": 0,
        "inference_pass": 0, "inference_fail": 0,
        "training_pass": 0, "training_fail": 0,
        "skipped": 0,
    }

    for family_name, family_info in families_to_test.items():
        algorithms = family_info["algorithms"]
        if args.quick:
            algorithms = [family_info["quick_pick"]]

        print(f"\n{'='*70}")
        print(f"算法族: {family_name} ({len(algorithms)} 个算法)")
        print(f"{'='*70}")

        for alg_name in algorithms:
            print(f"\n--- {alg_name} ---")

            # 1. 依赖检查
            dep_result = check_dependencies(alg_name, family_name)
            all_results.append(dep_result)
            if dep_result["success"]:
                print(f"  [依赖] ✓ 全部满足 {dep_result['details'].get('checked_deps', [])}")
                summary["dependency_pass"] += 1
            else:
                print(f"  [依赖] ✗ 缺少: {dep_result['missing']}")
                summary["dependency_fail"] += 1
                # 依赖缺失则跳过后续测试
                if not args.only_deps:
                    print(f"  [推理] 跳过 (依赖缺失)")
                    print(f"  [训练] 跳过 (依赖缺失)")
                    summary["skipped"] += 2
                continue

            if args.only_deps:
                continue

            # 2. 推理测试
            if not args.skip_inference:
                print(f"  [推理] 测试中...")
                inf_result = test_inference(alg_name, test_image)
                all_results.append(inf_result)
                if inf_result["success"]:
                    score = inf_result["details"].get("score")
                    has_map = inf_result["details"].get("has_anomaly_map", False)
                    print(f"  [推理] ✓ 成功 (score={score}, anomaly_map={has_map}, {inf_result['duration_s']}s)")
                    summary["inference_pass"] += 1
                else:
                    print(f"  [推理] ✗ 失败: {inf_result['error'][:200]}")
                    summary["inference_fail"] += 1
            else:
                summary["skipped"] += 1

            # 3. 训练测试
            if not args.skip_training:
                print(f"  [训练] 测试中 (10 iters)...")
                train_result = test_training(alg_name, family_name, test_category, data_source)
                all_results.append(train_result)
                if train_result["success"]:
                    print(f"  [训练] ✓ 成功 ({train_result['duration_s']}s)")
                    summary["training_pass"] += 1
                else:
                    err = (train_result["error"] or "")[:200]
                    print(f"  [训练] ✗ 失败: {err}")
                    summary["training_fail"] += 1
            else:
                summary["skipped"] += 1

    # 打印汇总
    print(f"\n{'='*70}")
    print("汇总")
    print(f"{'='*70}")
    print(f"  依赖检查: {summary['dependency_pass']} 通过, {summary['dependency_fail']} 失败")
    print(f"  推理测试: {summary['inference_pass']} 通过, {summary['inference_fail']} 失败")
    print(f"  训练测试: {summary['training_pass']} 通过, {summary['training_fail']} 失败")
    if summary['skipped'] > 0:
        print(f"  跳过: {summary['skipped']}")

    # 列出失败项
    failures = [r for r in all_results if not r["success"]]
    if failures:
        print(f"\n失败项 ({len(failures)}):")
        for r in failures:
            test_type = r.get("test", "?")
            alg = r.get("algorithm", "?")
            err = (r.get("error") or r.get("missing", []) or "未知")[:150]
            print(f"  [{test_type}] {alg}: {err}")

    # 保存详细结果
    results_path = PROJECT_ROOT / "records" / "algorithm_verification_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已保存到: {results_path}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
