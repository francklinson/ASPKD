#!/usr/bin/env python3
"""
综合算法验证脚本
逐一验证所有注册算法的推理和训练能力
数据集: data/public_dataset/mvtec/bottle

用法:
    python3 tests/verify_all_algorithms.py [--quick] [--family FAMILY] [--skip-training]

    --quick: 快速模式，只验证注册和创建，不实际加载模型
    --family: 只验证指定算法族 (dinomaly/dinomaly2/anomalib/ader/baseasd/musc/subspacead/other)
    --skip-training: 跳过训练验证
    --inference-only: 只验证推理
    --training-only: 只验证训练
"""
import os
import sys
import time
import json
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 设置环境变量
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DINOMALY_ENCODER_DIR", os.path.join(PROJECT_ROOT, "models", "pre_trained"))
os.environ.setdefault("PRETRAINED_MODELS_DIR", os.path.join(PROJECT_ROOT, "models", "pre_trained"))
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_ROOT, "models", "pre_trained"))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, "models", "pre_trained", "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(PROJECT_ROOT, "models", "pre_trained", "huggingface"))

# 数据集路径
DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "public_dataset", "mvtec", "bottle")
TEST_GOOD_DIR = os.path.join(DATASET_ROOT, "test", "good")
TEST_BAD_DIR = os.path.join(DATASET_ROOT, "test", "broken_large")
TRAIN_GOOD_DIR = os.path.join(DATASET_ROOT, "train", "good")
REPORT_PATH = os.path.join(PROJECT_ROOT, "records", "verify_all_algorithms_report.json")

# ---- 算法族定义 ----
ALGORITHM_FAMILIES = {
    "dinomaly": {
        "name": "Dinomaly",
        "trainable": True,
        "algorithms": [
            "dinomaly_dinov3_small", "dinomaly_dinov3_base", "dinomaly_dinov3_large",
            "dinomaly_dinov2_small", "dinomaly_dinov2_base", "dinomaly_dinov2_large",
        ]
    },
    "dinomaly2": {
        "name": "Dinomaly2",
        "trainable": False,
        "algorithms": [
            "dinomaly2_dinov2_small", "dinomaly2_dinov2_base", "dinomaly2_dinov2_large",
            "dinomaly2_dinov3_small", "dinomaly2_dinov3_base", "dinomaly2_dinov3_large",
        ]
    },
    "anomalib": {
        "name": "Anomalib",
        "trainable": True,
        "algorithms": [
            "patchcore", "cfa", "csflow", "dfkde", "dfm", "draem", "dsr",
            "efficient_ad", "fastflow", "fre", "padim", "reverse_distillation",
            "stfpm", "ganomaly", "supersimplenet", "uflow", "uninet", "vlm_ad", "winclip",
            "anomalyvfm", "cfm", "general_ad", "glass", "inp_former", "l2bt",
            "patchflow", "anomaly_dino",
        ]
    },
    "ader": {
        "name": "ADer",
        "trainable": True,
        "algorithms": [
            "mambaad", "invad", "vitad", "unad", "cflow", "pyramidflow", "simplenet",
        ]
    },
    "baseasd": {
        "name": "BaseASD",
        "trainable": False,
        "algorithms": [
            "denseae", "cae", "vae", "aegan", "differnet",
        ]
    },
    "musc": {
        "name": "MuSc (零样本)",
        "trainable": False,
        "algorithms": [
            "musc_clip_b32_512", "musc_clip_b16_512", "musc_clip_l14_336",
            "musc_clip_l14_518", "musc_dinov2_b14_336", "musc_dinov2_b14_518",
            "musc_dinov2_l14_336", "musc_dinov2_l14_518",
        ]
    },
    "subspacead": {
        "name": "SubspaceAD (少样本)",
        "trainable": False,
        "algorithms": [
            "subspacead_dinov2_large_672", "subspacead_dinov2_large_518",
            "subspacead_dinov2_large_336", "subspacead_dinov2_base_672",
            "subspacead_dinov2_base_518", "subspacead_dinov2_small_672",
        ]
    },
    "other": {
        "name": "Other (存根)",
        "trainable": False,
        "algorithms": [
            "hiad", "multiads", "musc", "dictas", "subspacead", "diad", "audio_feature_cluster",
        ]
    },
}


def get_test_images() -> Tuple[List[str], List[str]]:
    """获取测试用图片"""
    normal_images = []
    anomaly_images = []

    if os.path.exists(TEST_GOOD_DIR):
        for f in sorted(os.listdir(TEST_GOOD_DIR)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                normal_images.append(os.path.join(TEST_GOOD_DIR, f))

    if os.path.exists(TEST_BAD_DIR):
        for f in sorted(os.listdir(TEST_BAD_DIR)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                anomaly_images.append(os.path.join(TEST_BAD_DIR, f))

    # fallback: 使用 train/good
    if not normal_images and os.path.exists(TRAIN_GOOD_DIR):
        for f in sorted(os.listdir(TRAIN_GOOD_DIR)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                normal_images.append(os.path.join(TRAIN_GOOD_DIR, f))

    return normal_images, anomaly_images


def get_reference_images(normal_images: List[str], k: int = 5) -> List[str]:
    """获取参考图片（用于少样本算法）"""
    return normal_images[:min(k, len(normal_images))]


class VerificationReport:
    """验证报告收集器"""

    def __init__(self):
        self.results = {}
        self.start_time = datetime.now().isoformat()
        self.summary = {
            "total": 0,
            "inference_pass": 0,
            "inference_fail": 0,
            "inference_skip": 0,
            "training_pass": 0,
            "training_fail": 0,
            "training_skip": 0,
            "errors": [],
        }

    def add_result(self, alg_name: str, family: str, result: dict):
        self.results[alg_name] = {
            "family": family,
            **result
        }
        self.summary["total"] += 1

        if result.get("inference") == "pass":
            self.summary["inference_pass"] += 1
        elif result.get("inference") == "fail":
            self.summary["inference_fail"] += 1
        elif result.get("inference") == "skip":
            self.summary["inference_skip"] += 1

        if result.get("training") == "pass":
            self.summary["training_pass"] += 1
        elif result.get("training") == "fail":
            self.summary["training_fail"] += 1
        elif result.get("training") == "skip":
            self.summary["training_skip"] += 1

        if result.get("error"):
            self.summary["errors"].append(f"{alg_name}: {result['error']}")

    def save(self):
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        report = {
            "timestamp": datetime.now().isoformat(),
            "start_time": self.start_time,
            "summary": self.summary,
            "results": self.results,
        }
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return REPORT_PATH

    def print_summary(self):
        s = self.summary
        print("\n" + "=" * 80)
        print("验证报告汇总")
        print("=" * 80)
        print(f"总算法数:       {s['total']}")
        print(f"推理通过:       {s['inference_pass']} ✅")
        print(f"推理失败:       {s['inference_fail']} ❌")
        print(f"推理跳过:       {s['inference_skip']} ⏭️")
        print(f"训练通过:       {s['training_pass']} ✅")
        print(f"训练失败:       {s['training_fail']} ❌")
        print(f"训练跳过:       {s['training_skip']} ⏭️")
        if s["errors"]:
            print(f"\n错误列表 ({len(s['errors'])} 个):")
            for err in s["errors"][:20]:
                print(f"  - {err}")
            if len(s["errors"]) > 20:
                print(f"  ... 及其他 {len(s['errors']) - 20} 个错误")
        print(f"\n报告已保存到: {REPORT_PATH}")


def test_registration():
    """测试算法注册是否正常"""
    print("\n" + "=" * 80)
    print("阶段 0: 验证算法注册")
    print("=" * 80)

    from algorithms import list_available_algorithms
    from backend.core import AlgorithmRegistry

    all_algs = list_available_algorithms()
    print(f"已注册算法总数: {len(all_algs)}")

    registered_names = set(all_algs)
    expected_names = set()
    for family, info in ALGORITHM_FAMILIES.items():
        expected_names.update(info["algorithms"])

    missing = expected_names - registered_names
    extra = registered_names - expected_names

    if missing:
        print(f"⚠️  预期但未注册的算法 ({len(missing)}): {sorted(missing)}")
    if extra:
        print(f"ℹ️  额外注册的算法 ({len(extra)}): {sorted(extra)}")
    if not missing and not extra:
        print("✅ 所有预期算法均已注册，无多余注册")

    print(f"\n各算法族注册情况:")
    for family, info in ALGORITHM_FAMILIES.items():
        registered = [a for a in info["algorithms"] if a in registered_names]
        unregistered = [a for a in info["algorithms"] if a not in registered_names]
        status = "✅" if not unregistered else "⚠️"
        print(f"  {status} {family} ({info['name']}): {len(registered)}/{len(info['algorithms'])} 已注册")

    return all_algs, registered_names


def test_inference(alg_name: str, family: str, test_img: str,
                   ref_imgs: List[str], quick: bool = False) -> dict:
    """测试单个算法的推理能力"""
    result = {
        "inference": "skip",
        "inference_time_ms": 0,
        "anomaly_score": 0,
        "error": None,
        "warnings": [],
    }

    if quick:
        result["inference"] = "skip"
        result["error"] = "quick mode"
        return result

    from algorithms import create_detector

    print(f"\n{'─' * 60}")
    print(f"🧪 推理测试: {alg_name} ({family})")
    print(f"   测试图片: {os.path.basename(test_img)}")

    detector = None
    try:
        # 清空 CUDA 缓存
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 1. 创建检测器
        t0 = time.time()
        detector = create_detector(alg_name)
        create_time = (time.time() - t0) * 1000
        print(f"   ✅ 创建检测器: {create_time:.0f}ms")

        # 2. 加载模型
        t0 = time.time()
        detector.load_model()
        load_time = (time.time() - t0) * 1000
        print(f"   ✅ 加载模型: {load_time:.0f}ms")

        # 3. 推理
        t0 = time.time()

        # SubspaceAD 需要参考图片
        if family == "subspacead":
            if len(ref_imgs) >= 1:
                pred_result = detector.predict(test_img, reference_paths=ref_imgs)
            else:
                pred_result = detector.predict(test_img)
        else:
            pred_result = detector.predict(test_img)

        inference_time = (time.time() - t0) * 1000
        result["inference"] = "pass"
        result["inference_time_ms"] = round(inference_time, 2)
        result["anomaly_score"] = round(float(pred_result.anomaly_score), 6)
        result["is_anomaly"] = bool(pred_result.is_anomaly)
        result["has_anomaly_map"] = pred_result.anomaly_map is not None
        result["load_time_ms"] = round(load_time, 0)
        result["create_time_ms"] = round(create_time, 0)

        print(f"   ✅ 推理完成: {inference_time:.0f}ms, score={pred_result.anomaly_score:.6f}, "
              f"is_anomaly={pred_result.is_anomaly}, has_map={pred_result.anomaly_map is not None}")

    except NotImplementedError as e:
        result["inference"] = "fail"
        result["error"] = f"NotImplementedError: {str(e)[:200]}"
        print(f"   ❌ 未实现: {str(e)[:200]}")
    except FileNotFoundError as e:
        result["inference"] = "fail"
        result["error"] = f"FileNotFoundError: {str(e)[:200]}"
        print(f"   ❌ 文件未找到: {str(e)[:200]}")
    except ImportError as e:
        result["inference"] = "fail"
        result["error"] = f"ImportError: {str(e)[:200]}"
        print(f"   ❌ 导入错误: {str(e)[:200]}")
    except Exception as e:
        result["inference"] = "fail"
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"   ❌ 推理失败: {type(e).__name__}: {str(e)[:300]}")
        # 打印完整 traceback 用于调试
        traceback.print_exc()
    finally:
        if detector is not None:
            try:
                detector.release()
            except Exception:
                pass
        # 清空 CUDA 缓存，释放 GPU 内存
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return result


def test_training(alg_name: str, family: str, quick: bool = False) -> dict:
    """测试单个算法的训练管线"""
    result = {
        "training": "skip",
        "error": None,
    }

    if not ALGORITHM_FAMILIES.get(family, {}).get("trainable", False):
        result["training"] = "skip"
        result["error"] = "算法族不支持训练"
        return result

    if quick:
        result["training"] = "skip"
        result["error"] = "quick mode"
        return result

    print(f"\n{'─' * 60}")
    print(f"🔧 训练验证: {alg_name} ({family})")

    try:
        # 验证训练管线可以初始化
        # 1. 检查训练脚本是否存在
        if family == "dinomaly":
            train_module = "algorithms.Dinomaly.dinomaly_train_evaluate"
            # 尝试导入
            try:
                __import__(train_module)
                print(f"   ✅ 训练模块 {train_module} 可导入")
                result["training"] = "pass"
                result["train_module"] = train_module
            except ImportError as e:
                result["training"] = "fail"
                result["error"] = f"训练模块导入失败: {str(e)}"
                print(f"   ❌ 训练模块导入失败: {str(e)}")

        elif family == "anomalib":
            # 检查 Anomalib CLI 和模型
            from anomalib.models import get_model
            try:
                model = get_model(alg_name)
                print(f"   ✅ Anomalib 模型 '{alg_name}' 可创建")
                result["training"] = "pass"

                # 检查是否有对应的 lightning_model
                model_dir = os.path.join(PROJECT_ROOT, "algorithms", "Anomalib", "models", "image", alg_name)
                if os.path.exists(model_dir):
                    print(f"   ✅ 模型目录存在: {model_dir}")
                else:
                    # 尝试其他可能的路径
                    alt_paths = [
                        os.path.join(PROJECT_ROOT, "algorithms", "Anomalib", "models", "image", alg_name.replace("_", "")),
                    ]
                    found = False
                    for p in alt_paths:
                        if os.path.exists(p):
                            print(f"   ✅ 模型目录存在: {p}")
                            found = True
                            break
                    if not found:
                        result["warnings"] = result.get("warnings", []) + [f"模型目录未找到: {alg_name}"]
                        print(f"   ⚠️  模型目录未找到（可能使用内置注册）")
            except Exception as e:
                result["training"] = "fail"
                result["error"] = f"Anomalib 模型创建失败: {str(e)}"
                print(f"   ❌ Anomalib 模型创建失败: {str(e)}")

        elif family == "ader":
            # 检查 ADer run.py 和 config
            def _ader_method_name_local(algo_name: str) -> str:
                mapping = {
                    "mambaad": "MambaAD", "invad": "InVad", "vitad": "ViTAD",
                    "unad": "UniAD", "cflow": "CFlow", "pyramidflow": "PyramidFlow",
                    "simplenet": "SimpleNet",
                }
                return mapping.get(algo_name, "MambaAD")
            method_name = _ader_method_name_local(alg_name)

            run_py = os.path.join(PROJECT_ROOT, "algorithms", "ADer", "run.py")
            config_dir = os.path.join(PROJECT_ROOT, "algorithms", "ADer", "configs", method_name.lower())

            if os.path.exists(run_py):
                print(f"   ✅ ADer run.py 存在: {run_py}")
            else:
                print(f"   ❌ ADer run.py 不存在: {run_py}")
                result["training"] = "fail"
                result["error"] = "ADer run.py 不存在"
                return result

            # 查找配置文件
            config_file = None
            if os.path.exists(config_dir):
                for fn in os.listdir(config_dir):
                    if fn.endswith("_spk.py"):
                        config_file = os.path.join(config_dir, fn)
                        break
                # 如果没有 _spk 配置，找任意 .py
                if not config_file:
                    for fn in os.listdir(config_dir):
                        if fn.endswith(".py"):
                            config_file = os.path.join(config_dir, fn)
                            break

            if config_file:
                print(f"   ✅ 配置文件: {config_file}")
                result["training"] = "pass"
                result["config_file"] = config_file
            else:
                print(f"   ⚠️  未找到 _spk 配置文件，使用默认配置")
                result["training"] = "pass"
                result["warnings"] = result.get("warnings", []) + ["未找到 _spk 配置文件"]

    except Exception as e:
        result["training"] = "fail"
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"   ❌ 训练验证失败: {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="综合算法验证")
    parser.add_argument("--quick", action="store_true", help="快速模式（仅验证注册）")
    parser.add_argument("--family", type=str, help="只验证指定算法族")
    parser.add_argument("--skip-training", action="store_true", help="跳过训练验证")
    parser.add_argument("--inference-only", action="store_true", help="仅验证推理")
    parser.add_argument("--training-only", action="store_true", help="仅验证训练")
    parser.add_argument("--no-inference", action="store_true", help="跳过推理验证")
    args = parser.parse_args()

    print("=" * 80)
    print("综合算法验证 - 推理 & 训练")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"数据集: {DATASET_ROOT}")
    print(f"模式: {'快速' if args.quick else '完整'}")
    print("=" * 80)

    # 检查 CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA: {'✅ 可用' if cuda_available else '❌ 不可用 (CPU 模式)'}")
        if cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        print("CUDA: 无法检测")

    # 检查数据集
    normal_imgs, anomaly_imgs = get_test_images()
    if not normal_imgs:
        print(f"❌ 未找到测试图片，请确保数据集存在: {DATASET_ROOT}")
        sys.exit(1)
    print(f"测试图片: {len(normal_imgs)} 正常 + {len(anomaly_imgs)} 异常")
    test_img = normal_imgs[0]
    ref_imgs = get_reference_images(normal_imgs, k=5)
    print(f"推理测试图: {os.path.basename(test_img)}")
    print(f"参考图片数: {len(ref_imgs)}")

    # 阶段 0: 验证注册
    all_algs, registered_names = test_registration()

    # 确定要测试的算法族
    families_to_test = list(ALGORITHM_FAMILIES.keys())
    if args.family:
        if args.family in ALGORITHM_FAMILIES:
            families_to_test = [args.family]
        else:
            print(f"❌ 未知算法族: {args.family}")
            print(f"可用: {list(ALGORITHM_FAMILIES.keys())}")
            sys.exit(1)

    report = VerificationReport()

    # 逐个算法族测试
    for family in families_to_test:
        info = ALGORITHM_FAMILIES[family]
        print(f"\n{'=' * 80}")
        print(f"阶段: {info['name']} ({family})")
        print(f"算法数: {len(info['algorithms'])}, 可训练: {info['trainable']}")
        print(f"{'=' * 80}")

        for alg_name in info["algorithms"]:
            if alg_name not in registered_names:
                print(f"\n  ⚠️  {alg_name}: 未注册，跳过")
                report.add_result(alg_name, family, {
                    "inference": "skip",
                    "training": "skip",
                    "error": "未注册",
                })
                continue

            # 推理测试
            if args.no_inference or args.training_only:
                inference_result = {"inference": "skip", "error": "skipped by args"}
            else:
                inference_result = test_inference(
                    alg_name, family, test_img, ref_imgs, quick=args.quick
                )

            # 训练测试
            if args.skip_training or args.inference_only:
                training_result = {"training": "skip", "error": "skipped by args"}
            else:
                training_result = test_training(alg_name, family, quick=args.quick)

            combined = {**inference_result, **training_result}
            report.add_result(alg_name, family, combined)

            # 每测试完一个算法就保存报告（防止中断丢失）
            report.save()

    # 打印最终汇总
    report.print_summary()

    # 保存最终报告
    report_path = report.save()
    print(f"\n详细报告: {report_path}")

    # 返回退出码
    if report.summary["inference_fail"] > 0 or report.summary["training_fail"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
