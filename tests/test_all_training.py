#!/usr/bin/env python3
"""
全算法训练测试脚本

验证训练页面列出的所有可训练算法都能正常完成训练流程。
共 35 个可训练算法：Dinomaly 6 + Dinomaly2 3 + Anomalib 19 + ADer 7

用法:
  python tests/test_all_training.py                      # 全部算法
  python tests/test_all_training.py --quick              # 快速模式（每族 1 个）
  python tests/test_all_training.py --families dinomaly  # 指定族
  python tests/test_all_training.py --category bottle    # 指定数据集类别
  python tests/test_all_training.py --timeout 300        # 指定超时(秒)
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

# ==============================================================================
# 路径设置
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "algorithms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

SAVED_DIR = os.path.join(PROJECT_ROOT, "models", "saved", "训练验证")
os.makedirs(SAVED_DIR, exist_ok=True)

# ==============================================================================
# 环境变量设置 (离线模式, 使用本地缓存)
# ==============================================================================
_PRETRAINED = os.path.join(PROJECT_ROOT, "models", "pre_trained")
_ENV_VARS = {
    "HF_HUB_OFFLINE": "1",
    "HF_HOME": os.path.join(_PRETRAINED, "huggingface"),
    "TORCH_HOME": _PRETRAINED,
    "TRANSFORMERS_CACHE": os.path.join(_PRETRAINED, "huggingface"),
    "HUGGINGFACE_HUB_CACHE": os.path.join(_PRETRAINED, "huggingface"),
    "DINOMALY_ENCODER_DIR": _PRETRAINED,
    "PRETRAINED_MODELS_DIR": _PRETRAINED,
    "OPEN_CLIP_CACHE_DIR": os.path.join(_PRETRAINED, "open_clip"),
}
for k, v in _ENV_VARS.items():
    if k not in os.environ:
        os.environ[k] = v

MVTEC_ROOT = os.path.join(PROJECT_ROOT, "data", "public_dataset", "mvtec")

# ==============================================================================
# 算法族定义 — 与训练页面 ALGORITHM_FAMILIES 一致
# ==============================================================================

# Dinomaly 算法
DINOMALY_ALGORITHMS = OrderedDict([
    ("dinomaly_dinov3_small", {"model_type": "dinov3", "model_size": "small"}),
    ("dinomaly_dinov3_base",  {"model_type": "dinov3", "model_size": "base"}),
    ("dinomaly_dinov3_large", {"model_type": "dinov3", "model_size": "large"}),
    ("dinomaly_dinov2_small", {"model_type": "dinov2", "model_size": "small"}),
    ("dinomaly_dinov2_base",  {"model_type": "dinov2", "model_size": "base"}),
    ("dinomaly_dinov2_large", {"model_type": "dinov2", "model_size": "large"}),
])

# Dinomaly2 算法
DINOMALY2_ALGORITHMS = OrderedDict([
    ("dinomaly2_dinov2_small", {"model_type": "dinov2", "model_size": "small"}),
    ("dinomaly2_dinov2_base",  {"model_type": "dinov2", "model_size": "base"}),
    ("dinomaly2_dinov2_large", {"model_type": "dinov2", "model_size": "large"}),
    ("dinomaly2_dinov3_small", {"model_type": "dinov3", "model_size": "small"}),
    ("dinomaly2_dinov3_base",  {"model_type": "dinov3", "model_size": "base"}),
    ("dinomaly2_dinov3_large", {"model_type": "dinov3", "model_size": "large"}),
])

# Anomalib 可训练算法 (trainable: False 的不在列表中)
ANOMALIB_ALGORITHMS = OrderedDict([
    ("patchcore",            {}),
    ("cfa",                  {}),
    ("csflow",               {}),
    ("dfm",                  {}),
    ("draem",                {}),
    ("dsr",                  {}),
    ("fre",                  {}),
    ("reverse_distillation", {}),
    ("ganomaly",             {}),
    ("supersimplenet",       {}),
    ("uninet",               {}),
    ("winclip",              {}),
    ("anomalyvfm",           {}),
    ("general_ad",           {}),
    ("glass",                {}),
    ("inp_former",           {}),
    ("l2bt",                 {}),
    ("patchflow",            {}),
    ("anomaly_dino",         {}),
])

# Anomalib 训练 API 算法名 → anomalib 注册名映射
_ANOMALIB_NAME_MAP = {
    "anomalyvfm": "anomaly_v_f_m",
    "cfm": "c_f_m",
    "general_ad": "general_a_d",
    "glass": "glass",
    "inp_former": "inp_former",
    "l2bt": "l2_b_t",
    "patchflow": "patchflow",
    "anomaly_dino": "anomaly_d_i_n_o",
    "winclip": "win_clip",
    "uninet": "uni_net",
    "vlm_ad": "vlm_ad",
    "efficient_ad": "efficient_ad",
}

# ADer 可训练算法
ADER_ALGORITHMS = OrderedDict([
    ("mambaad",     {"method": "mambaad"}),
    ("invad",       {"method": "invad"}),
    ("vitad",       {"method": "vitad"}),
    ("unad",        {"method": "uniad"}),
    ("cflow",       {"method": "cflow"}),
    ("pyramidflow", {"method": "pyramidflow"}),
    ("simplenet",   {"method": "simplenet"}),
])

# Dinomaly2 backbone 映射
_D2_BACKBONE_MAP = {
    ("dinov2", "small"): "dinov2reg_vit_small_14",
    ("dinov2", "base"):  "dinov2reg_vit_base_14",
    ("dinov2", "large"): "dinov2reg_vit_large_14",
}

# 按算法族分组
FAMILIES = OrderedDict([
    ("dinomaly",  {"label": "Dinomaly",  "algorithms": DINOMALY_ALGORITHMS}),
    ("dinomaly2", {"label": "Dinomaly2", "algorithms": DINOMALY2_ALGORITHMS}),
    ("anomalib",  {"label": "Anomalib",  "algorithms": ANOMALIB_ALGORITHMS}),
    ("ader",      {"label": "ADer",      "algorithms": ADER_ALGORITHMS}),
])

# 快速模式：每族选 1 个代表
QUICK_SELECTION = {
    "dinomaly":  ["dinomaly_dinov3_small"],
    "dinomaly2": ["dinomaly2_dinov2_small"],
    "anomalib":  ["patchcore"],
    "ader":      ["mambaad"],
}

TIMEOUT_DEFAULT = 600  # 单算法超时(秒)


# ==============================================================================
# 结果记录
# ==============================================================================
@dataclass
class TestResult:
    algorithm: str
    family: str
    status: str = "pending"  # pass / fail / skip
    duration_s: float = 0.0
    returncode: Optional[int] = None
    error: Optional[str] = None
    cmd: Optional[str] = None


ALL_RESULTS: List[TestResult] = []


# ==============================================================================
# 工具函数
# ==============================================================================
def gpu_cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def build_env(extra: dict = None) -> dict:
    merged = os.environ.copy()
    merged["HF_HUB_OFFLINE"] = "1"
    merged["TORCH_HOME"] = _PRETRAINED
    merged["HF_HOME"] = os.path.join(_PRETRAINED, "huggingface")
    merged["TRANSFORMERS_CACHE"] = os.path.join(_PRETRAINED, "huggingface")
    merged["HUGGINGFACE_HUB_CACHE"] = os.path.join(_PRETRAINED, "huggingface", "hub")
    merged["DINOMALY_ENCODER_DIR"] = _PRETRAINED
    merged["PRETRAINED_MODELS_DIR"] = _PRETRAINED
    merged["OPEN_CLIP_CACHE_DIR"] = os.path.join(_PRETRAINED, "open_clip")
    algo_path = os.path.join(PROJECT_ROOT, "algorithms")
    backend_path = os.path.join(PROJECT_ROOT, "backend")
    existing_pp = merged.get("PYTHONPATH", "")
    merged["PYTHONPATH"] = f"{algo_path}:{backend_path}:{PROJECT_ROOT}" + (f":{existing_pp}" if existing_pp else "")
    if extra:
        merged.update(extra)
    return merged


def get_venv_python():
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    return venv_python if os.path.exists(venv_python) else sys.executable


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


# ==============================================================================
# Dinomaly Handler
# ==============================================================================
def test_dinomaly(alg_name: str, config: dict, category: str, timeout: int) -> TestResult:
    result = TestResult(algorithm=alg_name, family="dinomaly")
    model_type = config["model_type"]
    model_size = config["model_size"]

    save_name = f"test_{alg_name}_{datetime.now():%Y%m%d_%H%M%S}"
    cmd = [
        get_venv_python(), "-m", "algorithms.Dinomaly.dinomaly_train_evaluate",
        "--data_path", MVTEC_ROOT,
        "--save_dir", SAVED_DIR,
        "--save_name", save_name,
        "--model_size", model_size,
        "--model_type", model_type,
        "--batch_size", "4",
        "--total_iters", "10",
        "--categories", category,
    ]
    result.cmd = " ".join(cmd)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=PROJECT_ROOT, env=build_env(),
        )
        result.returncode = proc.returncode
        result.duration_s = round(time.time() - t0, 1)

        if proc.returncode == 0:
            result.status = "pass"
            log(f"  PASS ({result.duration_s:.0f}s)")
        else:
            err = (proc.stderr or "")[-500:]
            result.status = "fail"
            result.error = err
            log(f"  FAIL ({result.duration_s:.0f}s): {err[:200]}")

    except subprocess.TimeoutExpired:
        result.status = "fail"
        result.duration_s = timeout
        result.error = "训练超时"
        log(f"  TIMEOUT ({timeout}s)")
    except Exception as e:
        result.status = "fail"
        result.duration_s = round(time.time() - t0, 1)
        result.error = str(e)[:300]
        log(f"  ERROR: {e}")

    gpu_cleanup()
    return result


# ==============================================================================
# Dinomaly2 Handler
# ==============================================================================
def test_dinomaly2(alg_name: str, config: dict, category: str, timeout: int) -> TestResult:
    result = TestResult(algorithm=alg_name, family="dinomaly2")
    model_type = config["model_type"]
    model_size = config["model_size"]

    backbone = _D2_BACKBONE_MAP.get((model_type, model_size), "dinov2reg_vit_small_14")
    save_name = f"test_{alg_name}_{datetime.now():%Y%m%d_%H%M%S}"

    script_path = os.path.join(PROJECT_ROOT, "algorithms", "Dinomaly2", "dinomaly_2D.py")
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

    cmd = [
        get_venv_python(), script_path,
        "--data_path", MVTEC_ROOT,
        "--save_dir", SAVED_DIR,
        "--save_name", save_name,
        "--backbone", backbone,
        "--cuda", str(gpu_id),
        "--categories", category,
        "--total_iters", "10",
        "--image_size", "448",
        "--crop_size", "392",
        "--la", "1",
        "--lc", "2",
        "--cr", "1",
    ]
    result.cmd = " ".join(cmd)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=PROJECT_ROOT, env=build_env(),
        )
        result.returncode = proc.returncode
        result.duration_s = round(time.time() - t0, 1)

        if proc.returncode == 0:
            result.status = "pass"
            log(f"  PASS ({result.duration_s:.0f}s)")
        else:
            err = (proc.stderr or "")[-500:]
            result.status = "fail"
            result.error = err
            log(f"  FAIL ({result.duration_s:.0f}s): {err[:200]}")

    except subprocess.TimeoutExpired:
        result.status = "fail"
        result.duration_s = timeout
        result.error = "训练超时"
        log(f"  TIMEOUT ({timeout}s)")
    except Exception as e:
        result.status = "fail"
        result.duration_s = round(time.time() - t0, 1)
        result.error = str(e)[:300]
        log(f"  ERROR: {e}")

    gpu_cleanup()
    return result


# ==============================================================================
# Anomalib Handler
# ==============================================================================
def test_anomalib(alg_name: str, config: dict, category: str, timeout: int) -> TestResult:
    result = TestResult(algorithm=alg_name, family="anomalib")

    anomalib_algo = _ANOMALIB_NAME_MAP.get(alg_name, alg_name)
    save_dir = os.path.join(SAVED_DIR, f"test_{alg_name}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(save_dir, exist_ok=True)

    # 生成临时训练脚本
    train_script = f"""
import os, sys
sys.path.insert(0, '{PROJECT_ROOT}')
sys.path.insert(0, '{os.path.join(PROJECT_ROOT, "algorithms")}')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HOME', '{os.path.join(_PRETRAINED, "huggingface")}')
os.environ.setdefault('TORCH_HOME', '{_PRETRAINED}')
os.environ.setdefault('TRANSFORMERS_CACHE', '{os.path.join(_PRETRAINED, "huggingface")}')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '{os.path.join(_PRETRAINED, "huggingface", "hub")}')
os.environ.setdefault('PRETRAINED_MODELS_DIR', '{_PRETRAINED}')
os.environ.setdefault('DINOMALY_ENCODER_DIR', '{_PRETRAINED}')
import warnings
warnings.filterwarnings("ignore")

from anomalib.engine import Engine
from anomalib.models import get_model

model = get_model('{anomalib_algo}')
engine = Engine(
    max_epochs=1,
    default_root_dir='{save_dir}',
)
from anomalib.data import MVTecAD
datamodule = MVTecAD(root='{MVTEC_ROOT}', category='{category}', train_batch_size=4)
engine.fit(model=model, datamodule=datamodule)
print(f'Training completed: {alg_name}')
"""
    script_path = os.path.join(save_dir, "_train.py")
    with open(script_path, "w") as f:
        f.write(train_script)

    cmd = [get_venv_python(), script_path]
    result.cmd = " ".join(cmd)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=PROJECT_ROOT, env=build_env(),
        )
        result.returncode = proc.returncode
        result.duration_s = round(time.time() - t0, 1)

        if proc.returncode == 0:
            result.status = "pass"
            log(f"  PASS ({result.duration_s:.0f}s)")
        else:
            # 合并 stdout 和 stderr 做错误信息
            err = (proc.stderr or proc.stdout or "")[-500:]
            result.status = "fail"
            result.error = err
            log(f"  FAIL ({result.duration_s:.0f}s): {err[:200]}")

    except subprocess.TimeoutExpired:
        result.status = "fail"
        result.duration_s = timeout
        result.error = "训练超时"
        log(f"  TIMEOUT ({timeout}s)")
    except Exception as e:
        result.status = "fail"
        result.duration_s = round(time.time() - t0, 1)
        result.error = str(e)[:300]
        log(f"  ERROR: {e}")
    finally:
        # 清理临时脚本
        if os.path.exists(script_path):
            os.remove(script_path)

    gpu_cleanup()
    return result


# ==============================================================================
# ADer Handler
# ==============================================================================
def test_ader(alg_name: str, config: dict, category: str, timeout: int) -> TestResult:
    result = TestResult(algorithm=alg_name, family="ader")
    method_lower = config["method"]

    ader_dir = os.path.join(PROJECT_ROOT, "algorithms", "ADer")
    script_path = os.path.join(ader_dir, "run.py")

    # 查找 benchmark 配置
    cfg_file_name = f"{method_lower}_256_100e.py"
    benchmark_dir = os.path.join(ader_dir, "configs", "benchmark", method_lower)
    benchmark_cfg = None
    for candidate in [cfg_file_name, f"{method_lower}_mvtec.py", f"{method_lower}_256_300e.py"]:
        path = os.path.join(benchmark_dir, candidate)
        if os.path.isfile(path):
            benchmark_cfg = path
            break

    if not benchmark_cfg:
        result.status = "fail"
        result.error = f"benchmark 配置不存在: {benchmark_dir}/{cfg_file_name}"
        log(f"  SKIP (配置不存在)")
        return result

    cfg_path = f"configs/benchmark/{method_lower}/{os.path.basename(benchmark_cfg)}"

    # 计算相对路径
    ader_abs = os.path.abspath(ader_dir)
    data_abs = os.path.abspath(MVTEC_ROOT)
    try:
        data_rel = os.path.relpath(data_abs, ader_abs)
    except ValueError:
        data_rel = data_abs

    saved_abs = os.path.abspath(SAVED_DIR)
    try:
        checkpoint_rel = os.path.relpath(saved_abs, ader_abs)
    except ValueError:
        checkpoint_rel = saved_abs

    # 最小训练配置：1 epoch
    cmd = [
        get_venv_python(), script_path,
        "-c", cfg_path,
        "-m", "train",
        f"data.root={data_rel}",
        f"trainer.checkpoint={checkpoint_rel}",
        f"data.cls_names=['{category}']",
        "epoch_full=1",
        "test_start_epoch=1",
        "test_per_epoch=1",
        "trainer.epoch_full=1",
        "trainer.test_start_epoch=1",
        "trainer.test_per_epoch=1",
        "use_adeval=False",
        "metrics=['mAUROC_sp_max','mAUROC_sp_mean']",
        "evaluator.kwargs.use_adeval=False",
        "evaluator.kwargs.metrics=['mAUROC_sp_max','mAUROC_sp_mean']",
    ]
    # simplenet 测试阶段 OOM，减小 test batch size
    if alg_name == "simplenet":
        cmd.append("trainer.data.batch_size_per_gpu_test=8")
    result.cmd = " ".join(cmd)

    # ADer 需要 algorithms/ 在 PYTHONPATH 中且 CWD 为 algorithms/ADer/
    env = build_env({
        "PYTHONPATH": f"{os.path.join(PROJECT_ROOT, 'algorithms')}:{os.environ.get('PYTHONPATH', '')}",
    })

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=ader_dir, env=env,
        )
        result.returncode = proc.returncode
        result.duration_s = round(time.time() - t0, 1)

        if proc.returncode == 0:
            result.status = "pass"
            log(f"  PASS ({result.duration_s:.0f}s)")
        else:
            err = (proc.stderr or proc.stdout or "")[-500:]
            result.status = "fail"
            result.error = err
            log(f"  FAIL ({result.duration_s:.0f}s): {err[:200]}")

    except subprocess.TimeoutExpired:
        result.status = "fail"
        result.duration_s = timeout
        result.error = "训练超时"
        log(f"  TIMEOUT ({timeout}s)")
    except Exception as e:
        result.status = "fail"
        result.duration_s = round(time.time() - t0, 1)
        result.error = str(e)[:300]
        log(f"  ERROR: {e}")

    gpu_cleanup()
    return result


# ==============================================================================
# Handler 分发
# ==============================================================================
HANDLERS = {
    "dinomaly":  test_dinomaly,
    "dinomaly2": test_dinomaly2,
    "anomalib":  test_anomalib,
    "ader":      test_ader,
}


# ==============================================================================
# 主流程
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="全算法训练测试")
    parser.add_argument("--category", default="capsule", help="MVTec 类别 (默认: capsule)")
    parser.add_argument("--families", nargs="+", default=None,
                        help="限定测试的 family (如: dinomaly anomalib ader)")
    parser.add_argument("--quick", action="store_true", help="快速模式 (每族 1 个)")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_DEFAULT, help="单算法超时(秒)")
    parser.add_argument("--report", default="records/training_test_report.json",
                        help="报告输出路径")
    args = parser.parse_args()

    # 验证数据集
    category_dir = os.path.join(MVTEC_ROOT, args.category, "train", "good")
    if not os.path.isdir(category_dir):
        log(f"错误: 训练目录不存在 {category_dir}")
        sys.exit(1)

    log(f"{'=' * 60}")
    log(f"全算法训练测试")
    log(f"数据集: {MVTEC_ROOT}/{args.category}")
    log(f"训练样本: {len(os.listdir(category_dir))} 张")
    log(f"模式: {'快速' if args.quick else '完整'}")
    log(f"超时: {args.timeout}s")
    log(f"{'=' * 60}")

    total = 0
    passed = 0
    failed = 0

    for family_key, family_def in FAMILIES.items():
        if args.families and family_key not in args.families:
            continue

        handler = HANDLERS.get(family_key)
        if handler is None:
            continue

        algorithms = family_def["algorithms"]
        if args.quick:
            quick_list = QUICK_SELECTION.get(family_key, [])
            algorithms = OrderedDict([(k, v) for k, v in algorithms.items() if k in quick_list])

        log(f"\n{'=' * 40}")
        log(f"Family: {family_def['label']} ({len(algorithms)} 算法)")

        for alg_name, alg_config in algorithms.items():
            total += 1
            log(f"\n--- [{total}] {alg_name} ---")

            try:
                r = handler(alg_name, alg_config, args.category, args.timeout)
            except Exception as e:
                traceback.print_exc()
                r = TestResult(algorithm=alg_name, family=family_key, status="fail",
                               error=f"未捕获异常: {str(e)[:300]}")

            ALL_RESULTS.append(r)

            if r.status == "pass":
                passed += 1
            else:
                failed += 1

            # 每个算法测完保存一次报告
            _save_report(args.report, args)

            time.sleep(2)  # GPU 冷却

    # ==========================================================================
    # 报告
    # ==========================================================================
    log(f"\n{'=' * 60}")
    log(f"测试完成")
    log(f"总计: {total}  |  通过: {passed}  |  失败: {failed}")
    log(f"{'=' * 60}")

    if failed > 0:
        log(f"\n失败算法:")
        for r in ALL_RESULTS:
            if r.status == "fail":
                log(f"  - {r.algorithm} ({r.family}): {r.error[:200] if r.error else 'unknown'}")

    _save_report(args.report, args)

    return 0 if failed == 0 else 1


def _save_report(report_path: str, args):
    os.makedirs(os.path.dirname(os.path.join(PROJECT_ROOT, report_path)), exist_ok=True)
    full_path = os.path.join(PROJECT_ROOT, report_path)

    passed = sum(1 for r in ALL_RESULTS if r.status == "pass")
    failed = sum(1 for r in ALL_RESULTS if r.status == "fail")
    total = len(ALL_RESULTS)

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": f"data/public_dataset/mvtec/{args.category}",
        "mode": "quick" if args.quick else "full",
        "timeout": args.timeout,
        "summary": {"total": total, "passed": passed, "failed": failed},
        "results": [asdict(r) for r in ALL_RESULTS],
    }
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
