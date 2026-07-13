#!/usr/bin/env python3
"""
批量训练脚本：为 ADer / Dinomaly2 / Anomalib 缺失 checkpoint 的算法
使用 MVTec bottle 数据集进行快速训练（5 epochs）。

用法: python batch_train.py
"""

import os
import sys
import shutil
import subprocess
import time
import glob
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ADER_DIR = os.path.join(PROJECT_ROOT, "algorithms", "ADer")
D2_SCRIPT = os.path.join(PROJECT_ROOT, "algorithms", "Dinomaly2", "dinomaly_2D.py")
SAVED_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "models", "pre_trained")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "public_dataset", "mvtec")
CATEGORIES = ["bottle"]
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

os.makedirs(SAVED_DIR, exist_ok=True)
os.makedirs(PRETRAINED_DIR, exist_ok=True)

# ============================================================
# ADer 算法列表及配置
# ============================================================
ADER_ALGOS = [
    # (algo_name, config_path, trainer_class_name)
    ("mambaad", "configs/benchmark/mambaad/mambaad_256_100e.py", "MambaADTrainer"),
    ("invad", "configs/benchmark/invad/invad_256_100e.py", "InVadTrainer"),
    ("vitad", "configs/benchmark/vitad/vitad_256_100e.py", "ViTADTrainer"),
    ("unad", "configs/benchmark/uniad/uniad_256_100e.py", "UniADTrainer"),
    ("cflow", "configs/benchmark/cflow/cflow_256_100e.py", "CFlowTrainer"),
    ("pyramidflow", "configs/benchmark/pyramidflow/pyramidflow_256_100e.py", "PyramidFlowTrainer"),
    ("simplenet", "configs/benchmark/simplenet/simplenet_256_100e.py", "SimpleNetTrainer"),
    ("destseg", "configs/benchmark/destseg/destseg_256_100e.py", "DeSTSegTrainer"),
    ("realnet", "configs/benchmark/realnet/realnet_256_100e.py", "RealNetTrainer"),
    ("rdpp", "configs/benchmark/rdpp/rdpp_256_100e.py", "RDPlusPlusTrainer"),
]

# ============================================================
# Dinomaly2 算法列表
# ============================================================
D2_ALGOS = [
    ("dinov2", "small", "dinov2reg_vit_small_14"),
    ("dinov2", "base", "dinov2reg_vit_base_14"),
    ("dinov2", "large", "dinov2reg_vit_large_14"),
    ("dinov3", "small", "dinov3_vit_small_16"),
    ("dinov3", "base", "dinov3_vit_base_16"),
    ("dinov3", "large", "dinov3_vit_large_16"),
]

# ============================================================
# Anomalib 算法列表（训练可用但 checkpoints 缺失）
# ============================================================
ANOMALIB_ALGOS = [
    "patchcore", "cfa", "csflow", "dfm", "draem", "dsr",
    "fre", "ganomaly", "reverse_distillation", "supersimplenet",
    "uninet", "winclip",
]


def run_cmd(cmd, cwd=None, env=None, timeout=1800):
    """运行命令，返回 (success, output)"""
    start = time.time()
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] 运行: {' '.join(cmd[:6])}...")
    try:
        result = subprocess.run(
            cmd, cwd=cwd, env=env or os.environ,
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - start
        ok = result.returncode == 0 or "finish training" in result.stdout + result.stderr
        status = "✓" if ok else "✗"
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {status} 完成 ({elapsed:.0f}s, rc={result.returncode})")
        if not ok:
            tail = (result.stdout + result.stderr)[-500:]
            print(f"  输出尾部: {tail[:300]}")
        return ok, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] ✗ 超时 ({elapsed:.0f}s)")
        return False, ""
    except Exception as e:
        print(f"  异常: {e}")
        return False, ""


def train_ader(algo_name, config_path):
    """训练单个 ADer 算法"""
    print(f"\n{'='*60}")
    print(f"ADer: {algo_name}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}/algorithms:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [
        VENV_PYTHON, "run.py",
        "-c", config_path,
        "-m", "train",
        f"data.root=../../data/public_dataset/mvtec",
        f"trainer.checkpoint=../../models/saved",
        "data.cls_names=['bottle']",
        "epoch_full=5",
        "trainer.test_start_epoch=999",
        "trainer.test_per_epoch=999",
        "batch_train=8",
        "trainer.data.batch_size_per_gpu_test=1",
        "use_adeval=False",
    ]
    return run_cmd(cmd, cwd=ADER_DIR, env=env, timeout=1800)


def find_latest_ader_checkpoint(algo_name):
    """找到最新训练的 ADer checkpoint net.pth"""
    pattern = os.path.join(SAVED_DIR, f"*{algo_name}*", "net.pth")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    # 排除 soft link targets 的问题，直接找目录下的 net.pth
    for root, dirs, files in os.walk(SAVED_DIR):
        for d in sorted(dirs, reverse=True):
            if algo_name.lower() in d.lower():
                net_path = os.path.join(root, d, "net.pth")
                if os.path.exists(net_path) and os.path.getsize(net_path) > 1000000:
                    return net_path
    return None


def train_dinomaly2(model_type, model_size, backbone):
    """训练单个 Dinomaly2 算法"""
    algo_label = f"dinomaly2_{model_type}_{model_size}"
    save_name = f"dinomaly2_{model_type}_{model_size}_bottle_quick"

    print(f"\n{'='*60}")
    print(f"Dinomaly2: {algo_label} (backbone={backbone})")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HOME"] = os.path.join(PROJECT_ROOT, "models", "pre_trained", "huggingface")

    cmd = [
        VENV_PYTHON, D2_SCRIPT,
        "--data_path", DATA_ROOT,
        "--save_dir", SAVED_DIR,
        "--save_name", save_name,
        "--backbone", backbone,
        "--cuda", "0",
        "--categories", "bottle",
        "--total_iters", "500",
        "--image_size", "448",
        "--crop_size", "392",
        "--la", "1", "--lc", "2", "--cr", "1",
    ]
    if "v3" in backbone:
        cmd.append("--use_get_intermediate")

    return run_cmd(cmd, cwd=PROJECT_ROOT, env=env, timeout=1800)


def find_latest_d2_checkpoint(algo_label):
    """找到最新训练的 Dinomaly2 checkpoint model.pth"""
    for root, dirs, files in os.walk(SAVED_DIR):
        for d in sorted(dirs, reverse=True):
            if algo_label in d.lower() or algo_label.replace("_", "") in d.lower():
                model_path = os.path.join(root, d, "model.pth")
                if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
                    return model_path
                # Also check for .pth files
                for f in files:
                    if f.endswith(".pth") and os.path.getsize(os.path.join(root, d, f)) > 1000000:
                        return os.path.join(root, d, f)
    return None


def train_anomalib(algo_name):
    """训练单个 Anomalib 算法"""
    print(f"\n{'='*60}")
    print(f"Anomalib: {algo_name}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HOME"] = os.path.join(PROJECT_ROOT, "models", "pre_trained", "huggingface")

    save_name = f"anomalib_{algo_name}_bottle_quick"
    algorithms_dir = os.path.join(PROJECT_ROOT, "algorithms")

    # Anomalib name mapping
    anomalib_name_map = {
        "patchcore": "patchcore", "cfa": "cfa", "csflow": "csflow",
        "dfm": "dfm", "draem": "draem", "dsr": "dsr",
        "fre": "fre", "ganomaly": "ganomaly",
        "reverse_distillation": "reverse_distillation",
        "supersimplenet": "supersimplenet", "uninet": "uni_net",
        "winclip": "win_clip",
    }
    anomalib_algo = anomalib_name_map.get(algo_name, algo_name)

    script_content = f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
sys.path.insert(0, '{algorithms_dir}')
from anomalib.engine import Engine
from anomalib.models import get_model

model = get_model('{anomalib_algo}')
engine = Engine(max_epochs=5, default_root_dir='{SAVED_DIR}/{save_name}')
from anomalib.data import MVTecAD
datamodule = MVTecAD(root='{DATA_ROOT}', category='bottle', train_batch_size=8)
engine.fit(model=model, datamodule=datamodule)
print('Training completed: {algo_name}')
"""
    tmp_script = os.path.join(SAVED_DIR, f"_train_{algo_name}.py")
    with open(tmp_script, "w") as f:
        f.write(script_content)

    ok, output = run_cmd([VENV_PYTHON, tmp_script], cwd=PROJECT_ROOT, env=env, timeout=1800)
    if os.path.exists(tmp_script):
        os.remove(tmp_script)
    return ok, output


def find_latest_anomalib_checkpoint(algo_name):
    """找到最新训练的 Anomalib checkpoint model.ckpt"""
    for root, dirs, files in os.walk(SAVED_DIR):
        for d in sorted(dirs, reverse=True):
            if algo_name.lower() in d.lower():
                for sub_root, sub_dirs, sub_files in os.walk(os.path.join(root, d)):
                    for f in sub_files:
                        if f == "model.ckpt" and os.path.getsize(os.path.join(sub_root, f)) > 1000000:
                            return os.path.join(sub_root, f)
    return None


def copy_checkpoint(src, algo_name):
    """复制 checkpoint 到 models/pre_trained/"""
    if not src or not os.path.exists(src):
        print(f"  ✗ 找不到 checkpoint for {algo_name}")
        return False

    dst = os.path.join(PRETRAINED_DIR, f"{algo_name}_best.pth")
    size_mb = os.path.getsize(src) / (1024 * 1024)
    shutil.copy2(src, dst)
    print(f"  ✓ {algo_name}_best.pth ({size_mb:.0f}MB)")
    return True


def main():
    results = {"success": [], "failed": []}

    # ── ADer ──
    print("\n" + "=" * 70)
    print("阶段 1/3: ADer 算法训练 (10个)")
    print("=" * 70)

    for algo_name, config_path, _ in ADER_ALGOS:
        # Skip if already exists
        dst = os.path.join(PRETRAINED_DIR, f"{algo_name}_best.pth")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000000:
            print(f"\n  ⏭ {algo_name} 已有 checkpoint，跳过")
            results["success"].append(algo_name)
            continue

        ok, _ = train_ader(algo_name, config_path)
        checkpoint = find_latest_ader_checkpoint(algo_name)
        if checkpoint and copy_checkpoint(checkpoint, algo_name):
            results["success"].append(algo_name)
        else:
            results["failed"].append(algo_name)

    # ── Dinomaly2 ──
    print("\n" + "=" * 70)
    print("阶段 2/3: Dinomaly2 算法训练 (6个)")
    print("=" * 70)

    for model_type, model_size, backbone in D2_ALGOS:
        algo_label = f"dinomaly2_{model_type}_{model_size}"
        dst = os.path.join(PRETRAINED_DIR, f"{algo_label}.pth")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000000:
            print(f"\n  ⏭ {algo_label} 已有 checkpoint，跳过")
            results["success"].append(algo_label)
            continue

        ok, _ = train_dinomaly2(model_type, model_size, backbone)
        checkpoint = find_latest_d2_checkpoint(algo_label)
        if checkpoint and copy_checkpoint(checkpoint, algo_label):
            results["success"].append(algo_label)
        else:
            results["failed"].append(algo_label)

    # ── Anomalib ──
    print("\n" + "=" * 70)
    print("阶段 3/3: Anomalib 算法训练 (12个)")
    print("=" * 70)

    for algo_name in ANOMALIB_ALGOS:
        dst = os.path.join(PRETRAINED_DIR, f"{algo_name}_best.pth")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000000:
            print(f"\n  ⏭ {algo_name} 已有 checkpoint，跳过")
            results["success"].append(algo_name)
            continue

        ok, _ = train_anomalib(algo_name)
        checkpoint = find_latest_anomalib_checkpoint(algo_name)
        if checkpoint and copy_checkpoint(checkpoint, algo_name):
            results["success"].append(algo_name)
        else:
            results["failed"].append(algo_name)

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("训练完成汇总")
    print("=" * 70)
    print(f"成功: {len(results['success'])} 个")
    for a in results["success"]:
        print(f"  ✓ {a}")
    if results["failed"]:
        print(f"失败: {len(results['failed'])} 个")
        for a in results["failed"]:
            print(f"  ✗ {a}")


if __name__ == "__main__":
    main()
