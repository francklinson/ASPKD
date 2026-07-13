#!/usr/bin/env python3
"""单算法快速训练 — 不设置 RANK/WORLD_SIZE 让 ADer 自动禁用 DDP"""
import os, sys, shutil, subprocess, time, re

PROJECT_ROOT = "/home/zhouchenghao/PycharmProjects/ASD_for_SPK"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "public_dataset", "mvtec")
SAVED = os.path.join(PROJECT_ROOT, "models", "saved")
PRETRAINED = os.path.join(PROJECT_ROOT, "models", "pre_trained")
PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
ADER_DIR = os.path.join(PROJECT_ROOT, "algorithms", "ADer")

# 需要训练的算法: (算法名, 配置目录名, 配置文件前缀)
ADER_TODO = [
    ("vitad", "vitad", "vitad"),
    ("unad", "uniad", "uniad"),      # 配置目录是 uniad 不是 unad
    ("cflow", "cflow", "cflow"),
    ("pyramidflow", "pyramidflow", "pyramidflow"),
    ("destseg", "destseg", "destseg"),
    ("realnet", "realnet", "realnet"),
    ("rdpp", "rdpp", "rdpp"),
]

for algo, cfg_dir, cfg_prefix in ADER_TODO:
    dst = os.path.join(PRETRAINED, f"{algo}_best.pth")
    if os.path.exists(dst) and os.path.getsize(dst) > 1000000:
        print(f"SKIP {algo} (already exists)")
        continue

    cfg = f"configs/benchmark/{cfg_dir}/{cfg_prefix}_256_100e.py"
    cfg_abs = os.path.join(ADER_DIR, cfg)
    if not os.path.exists(cfg_abs):
        print(f"SKIP {algo} (no config: {cfg})")
        continue

    print(f"\n=== Training {algo} ({cfg_dir}) ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}/algorithms:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # 关键：不设置 RANK/WORLD_SIZE，让 ADer 自动禁用 DDP

    # 清理可能残留的环境变量
    for var in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(var, None)

    cmd = [
        PYTHON, "run.py", "-c", cfg, "-m", "train",
        "data.root=../../data/public_dataset/mvtec",
        "trainer.checkpoint=../../models/saved",
        "data.cls_names=['bottle']",
        "epoch_full=5",
        "trainer.test_start_epoch=999",
        "trainer.test_per_epoch=999",
        "batch_train=8",
        "trainer.data.batch_size_per_gpu_test=1",
        "use_adeval=False",
    ]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ADER_DIR, env=env, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0
        out = p.stdout + p.stderr
        ok = "finish training" in out
        print(f"  {'OK' if ok else 'FAIL'} ({elapsed:.0f}s, rc={p.returncode})")
        if not ok:
            # Show relevant error lines
            error_lines = [l for l in out.split("\n") if "Error" in l or "Traceback" in l or "FAIL" in l]
            for l in error_lines[-5:]:
                print(f"  {l.strip()[:150]}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (600s)")

    # Find checkpoint
    found = None
    for root, dirs, files in os.walk(SAVED):
        for d in dirs:
            if algo.lower() in d.lower() or cfg_dir.lower() in d.lower():
                np = os.path.join(root, d, "net.pth")
                if os.path.exists(np) and os.path.getsize(np) > 1000000:
                    mtime = os.path.getmtime(np)
                    if time.time() - mtime < 3600:  # 最近1小时内
                        found = np
                        break
        if found:
            break

    if found:
        shutil.copy2(found, dst)
        sz = os.path.getsize(dst) / 1e6
        print(f"  SAVED {algo}_best.pth ({sz:.0f}MB)")
    else:
        print(f"  NO CHECKPOINT for {algo}")

print("\nDone!")
