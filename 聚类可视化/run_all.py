#!/usr/bin/env python3
"""
完整实验流程控制脚本
一键运行：下载数据 -> 数据增强 -> 对比实验
"""
import sys
import subprocess
from pathlib import Path


def run_script(script_name: str, description: str) -> bool:
    """运行单个脚本并显示进度"""
    print("\n" + "="*70)
    print(f"步骤: {description}")
    print("="*70)
    
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"错误: 脚本不存在 {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=str(Path(__file__).parent)
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"脚本执行失败: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("音频异常检测对比实验 - 完整流程")
    print("="*70)
    print("\n本脚本将执行以下步骤:")
    print("  1. 从 LibriSpeech 下载三段典型人声素材")
    print("  2. 对音频进行多种处理，构造音频数据库")
    print("  3. 使用7种不同的特征提取器进行聚类对比")
    print("")
    
    # 确认执行
    response = input("是否开始执行? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("已取消")
        return
    
    # 步骤1: 下载数据
    if not run_script("01_download_audio.py", "下载开源人声素材"):
        print("\n✗ 步骤1失败，停止执行")
        return
    
    # 步骤2: 数据增强
    if not run_script("02_augment_audio.py", "构造音频数据库"):
        print("\n✗ 步骤2失败，停止执行")
        return
    
    # 步骤3: 对比实验
    if not run_script("03_compare_experiments.py", "运行对比实验"):
        print("\n✗ 步骤3失败")
        return
    
    print("\n" + "="*70)
    print("✓ 所有步骤执行完成!")
    print("="*70)
    print("\n输出文件:")
    base_dir = Path("/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化")
    print(f"  - 原始音频: {base_dir}/raw_audio/")
    print(f"  - 音频数据库: {base_dir}/audio_database/")
    print(f"  - 实验结果: {base_dir}/experiment_results/")
    print(f"  - 可视化结果: {base_dir}/anomaly_detection_result_*.png")


if __name__ == "__main__":
    main()
