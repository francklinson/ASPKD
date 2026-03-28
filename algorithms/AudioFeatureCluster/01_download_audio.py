"""
从 LibriSpeech 开源数据集下载三段典型人声素材
LibriSpeech 是一个开源的英语语音识别语料库，包含清晰的人声
"""
import os
import urllib.request
import tarfile
import shutil
from pathlib import Path


def download_file(url: str, output_path: str) -> bool:
    """下载文件并显示进度"""
    try:
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def extract_tar(tar_path: str, extract_to: str) -> bool:
    """解压tar.gz文件"""
    try:
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_to)
        print("Extraction complete")
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def setup_raw_data():
    """设置原始数据目录结构"""
    base_dir = Path("/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化")
    raw_dir = base_dir / "raw_audio"
    raw_dir.mkdir(exist_ok=True)
    return raw_dir


def download_librispeech_samples():
    """
    下载 LibriSpeech 测试集样本
    使用 dev-clean 子集，包含清晰的人声
    """
    raw_dir = setup_raw_data()
    
    # LibriSpeech dev-clean 子集 URL
    url = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
    tar_path = raw_dir / "dev-clean.tar.gz"
    
    # 下载数据
    if not tar_path.exists():
        if not download_file(url, str(tar_path)):
            print("Download failed. Please check your internet connection.")
            return False
    
    # 解压数据
    extract_dir = raw_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    
    if not any(extract_dir.iterdir()):
        if not extract_tar(str(tar_path), str(extract_dir)):
            return False
    
    # 选择三段不同的人声
    librispeech_dir = extract_dir / "LibriSpeech" / "dev-clean"
    
    if not librispeech_dir.exists():
        print(f"Expected directory not found: {librispeech_dir}")
        return False
    
    # 查找所有 .flac 文件
    flac_files = list(librispeech_dir.rglob("*.flac"))
    
    if len(flac_files) < 3:
        print(f"Not enough audio files found. Found {len(flac_files)} files.")
        return False
    
    # 选择三段来自不同说话人的音频（不同目录代表不同说话人）
    selected_files = []
    speakers = set()
    
    for flac_file in flac_files:
        # LibriSpeech 目录结构: .../speaker_id/chapter_id/file.flac
        speaker_id = flac_file.parent.parent.name
        
        if speaker_id not in speakers and len(selected_files) < 3:
            selected_files.append(flac_file)
            speakers.add(speaker_id)
    
    print(f"\nSelected {len(selected_files)} audio files from different speakers:")
    
    # 复制并重命名选中的文件
    final_files = []
    for i, src_file in enumerate(selected_files[:3], 1):
        dst_file = raw_dir / f"speaker_{i:02d}_original.wav"
        
        # 使用 ffmpeg 转换为 wav 格式
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", str(src_file),
            "-ar", "16000", "-ac", "1",  # 16kHz, 单声道
            str(dst_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  [{i}] {src_file.parent.parent.name} -> {dst_file.name}")
            final_files.append(dst_file)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {src_file}: {e}")
            continue
    
    # 清理临时文件
    if tar_path.exists():
        tar_path.unlink()
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    
    print(f"\nSuccessfully prepared {len(final_files)} audio files in: {raw_dir}")
    return len(final_files) == 3


def main():
    print("=" * 60)
    print("下载开源人声素材 (LibriSpeech)")
    print("=" * 60)
    
    success = download_librispeech_samples()
    
    if success:
        print("\n✓ 原始人声素材下载完成！")
        print("\n文件位置:")
        raw_dir = Path("/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化/raw_audio")
        for f in sorted(raw_dir.glob("*.wav")):
            print(f"  - {f.name}")
    else:
        print("\n✗ 下载失败，请检查网络连接")
        # 备用方案：创建模拟音频
        create_fallback_audio()


def create_fallback_audio():
    """如果下载失败，创建模拟音频用于测试"""
    print("\n创建备用测试音频...")
    
    import numpy as np
    import soundfile as sf
    
    raw_dir = Path("/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化/raw_audio")
    raw_dir.mkdir(exist_ok=True)
    
    sample_rate = 16000
    duration = 30  # 30秒
    
    for i in range(1, 4):
        # 创建模拟人声（不同频率的正弦波组合）
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 基础频率不同，模拟不同说话人
        base_freq = 150 + i * 30  # 180, 210, 240 Hz
        
        # 添加谐波和轻微频率变化模拟语音特征
        audio = np.sin(2 * np.pi * base_freq * t) * 0.5
        audio += np.sin(2 * np.pi * base_freq * 2 * t) * 0.25  # 二次谐波
        audio += np.sin(2 * np.pi * base_freq * 3 * t) * 0.125  # 三次谐波
        
        # 添加包络（模拟语音的停顿）
        envelope = np.ones_like(t)
        for pause_start in range(5, int(duration), 8):
            pause_idx_start = int(pause_start * sample_rate)
            pause_idx_end = int((pause_start + 0.5) * sample_rate)
            if pause_idx_end < len(envelope):
                envelope[pause_idx_start:pause_idx_end] = 0.1
        
        audio = audio * envelope
        
        # 添加轻微噪声
        noise = np.random.randn(len(audio)) * 0.01
        audio = audio + noise
        
        # 归一化
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        output_path = raw_dir / f"speaker_{i:02d}_original.wav"
        sf.write(output_path, audio, sample_rate)
        print(f"  Created: {output_path.name}")
    
    print("\n✓ 备用测试音频创建完成！")


if __name__ == "__main__":
    main()
