"""
音频数据增强脚本 - 为异常检测场景设计

数据分布设计:
- 正常样本 (约80%): 原始音频 + 轻微变化（音量、轻微噪声、轻微混响）
- 异常样本 (约20%): 明显异常（变速、变调、重噪声、削波、失真）

预期聚类效果:
- 3个清晰的聚类中心（对应3个说话人）
- 异常样本偏离各自的聚类中心
"""
import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Callable
import random


class AudioAugmentor:
    """音频数据增强器"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def add_noise(self, audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """添加白噪声，SNR越大噪声越小"""
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
        return audio + noise

    def add_reverb(self, audio: np.ndarray, reverberance: float = 0.5) -> np.ndarray:
        """添加混响效果"""
        reverb_length = int(self.sample_rate * 0.5)
        reverb_audio = np.copy(audio)

        for i in range(1, min(reverb_length, len(audio))):
            delay_samples = i
            if delay_samples < len(audio):
                decay = (reverberance ** (i / self.sample_rate * 10))
                reverb_audio[delay_samples:] += audio[:-delay_samples] * decay * 0.3

        max_val = np.max(np.abs(reverb_audio))
        if max_val > 0:
            reverb_audio = reverb_audio / max_val * np.max(np.abs(audio))

        return reverb_audio

    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """时间拉伸/压缩（变速不变调）- 用于生成异常"""
        result = librosa.effects.time_stretch(audio, rate=rate)
        # 保持长度一致
        if len(result) > len(audio):
            result = result[:len(audio)]
        elif len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)), mode='constant')
        return result

    def pitch_shift(self, audio: np.ndarray, n_steps: float = 0.0) -> np.ndarray:
        """音高变换（变调不变速）- 用于生成异常"""
        return librosa.effects.pitch_shift(
            audio, sr=self.sample_rate, n_steps=n_steps
        )

    def apply_eq(self, audio: np.ndarray, eq_type: str = "telephone") -> np.ndarray:
        """应用均衡器效果 - 用于生成异常"""
        from scipy import signal

        if eq_type == "telephone":
            # 电话音质 (300-3400 Hz带通)
            sos = signal.butter(4, [300, 3400], 'bandpass', fs=self.sample_rate, output='sos')
        elif eq_type == "muffled":
            # 闷声（低通滤波）
            sos = signal.butter(4, 1000, 'lowpass', fs=self.sample_rate, output='sos')
        else:
            return audio

        return signal.sosfilt(sos, audio)

    def clip_distortion(self, audio: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """添加削波失真 - 用于生成异常"""
        return np.clip(audio, -threshold, threshold)

    def volume_change(self, audio: np.ndarray, gain_db: float = 0.0) -> np.ndarray:
        """音量调整 - 轻微变化属正常范围"""
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear


def time_shift(audio: np.ndarray, shift_samples: int) -> np.ndarray:
    """时域循环偏移 - 用于生成正常样本的变体"""
    return np.roll(audio, shift_samples)


def create_audio_database():
    """
    创建适合异常检测的音频数据库 - 大量正常样本 + 少量异常样本

    数据分布:
    - 每个说话人 110个样本
    - 正常样本 (100个, ~90%): 原始 + 时域偏移 + 轻微变化
    - 异常样本 (10个, ~10%): 明显异常
    """
    base_dir = Path("/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化")
    raw_dir = base_dir / "raw_audio"
    database_dir = base_dir / "audio_database"

    # 清空并重建数据库目录
    if database_dir.exists():
        import shutil
        shutil.rmtree(database_dir)
    database_dir.mkdir(exist_ok=True)

    # 获取原始音频文件
    raw_files = sorted(raw_dir.glob("speaker_*_original.wav"))

    if len(raw_files) < 3:
        print(f"错误: 需要3个原始音频文件，但只找到 {len(raw_files)} 个")
        return False

    augmentor = AudioAugmentor(sample_rate=16000)

    # 目标：每个说话人100个正常样本，10个异常样本
    TARGET_NORMAL = 100
    TARGET_ANOMALY = 10

    print("=" * 70)
    print("构造音频数据库 - 异常检测场景")
    print("=" * 70)
    print(f"\n数据分布策略:")
    print(f"  - 正常样本: {TARGET_NORMAL} 个/说话人 (约90%)")
    print(f"  - 异常样本: {TARGET_ANOMALY} 个/说话人 (约10%)")
    print(f"\n正常样本生成方式:")
    print(f"  - 时域偏移 (主要方式): 循环偏移音频，保持内容相似性")
    print(f"  - 轻微音量变化")
    print(f"  - 轻微噪声和混响")
    print(f"\n预期聚类效果:")
    print(f"  - 3个清晰的聚类中心（对应3个说话人）")
    print(f"  - 大量正常样本形成密集的聚类区域")
    print(f"  - 异常样本明显偏离聚类中心")

    total_normal = 0
    total_anomaly = 0

    for raw_file in raw_files:
        speaker_name = raw_file.stem  # e.g., "speaker_01_original"
        speaker_id = speaker_name.split("_")[1]  # e.g., "01"

        # 加载原始音频
        audio, sr = librosa.load(raw_file, sr=16000)

        # 去除首尾静音
        audio, _ = librosa.effects.trim(audio, top_db=30)

        print(f"\n{'='*70}")
        print(f"处理 {raw_file.name}:")
        print(f"{'='*70}")

        # 创建说话人目录
        speaker_dir = database_dir / f"speaker_{speaker_id}"
        speaker_dir.mkdir(parents=True, exist_ok=True)

        # ========== 生成正常样本 (100个) ==========
        print(f"\n  [正常样本 - 生成 {TARGET_NORMAL} 个]")

        # 1. 原始音频副本 (5个)
        for i in range(5):
            output_name = f"{speaker_name.replace('_original', '')}_orig_{i:03d}_normal.wav"
            output_path = speaker_dir / output_name
            sf.write(output_path, audio, 16000)
            total_normal += 1
        print(f"    ✓ 原始副本: 5个")

        # 2. 时域偏移样本 (70个) - 循环偏移，保持内容相似
        np.random.seed(42 + int(speaker_id))  # 固定随机种子保证可重复
        for i in range(70):
            # 随机偏移量（0.1秒到2秒之间）
            shift_sec = np.random.uniform(0.1, 2.0)
            shift_samples = int(shift_sec * 16000)
            shifted_audio = time_shift(audio, shift_samples)

            output_name = f"{speaker_name.replace('_original', '')}_shift_{i:03d}_normal.wav"
            output_path = speaker_dir / output_name
            sf.write(output_path, shifted_audio, 16000)
            total_normal += 1
        print(f"    ✓ 时域偏移: 70个")

        # 3. 轻微音量变化 (10个)
        gain_values = [-6, -4, -2, -1, 1, 2, 4, 6, -3, 3]
        for i, gain in enumerate(gain_values):
            processed = augmentor.volume_change(audio, gain)
            max_val = np.max(np.abs(processed))
            if max_val > 1.0:
                processed = processed / max_val * 0.95

            output_name = f"{speaker_name.replace('_original', '')}_vol_{i:03d}_normal.wav"
            output_path = speaker_dir / output_name
            sf.write(output_path, processed, 16000)
            total_normal += 1
        print(f"    ✓ 音量变化: 10个")

        # 4. 轻微噪声 (10个)
        snr_values = [30, 28, 26, 24, 22, 25, 27, 29, 23, 21]
        for i, snr in enumerate(snr_values):
            processed = augmentor.add_noise(audio, snr_db=snr)

            output_name = f"{speaker_name.replace('_original', '')}_noise_{i:03d}_normal.wav"
            output_path = speaker_dir / output_name
            sf.write(output_path, processed, 16000)
            total_normal += 1
        print(f"    ✓ 轻微噪声: 10个")

        # 5. 轻微混响 (5个)
        reverb_values = [0.1, 0.15, 0.2, 0.12, 0.18]
        for i, rev in enumerate(reverb_values):
            processed = augmentor.add_reverb(audio, reverberance=rev)
            max_val = np.max(np.abs(processed))
            if max_val > 1.0:
                processed = processed / max_val * 0.95

            output_name = f"{speaker_name.replace('_original', '')}_reverb_{i:03d}_normal.wav"
            output_path = speaker_dir / output_name
            sf.write(output_path, processed, 16000)
            total_normal += 1
        print(f"    ✓ 轻微混响: 5个")

        # ========== 生成异常样本 (10个) ==========
        print(f"\n  [异常样本 - 生成 {TARGET_ANOMALY} 个]")

        anomaly_configs = [
            ("speed_up_1.3x", lambda x: augmentor.time_stretch(x, 1.3)),
            ("slow_down_0.7x", lambda x: augmentor.time_stretch(x, 0.7)),
            ("pitch_plus5", lambda x: augmentor.pitch_shift(x, 5)),
            ("pitch_minus5", lambda x: augmentor.pitch_shift(x, -5)),
            ("pitch_plus7", lambda x: augmentor.pitch_shift(x, 7)),
            ("noise_heavy_snr8", lambda x: augmentor.add_noise(x, snr_db=8)),
            ("noise_heavy_snr5", lambda x: augmentor.add_noise(x, snr_db=5)),
            ("clip_0.5", lambda x: augmentor.clip_distortion(x, 0.5)),
            ("clip_0.4", lambda x: augmentor.clip_distortion(x, 0.4)),
            ("telephone_eq", lambda x: augmentor.apply_eq(x, "telephone")),
        ]

        for i, (name, func) in enumerate(anomaly_configs):
            processed = func(audio)
            max_val = np.max(np.abs(processed))
            if max_val > 1.0:
                processed = processed / max_val * 0.95

            output_name = f"{speaker_name.replace('_original', '')}_{name}_anomaly.wav"
            output_path = speaker_dir / output_name
            sf.write(output_path, processed, 16000)
            total_anomaly += 1
            print(f"    ✗ {name}")

    print(f"\n{'='*70}")
    print(f"✓ 音频数据库创建完成！")
    print(f"{'='*70}")
    print(f"  总样本数: {total_normal + total_anomaly}")
    print(f"  - 正常样本: {total_normal} ({total_normal/(total_normal+total_anomaly)*100:.1f}%)")
    print(f"  - 异常样本: {total_anomaly} ({total_anomaly/(total_normal+total_anomaly)*100:.1f}%)")
    print(f"  每个说话人: {TARGET_NORMAL}正常 + {TARGET_ANOMALY}异常 = {TARGET_NORMAL + TARGET_ANOMALY}个")
    print(f"  数据库位置: {database_dir}")

    # 打印目录结构
    print(f"\n数据库结构:")
    for speaker_dir in sorted(database_dir.iterdir()):
        if speaker_dir.is_dir():
            normal_count = len(list(speaker_dir.glob("*_normal.wav")))
            anomaly_count = len(list(speaker_dir.glob("*_anomaly.wav")))
            print(f"  {speaker_dir.name}/")
            print(f"    - 正常样本: {normal_count} 个")
            print(f"    - 异常样本: {anomaly_count} 个")

    return True


def main():
    create_audio_database()


if __name__ == "__main__":
    main()
