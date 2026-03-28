#!/usr/bin/env python3
"""
快速测试单个特征提取器
用于快速验证和调试
"""
import sys
from pathlib import Path

# 导入主程序
sys.path.insert(0, str(Path(__file__).parent))
from audio_anomaly_detection import (
    Config, FeatureExtractorFactory, AudioDataLoader,
    AnomalyDetector, AnomalyReportGenerator
)
import numpy as np


def test_single_extractor(extractor_type: str = "mfcc", data_folder: str = None):
    """
    测试单个特征提取器
    
    Args:
        extractor_type: 特征提取器类型
        data_folder: 数据文件夹路径，默认为 audio_database
    """
    base_dir = Path("/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化")
    
    if data_folder is None:
        data_folder = base_dir / "audio_database"
    else:
        data_folder = Path(data_folder)
    
    print("="*60)
    print(f"快速测试: {extractor_type.upper()}")
    print("="*60)
    print(f"数据目录: {data_folder}")
    
    if not data_folder.exists():
        print(f"\n错误: 数据目录不存在!")
        print("请先运行: python 01_download_audio.py")
        print("          python 02_augment_audio.py")
        return
    
    # 配置
    config = Config(feature_extractor_type=extractor_type)
    
    # 创建特征提取器
    print(f"\n[1/4] 创建特征提取器...")
    try:
        feature_extractor = FeatureExtractorFactory.create_extractor(config)
    except Exception as e:
        print(f"创建失败: {e}")
        return
    
    # 加载数据
    print(f"\n[2/4] 加载音频数据...")
    try:
        data_loader = AudioDataLoader(str(data_folder))
        X, file_labels, file_paths = data_loader.load_data(feature_extractor)
        print(f"  提取了 {len(X)} 个特征向量，维度: {X.shape[1]}")
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    # 训练模型
    print(f"\n[3/4] 训练聚类模型...")
    try:
        detector = AnomalyDetector(config)
        detector.fit(X)
    except Exception as e:
        print(f"训练失败: {e}")
        return
    
    # 预测与可视化
    print(f"\n[4/4] 预测与可视化...")
    try:
        cluster_labels, outlier_mask = detector.predict(X)
        detector.visualize(X, outlier_mask, file_labels)

        # 生成报告
        AnomalyReportGenerator.generate_report(file_labels, outlier_mask)

        # 统计检测结果与真实标签的对比
        true_anomaly_mask = np.array(['_anomaly' in label for label in file_labels])
        true_normal_mask = ~true_anomaly_mask

        correctly_detected = np.sum(outlier_mask & true_anomaly_mask)
        false_positives = np.sum(outlier_mask & true_normal_mask)

        print(f"\n{'='*60}")
        print(f"✓ 测试完成!")
        print(f"  - 总样本: {len(X)} (正常: {np.sum(true_normal_mask)}, 异常: {np.sum(true_anomaly_mask)})")
        print(f"  - 检测到异常: {np.sum(outlier_mask)}")
        print(f"  - 正确检测异常: {correctly_detected}/{np.sum(true_anomaly_mask)} "
              f"({correctly_detected/np.sum(true_anomaly_mask)*100:.1f}%)")
        print(f"  - 误报数: {false_positives}")
        print(f"  - 输出图像: {config.output_image}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='快速测试单个特征提取器')
    parser.add_argument(
        'extractor',
        nargs='?',
        default='mfcc',
        choices=['hubert', 'mfcc', 'mel', 'mert', 'wavlm', 'xlsr-wav2vec2', 'ast'],
        help='特征提取器类型 (默认: mfcc)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='数据文件夹路径 (默认: audio_database)'
    )
    
    args = parser.parse_args()
    
    test_single_extractor(args.extractor, args.data)


if __name__ == "__main__":
    main()
