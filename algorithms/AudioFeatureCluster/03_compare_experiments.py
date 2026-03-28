"""
对比实验脚本
使用不同的音频特征提取方法，对比聚类效果
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time
from datetime import datetime

# 导入主程序中的类
sys.path.insert(0, str(Path(__file__).parent))
from audio_anomaly_detection import (
    Config, FeatureExtractorFactory, AudioDataLoader,
    AnomalyDetector, AnomalyReportGenerator
)


class ExperimentRunner:
    """实验运行器"""
    
    # 定义所有要测试的特征提取器
    EXTRACTORS = [
        "hubert",
        "mfcc", 
        "mel",
        "mert",
        "wavlm",
        "xlsr-wav2vec2",
        "ast"
    ]
    
    def __init__(self, data_folder: str, output_dir: str = "experiment_results"):
        self.data_folder = Path(data_folder)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def run_single_experiment(self, extractor_type: str) -> Dict:
        """
        运行单个特征提取器的实验
        
        Returns:
            实验结果字典
        """
        print(f"\n{'='*60}")
        print(f"运行实验: {extractor_type.upper()}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # 初始化配置
            config = Config(feature_extractor_type=extractor_type)
            
            # 创建特征提取器
            print(f"\n[1/4] 创建特征提取器: {extractor_type}")
            feature_extractor = FeatureExtractorFactory.create_extractor(config)
            
            # 加载数据
            print(f"\n[2/4] 加载音频数据")
            data_loader = AudioDataLoader(str(self.data_folder))
            X, file_labels, file_paths = data_loader.load_data(feature_extractor)
            
            if len(X) == 0:
                print(f"✗ 未提取到特征，跳过此实验")
                return None
            
            # 训练模型
            print(f"\n[3/4] 训练聚类模型")
            detector = AnomalyDetector(config)
            detector.fit(X)
            
            # 预测
            print(f"\n[4/4] 预测与可视化")
            cluster_labels, outlier_mask = detector.predict(X)

            # 生成可视化（传入 file_labels 以显示真实标签）
            detector.visualize(X, outlier_mask, file_labels)

            # 计算统计信息（包括真实标签对比）
            elapsed_time = time.time() - start_time

            # 计算检测准确性
            true_anomaly_mask = np.array(['_anomaly' in label for label in file_labels])
            true_normal_mask = ~true_anomaly_mask
            correctly_detected_anomalies = np.sum(outlier_mask & true_anomaly_mask)
            false_positives = np.sum(outlier_mask & true_normal_mask)
            false_negatives = np.sum(~outlier_mask & true_anomaly_mask)

            result = {
                "extractor_type": extractor_type,
                "n_samples": len(X),
                "feature_dim": X.shape[1] if len(X) > 0 else 0,
                "n_normal": int(np.sum(true_normal_mask)),
                "n_anomaly": int(np.sum(true_anomaly_mask)),
                "n_clusters": config.n_clusters,
                "n_outliers_detected": int(np.sum(outlier_mask)),
                "correctly_detected": int(correctly_detected_anomalies),
                "false_positives": int(false_positives),
                "false_negatives": int(false_negatives),
                "detection_accuracy": float(correctly_detected_anomalies / np.sum(true_anomaly_mask)) if np.sum(true_anomaly_mask) > 0 else 0,
                "outlier_ratio": float(np.sum(outlier_mask) / len(X)) if len(X) > 0 else 0,
                "cluster_distribution": self._get_cluster_distribution(cluster_labels),
                "elapsed_time": elapsed_time,
                "file_labels": file_labels,
                "outlier_mask": outlier_mask.tolist(),
                "output_image": config.output_image,
                "status": "success"
            }

            print(f"\n✓ 实验完成!")
            print(f"  - 样本数: {result['n_samples']} (正常: {result['n_normal']}, 异常: {result['n_anomaly']})")
            print(f"  - 特征维度: {result['feature_dim']}")
            print(f"  - 检测到异常: {result['n_outliers_detected']}")
            print(f"  - 正确检测异常: {result['correctly_detected']}/{result['n_anomaly']} ({result['detection_accuracy']*100:.1f}%)")
            print(f"  - 误报: {result['false_positives']}, 漏报: {result['false_negatives']}")
            print(f"  - 耗时: {elapsed_time:.1f}秒")
            
            return result
            
        except Exception as e:
            print(f"\n✗ 实验失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "extractor_type": extractor_type,
                "status": "failed",
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }
    
    def _get_cluster_distribution(self, labels: np.ndarray) -> Dict[int, int]:
        """获取聚类分布"""
        unique, counts = np.unique(labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}
    
    def run_all_experiments(self) -> None:
        """运行所有实验"""
        print("\n" + "="*70)
        print("开始对比实验 - 评估不同特征提取器的聚类效果")
        print("="*70)
        print(f"数据目录: {self.data_folder}")
        print(f"输出目录: {self.output_dir}")
        print(f"测试的提取器: {', '.join(self.EXTRACTORS)}")
        
        # 运行每个实验
        for extractor in self.EXTRACTORS:
            result = self.run_single_experiment(extractor)
            if result:
                self.results[extractor] = result
        
        # 生成对比报告
        self._generate_comparison_report()
        
        print("\n" + "="*70)
        print("所有实验完成!")
        print(f"结果保存到: {self.output_dir}")
        print("="*70)
    
    def _generate_comparison_report(self) -> None:
        """生成对比报告"""
        print("\n" + "="*60)
        print("生成对比报告")
        print("="*60)
        
        # 1. 保存详细结果到JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"comparison_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✓ 详细结果保存: {json_path}")
        
        # 2. 生成对比表格
        self._generate_comparison_table()
        
        # 3. 生成可视化对比图
        self._generate_comparison_plots()
        
        # 4. 生成综合分析报告
        self._generate_analysis_report()
    
    def _generate_comparison_table(self) -> None:
        """生成对比表格"""
        table_path = self.output_dir / "comparison_table.txt"
        
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("特征提取器对比结果\n")
            f.write("="*80 + "\n\n")
            
            # 表头
            header = f"{'提取器':<12} {'样本':>6} {'正常':>5} {'异常':>5} {'检测':>5} {'正确':>5} {'准确率%':>8} {'耗时(s)':>8} {'状态':>6}\n"
            f.write(header)
            f.write("-"*80 + "\n")

            # 数据行
            for extractor in self.EXTRACTORS:
                if extractor in self.results:
                    r = self.results[extractor]
                    if r.get('status') == 'success':
                        line = (f"{extractor:<12} "
                               f"{r['n_samples']:>6} "
                               f"{r['n_normal']:>5} "
                               f"{r['n_anomaly']:>5} "
                               f"{r['n_outliers_detected']:>5} "
                               f"{r['correctly_detected']:>5} "
                               f"{r['detection_accuracy']*100:>8.1f} "
                               f"{r['elapsed_time']:>8.1f} "
                               f"{'✓':>6}\n")
                    else:
                        line = f"{extractor:<12} {'-':>6} {'-':>5} {'-':>5} {'-':>5} {'-':>5} {'-':>8} {r.get('elapsed_time', 0):>8.1f} {'✗':>6}\n"
                    f.write(line)
            
            f.write("="*80 + "\n")
        
        # 同时打印到控制台
        with open(table_path, 'r') as f:
            print(f.read())
        
        print(f"✓ 对比表格保存: {table_path}")
    
    def _generate_comparison_plots(self) -> None:
        """生成对比可视化图"""
        
        # 过滤成功的实验
        success_results = {k: v for k, v in self.results.items() 
                          if v.get('status') == 'success'}
        
        if not success_results:
            print("没有成功的实验，跳过可视化")
            return
        
        extractors = list(success_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Feature Extractor Comparison', fontsize=16, fontweight='bold')
        
        # 1. 异常检测准确率对比
        ax1 = axes[0, 0]
        accuracies = [success_results[e]['detection_accuracy']*100 for e in extractors]
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(extractors)))
        bars1 = ax1.bar(extractors, accuracies, color=colors)
        ax1.set_ylabel('Detection Accuracy (%)')
        ax1.set_title('Anomaly Detection Accuracy')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{acc:.0f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. 处理时间对比
        ax2 = axes[0, 1]
        times = [success_results[e]['elapsed_time'] for e in extractors]
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(extractors)))
        bars2 = ax2.bar(extractors, times, color=colors)
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Processing Time')
        ax2.tick_params(axis='x', rotation=45)
        for bar, t in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{t:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # 3. 特征维度对比
        ax3 = axes[1, 0]
        dims = [success_results[e]['feature_dim'] for e in extractors]
        colors = plt.cm.Purples(np.linspace(0.4, 0.8, len(extractors)))
        bars3 = ax3.bar(extractors, dims, color=colors)
        ax3.set_ylabel('Feature Dimension')
        ax3.set_title('Feature Dimension Size')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_yscale('log')
        for bar, dim in zip(bars3, dims):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{dim}', ha='center', va='bottom', fontsize=9)
        
        # 4. 检测统计对比（正确检测 vs 误报）
        ax4 = axes[1, 1]
        x = np.arange(len(extractors))
        width = 0.25
        correct = [success_results[e]['correctly_detected'] for e in extractors]
        false_pos = [success_results[e]['false_positives'] for e in extractors]
        false_neg = [success_results[e]['false_negatives'] for e in extractors]

        bars4a = ax4.bar(x - width, correct, width, label='Correct', color='green', alpha=0.7)
        bars4b = ax4.bar(x, false_pos, width, label='False Pos', color='orange', alpha=0.7)
        bars4c = ax4.bar(x + width, false_neg, width, label='False Neg', color='red', alpha=0.7)
        ax4.set_ylabel('Count')
        ax4.set_title('Detection Statistics')
        ax4.set_xticks(x)
        ax4.set_xticklabels(extractors, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plot_path = self.output_dir / "comparison_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ 对比图表保存: {plot_path}")
        plt.close()
    
    def _generate_analysis_report(self) -> None:
        """生成分析报告"""
        report_path = self.output_dir / "analysis_report.md"
        
        success_results = {k: v for k, v in self.results.items() 
                          if v.get('status') == 'success'}
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 音频特征提取器对比分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. 实验概述
            f.write("## 1. 实验概述\n\n")
            f.write(f"- **数据目录**: `{self.data_folder}`\n")
            f.write(f"- **成功实验数**: {len(success_results)}/{len(self.EXTRACTORS)}\n")
            f.write(f"- **测试的特征提取器**: {', '.join(self.EXTRACTORS)}\n\n")
            
            # 2. 详细结果
            f.write("## 2. 详细结果\n\n")
            for extractor, result in success_results.items():
                f.write(f"### {extractor.upper()}\n\n")
                f.write(f"- 样本数: {result['n_samples']} (正常: {result['n_normal']}, 异常: {result['n_anomaly']})\n")
                f.write(f"- 特征维度: {result['feature_dim']}\n")
                f.write(f"- 检测到异常: {result['n_outliers_detected']}\n")
                f.write(f"- 正确检测异常: {result['correctly_detected']}/{result['n_anomaly']} ({result['detection_accuracy']*100:.1f}%)\n")
                f.write(f"- 误报: {result['false_positives']}, 漏报: {result['false_negatives']}\n")
                f.write(f"- 聚类分布: {result['cluster_distribution']}\n")
                f.write(f"- 处理时间: {result['elapsed_time']:.2f}秒\n")
                f.write(f"- 输出图像: `{result['output_image']}`\n\n")
            
            # 3. 对比分析
            f.write("## 3. 对比分析\n\n")
            
            if success_results:
                # 最快的方法
                fastest = min(success_results.items(), key=lambda x: x[1]['elapsed_time'])
                f.write(f"### 3.1 处理速度\n\n")
                f.write(f"- **最快**: {fastest[0].upper()} ({fastest[1]['elapsed_time']:.2f}秒)\n")
                
                # 最慢的方法
                slowest = max(success_results.items(), key=lambda x: x[1]['elapsed_time'])
                f.write(f"- **最慢**: {slowest[0].upper()} ({slowest[1]['elapsed_time']:.2f}秒)\n\n")
                
                # 异常检测
                f.write(f"### 3.2 异常检测准确性\n\n")
                # 按检测准确率排序
                sorted_by_accuracy = sorted(success_results.items(), key=lambda x: x[1]['detection_accuracy'], reverse=True)
                best = sorted_by_accuracy[0]
                worst = sorted_by_accuracy[-1]
                f.write(f"- **最佳检测准确率**: {best[0].upper()} ({best[1]['detection_accuracy']*100:.1f}%)\n")
                f.write(f"- **最低检测准确率**: {worst[0].upper()} ({worst[1]['detection_accuracy']*100:.1f}%)\n\n")
                
                f.write(f"检测准确率排名:\n")
                for i, (ext, res) in enumerate(sorted_by_accuracy[:5], 1):
                    f.write(f"  {i}. {ext.upper()}: {res['detection_accuracy']*100:.1f}% "
                           f"(正确 {res['correctly_detected']}/{res['n_anomaly']})\n")
                f.write(f"\n")
                
                # 特征维度
                f.write(f"### 3.3 特征维度\n\n")
                highest_dim = max(success_results.items(), key=lambda x: x[1]['feature_dim'])
                lowest_dim = min(success_results.items(), key=lambda x: x[1]['feature_dim'])
                f.write(f"- **最高维度**: {highest_dim[0].upper()} ({highest_dim[1]['feature_dim']})\n")
                f.write(f"- **最低维度**: {lowest_dim[0].upper()} ({lowest_dim[1]['feature_dim']})\n\n")
            
            # 4. 结论与建议
            f.write("## 4. 结论与建议\n\n")
            f.write("### 4.1 方法特点总结\n\n")
            f.write("| 方法 | 类型 | 特点 | 适用场景 |\n")
            f.write("|------|------|------|----------|\n")
            f.write("| MFCC | 传统特征 | 计算快，维度低 | 实时应用，资源受限 |\n")
            f.write("| Mel Spectrogram | 传统特征 | 频谱信息丰富 | 音频分类，音乐分析 |\n")
            f.write("| HuBERT | 深度学习 | 语音表示强大 | 语音识别，说话人识别 |\n")
            f.write("| WavLM | 深度学习 | 鲁棒性高 | 噪声环境，远场语音 |\n")
            f.write("| XLSR-Wav2Vec2 | 深度学习 | 多语言支持 | 多语言场景 |\n")
            f.write("| MERT | 深度学习 | 音乐理解好 | 音乐分析，旋律识别 |\n")
            f.write("| AST | 深度学习 | 频谱+Transformer | 通用音频理解 |\n\n")
            
            f.write("### 4.2 实验建议\n\n")
            f.write("1. **根据应用场景选择**: 如果是语音任务，优先选择HuBERT/WavLM；如果是音乐任务，选择MERT\n")
            f.write("2. **考虑计算资源**: MFCC和Mel Spectrogram计算最快，适合实时应用\n")
            f.write("3. **数据质量**: 深度学习模型需要更多数据才能发挥优势\n")
            f.write("4. **异常定义**: 不同特征对'异常'的定义不同，需要根据具体任务调整阈值\n\n")
            
            f.write("---\n")
            f.write("*报告由自动对比实验系统生成*\n")
        
        print(f"✓ 分析报告保存: {report_path}")


def main():
    """主函数"""
    # 数据目录
    base_dir = Path("/home/zhouchenghao/PycharmProjects/ASD_for_SPK/聚类可视化")
    
    # 可以选择使用全部数据或单个说话人的数据
    # 这里使用全部数据
    data_folder = base_dir / "audio_database"
    
    if not data_folder.exists():
        print(f"错误: 数据目录不存在: {data_folder}")
        print("请先运行 01_download_audio.py 和 02_augment_audio.py 准备数据")
        return
    
    # 创建实验运行器并运行
    runner = ExperimentRunner(
        data_folder=str(data_folder),
        output_dir=str(base_dir / "experiment_results")
    )
    
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
