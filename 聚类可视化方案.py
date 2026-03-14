import os
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import (
    HubertModel,
    AutoFeatureExtractor,
    AutoModel,
    EncodecModel,
    AutoProcessor,
    WavLMModel,
    Wav2Vec2FeatureExtractor,
    ASTModel,
    ClapModel, ClapProcessor, ClapAudioModel
)


class Config:
    """配置类，管理所有参数"""

    def __init__(self, feature_extractor_type: str = "hubert"):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 模型配置
        self.model_name = "facebook/hubert-base-ls960"

        # 音频处理参数
        self.sample_rate = 24000  # 大多数模型要求16k采样率
        self.segment_duration = 10  # 每个片段的时长（秒）

        # 聚类参数
        self.n_clusters = 2  # 聚类数量
        self.outlier_threshold_percentile = 95  # 异常阈值百分位数

        # 特征提取器类型
        self.feature_extractor_type = feature_extractor_type

        # 可视化参数
        self.tsne_perplexity = 30  # 默认perplexity值
        self.tsne_max_iter = 1000
        self.output_image = f"anomaly_detection_result_{self.feature_extractor_type}.png"


class BaseFeatureExtractor(ABC):
    """特征提取器基类"""

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取特征

        参数:
            audio_path: 音频文件路径

        返回:
            特征数组，形状为 (N_samples, Feature_dim)
        """
        pass


class HubertFeatureExtractor(BaseFeatureExtractor):
    """基于HuBert的音频特征提取器"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.feature_extractor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载预训练模型"""
        print(f"Loading model: {self.config.model_name}...")
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.config.model_name)
            self.model = HubertModel.from_pretrained(self.config.model_name).to(self.config.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取HuBert特征

        参数:
            audio_path: 音频文件路径

        返回:
            特征数组，形状为 (N_samples, Feature_dim)
        """
        # 加载音频
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.config.sample_rate,
            mono=True
        )

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.config.sample_rate)
        features_list = []

        # 滑动窗口切分音频
        for i in range(0, len(waveform), segment_samples):
            segment = waveform[i: i + segment_samples]

            # 丢弃太短的片段
            if len(segment) < segment_samples:
                continue

            # 转换为模型输入格式
            inputs = self.feature_extractor(
                segment,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            ).input_values.to(self.config.device)

            # 提取特征
            with torch.no_grad():
                hidden_states = self.model(inputs).last_hidden_state

            # 对时间维度求平均
            segment_feature = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            features_list.append(segment_feature)

        return np.array(features_list)


class ASTFeatureExtractor(BaseFeatureExtractor):
    """基于Audio Spectrogram Transformer (AST) 的音频特征提取器"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.feature_extractor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载预训练模型"""
        print("Loading AST model...")
        try:
            model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = ASTModel.from_pretrained(model_name).to(self.config.device)
            self.model.eval()
            print("AST model loaded successfully.")
        except Exception as e:
            print(f"Error loading AST model: {e}")
            raise

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取AST特征
        AST需要输入频谱图，因此采样率通常固定为16000，且输入长度会被处理

        参数:
            audio_path: 音频文件路径

        返回:
            特征数组，形状为 (N_samples, Feature_dim)
        """
        # AST默认使用16000Hz采样率
        target_sr = 16000
        waveform, orig_sr = librosa.load(audio_path, sr=target_sr, mono=True)

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * target_sr)
        features_list = []

        # 滑动窗口切分音频
        for i in range(0, len(waveform), segment_samples):
            segment = waveform[i: i + segment_samples]

            # 丢弃太短的片段
            if len(segment) < segment_samples:
                continue

            # 转换为模型输入格式
            inputs = self.feature_extractor(
                segment,
                sampling_rate=target_sr,
                return_tensors="pt"
            ).input_values.to(self.config.device)

            # 提取特征
            with torch.no_grad():
                hidden_states = self.model(inputs).last_hidden_state

            # 对时间维度求平均
            # AST输出形状通常是 (Batch, Sequence, Hidden_Dim)
            segment_feature = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            features_list.append(segment_feature)

        return np.array(features_list)



class MFCCFeatureExtractor(BaseFeatureExtractor):
    """基于MFCC的音频特征提取器"""

    def __init__(self, config: Config):
        super().__init__(config)

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取MFCC特征

        参数:
            audio_path: 音频文件路径

        返回:
            特征数组，形状为 (N_samples, Feature_dim)
        """
        # 加载音频
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.config.sample_rate,
            mono=True
        )

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.config.sample_rate)
        features_list = []

        # 滑动窗口切分音频
        for i in range(0, len(waveform), segment_samples):
            segment = waveform[i: i + segment_samples]

            # 丢弃太短的片段
            if len(segment) < segment_samples:
                continue

            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=segment,
                sr=self.config.sample_rate,
                n_mfcc=13  # 通常使用13个MFCC系数
            )

            # 对时间维度求平均
            segment_feature = np.mean(mfcc, axis=1)
            features_list.append(segment_feature)

        return np.array(features_list)


class MelSpectrogramExtractor(BaseFeatureExtractor):
    """基于Mel频谱图的音频特征提取器"""

    def __init__(self, config: Config):
        super().__init__(config)

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取Mel频谱图特征

        参数:
            audio_path: 音频文件路径

        返回:
            特征数组，形状为 (N_samples, Feature_dim)
        """
        # 加载音频
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.config.sample_rate,
            mono=True
        )

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.config.sample_rate)
        features_list = []

        # 滑动窗口切分音频
        for i in range(0, len(waveform), segment_samples):
            segment = waveform[i: i + segment_samples]

            # 丢弃太短的片段
            if len(segment) < segment_samples:
                continue

            # 提取Mel频谱图
            mel_spec = librosa.feature.melspectrogram(
                y=segment,
                sr=self.config.sample_rate,
                n_mels=128  # Mel频带数量
            )

            # 转换为dB刻度
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # 对时间维度求平均
            segment_feature = np.mean(mel_spec_db, axis=1)
            features_list.append(segment_feature)

        return np.array(features_list)


class MERTFeatureExtractor(BaseFeatureExtractor):
    """基于MERT的音频特征提取器"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.feature_extractor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载预训练模型"""
        print("Loading MERT model...")
        try:
            # 使用AutoFeatureExtractor和AutoModel加载MERT模型
            model_name = "m-a-p/MERT-v1-330M"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.config.device)
            self.model.eval()
            print("MERT model loaded successfully.")
        except Exception as e:
            print(f"Error loading MERT model: {e}")
            raise

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取MERT特征

        参数:
            audio_path: 音频文件路径

        返回:
            特征数组，形状为 (N_samples, Feature_dim)
        """
        # 加载音频
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.config.sample_rate,
            mono=True
        )

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.config.sample_rate)
        features_list = []

        # 滑动窗口切分音频
        for i in range(0, len(waveform), segment_samples):
            segment = waveform[i: i + segment_samples]

            # 丢弃太短的片段
            if len(segment) < segment_samples:
                continue

            # 转换为模型输入格式
            inputs = self.feature_extractor(
                segment,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            ).input_values.to(self.config.device)

            # 提取特征
            with torch.no_grad():
                hidden_states = self.model(inputs).last_hidden_state

            # 对时间维度求平均
            segment_feature = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            features_list.append(segment_feature)

        return np.array(features_list)


class WavLMFeatureExtractor(BaseFeatureExtractor):
    """基于WavLM的音频特征提取器"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.feature_extractor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载预训练模型"""
        print("Loading WavLM model...")
        try:
            # 使用AutoFeatureExtractor和AutoModel加载WavLM模型
            model_name = "microsoft/wavlm-base"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = WavLMModel.from_pretrained(model_name).to(self.config.device)
            self.model.eval()
            print("WavLM model loaded successfully.")
        except Exception as e:
            print(f"Error loading WavLM model: {e}")
            raise

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取WavLM特征

        参数:
            audio_path: 音频文件路径

        返回:
            特征数组，形状为 (N_samples, Feature_dim)
        """
        # 加载音频
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.config.sample_rate,
            mono=True
        )

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.config.sample_rate)
        features_list = []

        # 滑动窗口切分音频
        for i in range(0, len(waveform), segment_samples):
            segment = waveform[i: i + segment_samples]

            # 丢弃太短的片段
            if len(segment) < segment_samples:
                continue

            # 转换为模型输入格式
            inputs = self.feature_extractor(
                segment,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            ).input_values.to(self.config.device)

            # 提取特征
            with torch.no_grad():
                hidden_states = self.model(inputs).last_hidden_state

            # 对时间维度求平均
            segment_feature = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            features_list.append(segment_feature)

        return np.array(features_list)


class XLSRWav2Vec2FeatureExtractor(BaseFeatureExtractor):
    """基于XLSR-Wav2Vec2的音频特征提取器"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.feature_extractor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载预训练模型"""
        print("Loading XLSR-Wav2Vec2 model...")
        try:
            # 使用AutoFeatureExtractor和AutoModel加载XLSR-Wav2Vec2模型
            model_name = "facebook/wav2vec2-large-xlsr-53"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.config.device)
            self.model.eval()
            print("XLSR-Wav2Vec2 model loaded successfully.")
        except Exception as e:
            print(f"Error loading XLSR-Wav2Vec2 model: {e}")
            raise

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取XLSR-Wav2Vec2特征

        参数:
            audio_path: 音频文件路径

        返回:
            特征数组，形状为 (N_samples, Feature_dim)
        """
        # 加载音频
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.config.sample_rate,
            mono=True
        )

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.config.sample_rate)
        features_list = []

        # 滑动窗口切分音频
        for i in range(0, len(waveform), segment_samples):
            segment = waveform[i: i + segment_samples]

            # 丢弃太短的片段
            if len(segment) < segment_samples:
                continue

            # 转换为模型输入格式
            inputs = self.feature_extractor(
                segment,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            ).input_values.to(self.config.device)

            # 提取特征
            with torch.no_grad():
                hidden_states = self.model(inputs).last_hidden_state

            # 对时间维度求平均
            segment_feature = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            features_list.append(segment_feature)

        return np.array(features_list)


class FeatureExtractorFactory:
    """特征提取器工厂类"""

    @staticmethod
    def create_extractor(config: Config):
        """
        根据配置创建特征提取器

        参数:
            config: 配置对象

        返回:
            特征提取器实例
        """
        if config.feature_extractor_type == "hubert":
            return HubertFeatureExtractor(config)
        elif config.feature_extractor_type == "mfcc":
            return MFCCFeatureExtractor(config)
        elif config.feature_extractor_type == "mel":
            return MelSpectrogramExtractor(config)
        elif config.feature_extractor_type == "mert":
            return MERTFeatureExtractor(config)
        elif config.feature_extractor_type == "wavlm":
            return WavLMFeatureExtractor(config)
        elif config.feature_extractor_type == "xlsr-wav2vec2":
            return XLSRWav2Vec2FeatureExtractor(config)
        elif config.feature_extractor_type == "ast":
            return ASTFeatureExtractor(config)
        else:
            raise ValueError(f"Unknown feature extractor type: {config.feature_extractor_type}")


class AudioDataLoader:
    """音频数据加载器，处理文件夹中的音频文件"""

    def __init__(self, audio_folder: str):
        self.audio_folder = audio_folder
        self.audio_files = self._get_audio_files()

    def _get_audio_files(self) -> List[str]:
        """获取文件夹中的所有音频文件"""
        if not os.path.exists(self.audio_folder):
            os.makedirs(self.audio_folder)
            print(f"提示: 文件夹 '{self.audio_folder}' 不存在。请将你的音频文件放入该文件夹中。")
            return []

        audio_files = [
            f for f in os.listdir(self.audio_folder)
            if f.endswith(('.mp3', '.wav', '.flac'))
        ]

        if not audio_files:
            print("未找到音频文件。请检查路径。")

        return audio_files

    def load_data(self, feature_extractor: BaseFeatureExtractor) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        加载音频数据并提取特征

        参数:
            feature_extractor: 特征提取器实例

        返回:
            特征数组, 文件标签列表, 文件路径列表
        """
        all_features = []
        file_labels = []
        file_paths = []

        print("Extracting features...")
        for file_name in self.audio_files:
            path = os.path.join(self.audio_folder, file_name)
            print(f"Processing: {file_name}")

            try:
                features = feature_extractor.extract_features(path)
                if len(features) > 0:
                    all_features.append(features)
                    file_labels.extend([file_name] * len(features))
                    file_paths.extend([path] * len(features))
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        if not all_features:
            raise ValueError("未能提取到任何特征。")

        X = np.vstack(all_features)
        print(f"Total feature vectors extracted: {X.shape}")

        return X, file_labels, file_paths


class AnomalyDetector:
    """异常检测器，使用聚类算法检测异常"""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.kmeans = None
        self.tsne = None
        self.threshold = None

    def fit(self, X: np.ndarray) -> None:
        """
        训练模型

        参数:
            X: 特征数组
        """
        # 标准化数据
        X_scaled = self.scaler.fit_transform(X)

        # 聚类
        print("Clustering...")
        self.kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=42,
            n_init=10
        )
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # 计算阈值
        centers = self.kmeans.cluster_centers_
        distances = np.linalg.norm(X_scaled - centers[cluster_labels], axis=1)
        self.threshold = np.percentile(distances, self.config.outlier_threshold_percentile)

        # 动态调整perplexity值，确保它小于样本数量
        n_samples = X.shape[0]
        perplexity = min(self.config.tsne_perplexity, n_samples - 1)
        # 确保perplexity至少为5
        perplexity = max(perplexity, 5)
        print(f"Adjusting perplexity to {perplexity} (n_samples={n_samples})")

        # 初始化t-SNE
        self.tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            max_iter=self.config.tsne_max_iter
        )

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测异常

        参数:
            X: 特征数组

        返回:
            聚类标签, 异常掩码
        """
        X_scaled = self.scaler.transform(X)
        cluster_labels = self.kmeans.predict(X_scaled)
        centers = self.kmeans.cluster_centers_
        distances = np.linalg.norm(X_scaled - centers[cluster_labels], axis=1)
        outlier_mask = distances > self.threshold

        return cluster_labels, outlier_mask

    def visualize(self, X: np.ndarray, outlier_mask: np.ndarray) -> None:
        """
        可视化结果

        参数:
            X: 特征数组
            outlier_mask: 异常掩码
        """
        print("Generating visualization...")
        X_scaled = self.scaler.transform(X)
        X_tsne = self.tsne.fit_transform(X_scaled)

        plt.figure(figsize=(12, 8))

        # 绘制正常点
        normal_indices = ~outlier_mask
        plt.scatter(
            X_tsne[normal_indices, 0], X_tsne[normal_indices, 1],
            c='blue', label='Normal Segments', alpha=0.5, s=10
        )

        # 绘制异常点
        outlier_indices = outlier_mask
        plt.scatter(
            X_tsne[outlier_indices, 0], X_tsne[outlier_indices, 1],
            c='red', label='Detected Anomalies (Outliers)', alpha=0.8, edgecolors='k', s=50
        )

        # 添加图例和标题
        plt.title(
            f"Audio Anomaly Detection Visualization ({self.config.feature_extractor_type.upper()} + t-SNE)\n"
            f"Threshold: {self.config.outlier_threshold_percentile}th percentile",
            fontsize=14
        )
        plt.legend()
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True, linestyle='--', alpha=0.6)

        # 保存图片
        plt.savefig(self.config.output_image, dpi=300)
        print(f"Visualization saved to '{self.config.output_image}'")


class AnomalyReportGenerator:
    """异常报告生成器"""

    @staticmethod
    def generate_report(file_labels: List[str], outlier_mask: np.ndarray) -> None:
        """
        生成异常报告

        参数:
            file_labels: 文件标签列表
            outlier_mask: 异常掩码
        """
        print("\n--- Detailed Anomaly Report ---")
        outlier_file_names = np.array(file_labels)[outlier_mask]
        unique_outlier_files, counts = np.unique(outlier_file_names, return_counts=True)

        if len(unique_outlier_files) > 0:
            print("Files containing potential anomalies:")
            for file, count in zip(unique_outlier_files, counts):
                print(f"- {file}: {count} segments detected as anomaly")
        else:
            print("No anomalies detected.")


def main():
    # 初始化配置，可以选择使用不同的特征提取器:
    # "hubert", "mfcc", "mel", "mert","wavlm", "xlsr-wav2vec2", "ast"
    config = Config(feature_extractor_type="mert")  # 可以改为其他特征提取器

    # 使用工厂创建特征提取器
    feature_extractor = FeatureExtractorFactory.create_extractor(config)

    # 初始化数据加载器
    audio_folder = "/home/zhouchenghao/PycharmProjects/ASD_for_SPK/原始数据/标记后/auto_test/8k/good"
    data_loader = AudioDataLoader(audio_folder)

    # 加载数据并提取特征
    X, file_labels, file_paths = data_loader.load_data(feature_extractor)

    # 初始化异常检测器
    detector = AnomalyDetector(config)

    # 训练模型
    detector.fit(X)

    # 预测异常
    _, outlier_mask = detector.predict(X)

    # 可视化结果
    detector.visualize(X, outlier_mask)

    # 生成报告
    AnomalyReportGenerator.generate_report(file_labels, outlier_mask)

    print(f"Detected {np.sum(outlier_mask)} potential outlier segments out of {len(X)}.")


if __name__ == "__main__":
    main()
