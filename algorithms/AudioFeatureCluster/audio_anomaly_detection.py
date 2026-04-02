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
        self.sample_rate = 16000  # 语音模型通常使用16k采样率
        self.segment_duration = 3   # 每个片段的时长（秒），确保能处理短时音频
        self.use_first_segment_only = True  # 每段音频只取第一个片段，确保各说话人样本数一致

        # 聚类参数
        self.n_clusters = 3  # 聚类数量（对应3个说话人）
        self.outlier_threshold_percentile = 90  # 异常阈值百分位数

        # 特征提取器类型
        self.feature_extractor_type = feature_extractor_type

        # 可视化参数
        self.tsne_perplexity = 30  # 默认perplexity值
        self.tsne_max_iter = 1000
        # 输出路径设置为 experiment_results 目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, "experiment_results")
        os.makedirs(results_dir, exist_ok=True)
        self.output_image = os.path.join(results_dir, f"anomaly_detection_result_{self.feature_extractor_type}.png")


class BaseFeatureExtractor(ABC):
    """特征提取器基类"""

    def __init__(self, config: Config):
        self.config = config

    def _preprocess(self, audio: np.ndarray, sr: int, top_db: float = 30) -> np.ndarray:
        """
        音频预处理：去除开头和结尾的静音区域

        参数:
            audio: 原始音频数组
            sr: 采样率
            top_db: 静音阈值（低于该值视为静音），默认30dB

        返回:
            去除首尾静音后的音频数组
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed

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
    """基于HuBert的音频特征提取器 - 使用16kHz采样率"""

    # HuBERT模型使用16kHz
    SAMPLE_RATE = 16000

    def __init__(self, config: Config):
        super().__init__(config)
        self.feature_extractor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载预训练模型"""
        print(f"Loading model: {self.config.model_name} (采样率: {self.SAMPLE_RATE}Hz)...")
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
        # 加载音频 - 使用模型特定的采样率
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.SAMPLE_RATE,
            mono=True
        )

        # 去除首尾静音
        waveform = self._preprocess(waveform, self.SAMPLE_RATE)

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.SAMPLE_RATE)
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
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt"
            ).input_values.to(self.config.device)

            # 提取特征
            with torch.no_grad():
                hidden_states = self.model(inputs).last_hidden_state

            # 对时间维度求平均
            segment_feature = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            features_list.append(segment_feature)

            # 如果配置为只取第一个片段，则跳出循环
            if self.config.use_first_segment_only:
                break

        return np.array(features_list)


class ASTFeatureExtractor(BaseFeatureExtractor):
    """基于Audio Spectrogram Transformer (AST) 的音频特征提取器 - 使用16kHz采样率"""

    # AST模型使用16kHz
    SAMPLE_RATE = 16000

    def __init__(self, config: Config):
        super().__init__(config)
        self.feature_extractor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载预训练模型"""
        print(f"Loading AST model (采样率: {self.SAMPLE_RATE}Hz)...")
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

        参数:
            audio_path: 音频文件路径

        返回:
            特征数组，形状为 (N_samples, Feature_dim)
        """
        # 加载音频 - 使用模型特定的采样率
        waveform, orig_sr = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)

        # 去除首尾静音
        waveform = self._preprocess(waveform, self.SAMPLE_RATE)

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.SAMPLE_RATE)
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
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt"
            ).input_values.to(self.config.device)

            # 提取特征
            with torch.no_grad():
                hidden_states = self.model(inputs).last_hidden_state

            # 对时间维度求平均
            segment_feature = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            features_list.append(segment_feature)

            # 如果配置为只取第一个片段，则跳出循环
            if self.config.use_first_segment_only:
                break

        return np.array(features_list)


class MFCCFeatureExtractor(BaseFeatureExtractor):
    """基于MFCC的音频特征提取器 - 使用16kHz采样率（语音标准）"""

    # MFCC通常使用16kHz作为语音处理标准
    SAMPLE_RATE = 16000

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
        # 加载音频 - 使用模型特定的采样率
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.SAMPLE_RATE,
            mono=True
        )

        # 去除首尾静音
        waveform = self._preprocess(waveform, self.SAMPLE_RATE)

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.SAMPLE_RATE)
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
                sr=self.SAMPLE_RATE,
                n_mfcc=40  # 使用40个MFCC系数
            )

            # 对时间维度求平均
            segment_feature = np.mean(mfcc, axis=1)
            features_list.append(segment_feature)

            # 如果配置为只取第一个片段，则跳出循环
            if self.config.use_first_segment_only:
                break

        return np.array(features_list)


class MelSpectrogramExtractor(BaseFeatureExtractor):
    """基于Mel频谱图的音频特征提取器 - 使用16kHz采样率"""

    # Mel频谱图通常使用16kHz作为语音处理标准
    SAMPLE_RATE = 16000

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
        # 加载音频 - 使用模型特定的采样率
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.SAMPLE_RATE,
            mono=True
        )

        # 去除首尾静音
        waveform = self._preprocess(waveform, self.SAMPLE_RATE)

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.SAMPLE_RATE)
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
                sr=self.SAMPLE_RATE,
                n_mels=128  # Mel频带数量
            )

            # 转换为dB刻度
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # 对时间维度求平均
            segment_feature = np.mean(mel_spec_db, axis=1)
            features_list.append(segment_feature)

            # 如果配置为只取第一个片段，则跳出循环
            if self.config.use_first_segment_only:
                break

        return np.array(features_list)


class MERTFeatureExtractor(BaseFeatureExtractor):
    """基于MERT的音频特征提取器 - 使用24kHz采样率（音乐模型）"""

    # MERT模型使用24kHz（针对音乐优化）
    SAMPLE_RATE = 24000

    def __init__(self, config: Config):
        super().__init__(config)
        self.feature_extractor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载预训练模型"""
        print(f"Loading MERT model (采样率: {self.SAMPLE_RATE}Hz)...")
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
        # 加载音频 - 使用模型特定的采样率
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.SAMPLE_RATE,
            mono=True
        )

        # 去除首尾静音
        waveform = self._preprocess(waveform, self.SAMPLE_RATE)

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.SAMPLE_RATE)
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
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt"
            ).input_values.to(self.config.device)

            # 提取特征
            with torch.no_grad():
                hidden_states = self.model(inputs).last_hidden_state

            # 对时间维度求平均
            segment_feature = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            features_list.append(segment_feature)

            # 如果配置为只取第一个片段，则跳出循环
            if self.config.use_first_segment_only:
                break

        return np.array(features_list)


class WavLMFeatureExtractor(BaseFeatureExtractor):
    """基于WavLM的音频特征提取器 - 使用16kHz采样率"""

    # WavLM模型使用16kHz
    SAMPLE_RATE = 16000

    def __init__(self, config: Config):
        super().__init__(config)
        self.feature_extractor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载预训练模型"""
        print(f"Loading WavLM model (采样率: {self.SAMPLE_RATE}Hz)...")
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
        # 加载音频 - 使用模型特定的采样率
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.SAMPLE_RATE,
            mono=True
        )

        # 去除首尾静音
        waveform = self._preprocess(waveform, self.SAMPLE_RATE)

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.SAMPLE_RATE)
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
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt"
            ).input_values.to(self.config.device)

            # 提取特征
            with torch.no_grad():
                hidden_states = self.model(inputs).last_hidden_state

            # 对时间维度求平均
            segment_feature = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            features_list.append(segment_feature)

            # 如果配置为只取第一个片段，则跳出循环
            if self.config.use_first_segment_only:
                break

        return np.array(features_list)


class XLSRWav2Vec2FeatureExtractor(BaseFeatureExtractor):
    """基于XLSR-Wav2Vec2的音频特征提取器 - 使用16kHz采样率"""

    # XLSR-Wav2Vec2模型使用16kHz
    SAMPLE_RATE = 16000

    def __init__(self, config: Config):
        super().__init__(config)
        self.feature_extractor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载预训练模型"""
        print(f"Loading XLSR-Wav2Vec2 model (采样率: {self.SAMPLE_RATE}Hz)...")
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
        # 加载音频 - 使用模型特定的采样率
        waveform, orig_sr = librosa.load(
            audio_path,
            sr=self.SAMPLE_RATE,
            mono=True
        )

        # 去除首尾静音
        waveform = self._preprocess(waveform, self.SAMPLE_RATE)

        # 计算每个片段的样本数
        segment_samples = int(self.config.segment_duration * self.SAMPLE_RATE)
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
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt"
            ).input_values.to(self.config.device)

            # 提取特征
            with torch.no_grad():
                hidden_states = self.model(inputs).last_hidden_state

            # 对时间维度求平均
            segment_feature = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            features_list.append(segment_feature)

            # 如果配置为只取第一个片段，则跳出循环
            if self.config.use_first_segment_only:
                break

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
        """递归获取文件夹中的所有音频文件（包括子目录）"""
        if not os.path.exists(self.audio_folder):
            os.makedirs(self.audio_folder)
            print(f"提示: 文件夹 '{self.audio_folder}' 不存在。请将你的音频文件放入该文件夹中。")
            return []

        audio_files = []
        for root, dirs, files in os.walk(self.audio_folder):
            for f in files:
                if f.endswith(('.mp3', '.wav', '.flac')):
                    # 返回相对路径，便于后续使用
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, self.audio_folder)
                    audio_files.append(rel_path)

        if not audio_files:
            print(f"未找到音频文件。请检查路径: {self.audio_folder}")
        else:
            print(f"找到 {len(audio_files)} 个音频文件")

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
            # file_name 现在可能是相对路径（包含子目录）
            path = os.path.join(self.audio_folder, file_name)
            # 显示文件名（不含路径）以便阅读
            display_name = os.path.basename(file_name)
            print(f"Processing: {display_name}")

            try:
                features = feature_extractor.extract_features(path)
                if len(features) > 0:
                    all_features.append(features)
                    file_labels.extend([display_name] * len(features))
                    file_paths.extend([path] * len(features))
            except Exception as e:
                print(f"Error processing {display_name}: {e}")

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

        # 动态调整perplexity值，确保它严格小于样本数量
        n_samples = X.shape[0]
        # t-SNE要求: perplexity < n_samples
        max_perplexity = max(1, n_samples - 1)  # 至少留出1的余地
        perplexity = min(self.config.tsne_perplexity, max_perplexity)
        # 确保perplexity至少为1（最小有效值）
        perplexity = max(perplexity, 1)
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

    def visualize(self, X: np.ndarray, outlier_mask: np.ndarray, file_labels: List[str] = None, use_3d: bool = False) -> None:
        """
        可视化聚类结果 - 三种说话人用不同颜色，检测到的异常用黑色圆圈标记

        参数:
            X: 特征数组
            outlier_mask: 异常掩码（检测到的异常）
            file_labels: 文件标签列表（用于识别说话人）
            use_3d: 是否使用3D可视化
        """
        print(f"Generating visualization... (3D={use_3d})")
        X_scaled = self.scaler.transform(X)
        
        if use_3d:
            # 3D t-SNE
            print("  Using 3D t-SNE projection...")
            tsne_3d = TSNE(n_components=3, random_state=42, perplexity=self.tsne.perplexity, max_iter=self.config.tsne_max_iter)
            X_tsne = tsne_3d.fit_transform(X_scaled)
            print(f"  3D projection shape: {X_tsne.shape}")
        else:
            X_tsne = self.tsne.fit_transform(X_scaled)
            print(f"  2D projection shape: {X_tsne.shape}")

        # 创建图
        if use_3d:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(14, 11))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(12, 10))

        # 根据文件名识别说话人
        if file_labels:
            # 将 file_labels 转换为 numpy 数组以便索引
            file_labels_arr = np.array(file_labels)
            
            # 调试：打印前10个file_labels看看实际格式
            print(f"  Sample file_labels (first 10): {file_labels[:10]}")
            print(f"  Total file_labels: {len(file_labels)}")
            
            # 检查所有唯一的文件名前缀
            unique_prefixes = set()
            for label in file_labels[:50]:  # 检查前50个
                prefix = label.split('_')[0] + '_' + label.split('_')[1] if '_' in label else label
                unique_prefixes.add(prefix)
            print(f"  Unique speaker prefixes found: {unique_prefixes}")
            
            # 识别三种说话人 - 检查文件名中是否包含speaker_id
            speaker_01_mask = np.array(['speaker_01' in label for label in file_labels])
            speaker_02_mask = np.array(['speaker_02' in label for label in file_labels])
            speaker_03_mask = np.array(['speaker_03' in label for label in file_labels])
            
            # 调试输出
            print(f"  Speaker 01 samples: {np.sum(speaker_01_mask)}")
            print(f"  Speaker 02 samples: {np.sum(speaker_02_mask)}")
            print(f"  Speaker 03 samples: {np.sum(speaker_03_mask)}")
            
            # 三种颜色代表三个说话人
            colors = {
                'speaker_01': '#E63946',  # 红色
                'speaker_02': '#457B9D',  # 蓝色
                'speaker_03': '#2A9D8F'   # 绿色
            }

            # 绘制每个说话人的样本
            if use_3d:
                if np.sum(speaker_01_mask) > 0:
                    ax.scatter(
                        X_tsne[speaker_01_mask, 0], X_tsne[speaker_01_mask, 1], X_tsne[speaker_01_mask, 2],
                        c=colors['speaker_01'], label=f'Speaker 01 ({np.sum(speaker_01_mask)})',
                        alpha=0.6, s=40, edgecolors='white', linewidth=0.5
                    )
                if np.sum(speaker_02_mask) > 0:
                    ax.scatter(
                        X_tsne[speaker_02_mask, 0], X_tsne[speaker_02_mask, 1], X_tsne[speaker_02_mask, 2],
                        c=colors['speaker_02'], label=f'Speaker 02 ({np.sum(speaker_02_mask)})',
                        alpha=0.6, s=40, edgecolors='white', linewidth=0.5
                    )
                if np.sum(speaker_03_mask) > 0:
                    ax.scatter(
                        X_tsne[speaker_03_mask, 0], X_tsne[speaker_03_mask, 1], X_tsne[speaker_03_mask, 2],
                        c=colors['speaker_03'], label=f'Speaker 03 ({np.sum(speaker_03_mask)})',
                        alpha=0.6, s=40, edgecolors='white', linewidth=0.5
                    )
            else:
                if np.sum(speaker_01_mask) > 0:
                    ax.scatter(
                        X_tsne[speaker_01_mask, 0], X_tsne[speaker_01_mask, 1],
                        c=colors['speaker_01'], label=f'Speaker 01 ({np.sum(speaker_01_mask)})',
                        alpha=0.6, s=40, edgecolors='white', linewidth=0.5
                    )
                if np.sum(speaker_02_mask) > 0:
                    ax.scatter(
                        X_tsne[speaker_02_mask, 0], X_tsne[speaker_02_mask, 1],
                        c=colors['speaker_02'], label=f'Speaker 02 ({np.sum(speaker_02_mask)})',
                        alpha=0.6, s=40, edgecolors='white', linewidth=0.5
                    )
                if np.sum(speaker_03_mask) > 0:
                    ax.scatter(
                        X_tsne[speaker_03_mask, 0], X_tsne[speaker_03_mask, 1],
                        c=colors['speaker_03'], label=f'Speaker 03 ({np.sum(speaker_03_mask)})',
                        alpha=0.6, s=40, edgecolors='white', linewidth=0.5
                    )
        else:
            # 如果没有文件名标签，使用聚类标签着色
            cluster_labels = self.kmeans.predict(X_scaled)
            if use_3d:
                ax.scatter(
                    X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                    c=cluster_labels, cmap='tab10', alpha=0.6, s=40,
                    edgecolors='white', linewidth=0.5
                )
            else:
                ax.scatter(
                    X_tsne[:, 0], X_tsne[:, 1],
                    c=cluster_labels, cmap='tab10', alpha=0.6, s=40,
                    edgecolors='white', linewidth=0.5
                )

        # 绘制检测到的异常（黑色圆圈标记）
        if np.sum(outlier_mask) > 0:
            if use_3d:
                ax.scatter(
                    X_tsne[outlier_mask, 0], X_tsne[outlier_mask, 1], X_tsne[outlier_mask, 2],
                    facecolors='none', edgecolors='black', linewidths=2.5,
                    label=f'Detected Anomalies ({np.sum(outlier_mask)})', s=120
                )
            else:
                ax.scatter(
                    X_tsne[outlier_mask, 0], X_tsne[outlier_mask, 1],
                    facecolors='none', edgecolors='black', linewidths=2.5,
                    label=f'Detected Anomalies ({np.sum(outlier_mask)})', s=120
                )

        # 设置标题和标签
        dim_str = "3D" if use_3d else "2D"
        ax.set_title(
            f'Audio Anomaly Detection - {self.config.feature_extractor_type.upper()} + t-SNE ({dim_str})\n'
            f'Total: {len(X)} samples | Detected: {np.sum(outlier_mask)} outliers',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
        if use_3d:
            ax.set_zlabel("t-SNE Dimension 3", fontsize=12)

        # 添加图例
        ax.legend(loc='best', fontsize=10, framealpha=0.9)

        # 添加网格
        if use_3d:
            ax.grid(True, linestyle='--', alpha=0.3)
        else:
            ax.grid(True, linestyle='--', alpha=0.4)

        # 设置背景色
        if not use_3d:
            ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')

        plt.tight_layout()
        plt.savefig(self.config.output_image, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to '{self.config.output_image}'")
        plt.close()

    def visualize_interactive(self, X: np.ndarray, outlier_mask: np.ndarray, file_labels: List[str] = None) -> str:
        """
        生成交互式 3D 可视化 (使用 Plotly)
        
        返回:
            HTML 文件路径
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("[警告] Plotly 未安装，无法生成交互式图表")
            return None
        
        print("Generating interactive 3D visualization with Plotly...")
        X_scaled = self.scaler.transform(X)
        
        # 3D t-SNE
        tsne_3d = TSNE(n_components=3, random_state=42, perplexity=self.tsne.perplexity, max_iter=self.config.tsne_max_iter)
        X_tsne = tsne_3d.fit_transform(X_scaled)
        print(f"  3D projection shape: {X_tsne.shape}")
        
        # 创建 Plotly 图表
        fig = go.Figure()
        
        # 根据文件名识别说话人或使用聚类标签
        if file_labels:
            # 识别说话人
            speaker_01_mask = np.array(['speaker_01' in label for label in file_labels])
            speaker_02_mask = np.array(['speaker_02' in label for label in file_labels])
            speaker_03_mask = np.array(['speaker_03' in label for label in file_labels])
            
            colors = {
                'speaker_01': '#E63946',
                'speaker_02': '#457B9D',
                'speaker_03': '#2A9D8F'
            }
            
            # 添加每个说话人的散点
            if np.sum(speaker_01_mask) > 0:
                fig.add_trace(go.Scatter3d(
                    x=X_tsne[speaker_01_mask, 0],
                    y=X_tsne[speaker_01_mask, 1],
                    z=X_tsne[speaker_01_mask, 2],
                    mode='markers',
                    name=f'Speaker 01 ({np.sum(speaker_01_mask)})',
                    marker=dict(size=6, color=colors['speaker_01'], opacity=0.8),
                    text=[file_labels[i] for i in np.where(speaker_01_mask)[0]],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
                ))
            
            if np.sum(speaker_02_mask) > 0:
                fig.add_trace(go.Scatter3d(
                    x=X_tsne[speaker_02_mask, 0],
                    y=X_tsne[speaker_02_mask, 1],
                    z=X_tsne[speaker_02_mask, 2],
                    mode='markers',
                    name=f'Speaker 02 ({np.sum(speaker_02_mask)})',
                    marker=dict(size=6, color=colors['speaker_02'], opacity=0.8),
                    text=[file_labels[i] for i in np.where(speaker_02_mask)[0]],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
                ))
            
            if np.sum(speaker_03_mask) > 0:
                fig.add_trace(go.Scatter3d(
                    x=X_tsne[speaker_03_mask, 0],
                    y=X_tsne[speaker_03_mask, 1],
                    z=X_tsne[speaker_03_mask, 2],
                    mode='markers',
                    name=f'Speaker 03 ({np.sum(speaker_03_mask)})',
                    marker=dict(size=6, color=colors['speaker_03'], opacity=0.8),
                    text=[file_labels[i] for i in np.where(speaker_03_mask)[0]],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
                ))
            
            # 处理未分类样本
            unclassified_mask = ~(speaker_01_mask | speaker_02_mask | speaker_03_mask)
            if np.sum(unclassified_mask) > 0:
                fig.add_trace(go.Scatter3d(
                    x=X_tsne[unclassified_mask, 0],
                    y=X_tsne[unclassified_mask, 1],
                    z=X_tsne[unclassified_mask, 2],
                    mode='markers',
                    name=f'Others ({np.sum(unclassified_mask)})',
                    marker=dict(size=6, color='#888888', opacity=0.6),
                    text=[file_labels[i] for i in np.where(unclassified_mask)[0]],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
                ))
        else:
            # 使用聚类标签着色
            cluster_labels = self.kmeans.predict(X_scaled)
            unique_labels = np.unique(cluster_labels)
            
            for label in unique_labels:
                mask = cluster_labels == label
                fig.add_trace(go.Scatter3d(
                    x=X_tsne[mask, 0],
                    y=X_tsne[mask, 1],
                    z=X_tsne[mask, 2],
                    mode='markers',
                    name=f'Cluster {label} ({np.sum(mask)})',
                    marker=dict(size=6, opacity=0.8),
                    hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
                ))
        
        # 添加异常点（黑色边框）
        if np.sum(outlier_mask) > 0:
            fig.add_trace(go.Scatter3d(
                x=X_tsne[outlier_mask, 0],
                y=X_tsne[outlier_mask, 1],
                z=X_tsne[outlier_mask, 2],
                mode='markers',
                name=f'Anomalies ({np.sum(outlier_mask)})',
                marker=dict(size=10, color='rgba(0,0,0,0)', opacity=1,
                           line=dict(color='black', width=3)),
                text=[file_labels[i] if file_labels else f'Sample {i}' for i in np.where(outlier_mask)[0]],
                hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ))
        
        # 设置布局
        fig.update_layout(
            title=dict(
                text=f'Audio Anomaly Detection - {self.config.feature_extractor_type.upper()} + t-SNE (Interactive 3D)<br>'
                     f'<sup>Total: {len(X)} samples | Detected: {np.sum(outlier_mask)} outliers</sup>',
                x=0.5, xanchor='center'
            ),
            scene=dict(
                xaxis_title='t-SNE Dimension 1',
                yaxis_title='t-SNE Dimension 2',
                zaxis_title='t-SNE Dimension 3',
                aspectmode='cube'
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        
        # 保存为 HTML
        html_path = self.config.output_image.replace('.png', '_interactive.html')
        fig.write_html(html_path, include_plotlyjs='cdn')
        print(f"Interactive visualization saved to '{html_path}'")
        
        return html_path


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

    # 初始化数据加载器 - 使用音频数据库目录
    # 也可以指定原始数据目录: "./raw_audio" 或完整数据库: "./audio_database"
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(base_dir, "audio_database")
    
    # 如果音频数据库不存在，尝试使用原始数据
    if not os.path.exists(audio_folder):
        raw_folder = os.path.join(base_dir, "raw_audio")
        if os.path.exists(raw_folder):
            audio_folder = raw_folder
        else:
            # 使用默认路径
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

    # 可视化结果（传入 file_labels 以显示真实标签）
    detector.visualize(X, outlier_mask, file_labels)

    # 生成详细报告
    print("\n" + "="*70)
    print("异常检测分析报告")
    print("="*70)

    # 统计检测结果与真实标签的对比
    true_anomaly_mask = np.array(['_anomaly' in label for label in file_labels])
    true_normal_mask = ~true_anomaly_mask

    # 计算检测准确性
    correctly_detected_anomalies = np.sum(outlier_mask & true_anomaly_mask)
    correctly_detected_normal = np.sum(~outlier_mask & true_normal_mask)
    false_positives = np.sum(outlier_mask & true_normal_mask)  # 正常被误判为异常
    false_negatives = np.sum(~outlier_mask & true_anomaly_mask)  # 异常被误判为正常

    print(f"\n检测结果统计:")
    print(f"  总样本数: {len(X)}")
    print(f"  真实正常样本: {np.sum(true_normal_mask)}")
    print(f"  真实异常样本: {np.sum(true_anomaly_mask)}")
    print(f"  检测到异常: {np.sum(outlier_mask)}")
    print(f"\n检测准确性:")
    print(f"  正确检测异常: {correctly_detected_anomalies}/{np.sum(true_anomaly_mask)} "
          f"({correctly_detected_anomalies/np.sum(true_anomaly_mask)*100:.1f}%)")
    print(f"  正确识别正常: {correctly_detected_normal}/{np.sum(true_normal_mask)} "
          f"({correctly_detected_normal/np.sum(true_normal_mask)*100:.1f}%)")
    print(f"  误报 (正常→异常): {false_positives}")
    print(f"  漏报 (异常→正常): {false_negatives}")

    AnomalyReportGenerator.generate_report(file_labels, outlier_mask)

    print(f"\nDetected {np.sum(outlier_mask)} potential outlier segments out of {len(X)}.")


if __name__ == "__main__":
    main()
