import librosa
import os
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from BaseASD.ASDBase import AnomalySoundDetectionBase
from BaseASD.VAE.data import file_to_vector_array
from BaseASD.VAE.model import Autoencoder
from BaseASD.VAE.common import yaml_load

# 获取CPU核心数量
cpu_count = os.cpu_count()
# 设置LOKY的CPU核心数量
os.environ['LOKY_MAX_CPU_COUNT'] = str(int(cpu_count // 2))


class VAEInterface(AnomalySoundDetectionBase):
    def __init__(self):
        super(VAEInterface, self).__init__()
        self.const_param = yaml_load()
        self.input_size = self.const_param['input_size']
        self.hidden1 = self.const_param['hidden1']
        self.hidden2 = self.const_param['hidden2']
        self.hidden3 = self.const_param['hidden3']
        self.hidden4 = self.const_param['hidden4']
        self.latent_length = self.const_param['latent_length']
        self.threshold = self.const_param['anomaly_threshold']

    def load_audio_data(self, file_path):
        """
        加载音频文件并计算其潜在表示。
        """
        train_signal, samplerate = librosa.load(file_path, sr=None, mono=False)
        train_frqe = file_to_vector_array(train_signal, samplerate, n_mels=128, frames=5, n_fft=1024, hop_length=512)
        train_frqe = torch.tensor(train_frqe)
        inputs = train_frqe.to("cpu").to(dtype=torch.float32)

        # 加载保存的autoencoder参数
        param = torch.load(r"BaseASD/VAE/clustering/spk/epoch/0/autoencoder_id_01.pth")
        # autoencoder = torch.load(r"BaseASD/VAE/clustering/spk/epoch/0/autoencoder_id_01.pth")
        autoencoder = Autoencoder(self.input_size, self.hidden1, self.hidden2, self.hidden3, self.hidden4,
                                  self.latent_length)
        autoencoder.load_state_dict(param)
        autoencoder.eval()
        outputs, latent, latent_mean, latent_logvar = autoencoder(inputs.float())
        latent_com = latent
        latent_representation = latent_com.detach()
        return latent_representation

    def judge_is_normal(self, file_path):
        """
        判断音频文件是否异常。
        """
        # 加载音频文件的潜在表示
        latent_representation = self.load_audio_data(file_path)

        # 加载训练数据的潜在表示
        all_traning_latent = np.load(r"BaseASD/VAE/clustering/spk/epoch/95/all_traning_latentid_01.npy")
        all_traning_latent = all_traning_latent.reshape(-1, 30)
        # 计算音频文件的潜在表示与训练数据的对数似然值
        gmm = GaussianMixture()
        gmm.fit(all_traning_latent)

        latent_representation = latent_representation.reshape(-1, 30)

        llh1 = gmm.score_samples(all_traning_latent)
        llh2 = gmm.score_samples(latent_representation)

        llh1_llh2 = np.mean(llh1) - np.mean(llh2, axis=0)
        # 判断对数似然值是否低于阈值
        if llh1_llh2 < self.threshold:
            return True,llh1_llh2
        else:
            return False,llh1_llh2


if __name__ == '__main__':
    # 示例用法
    file = r"C:\data\音频素材\异音检测\dev_data\spk\test\anomaly_id_01_00000000.wav"  # 替换为你的音频文件路径
    ty_pe = 'swp'  # 数据类型
    ID = 'id_01'  # 数据集ID
    threshold = -10  # 对数似然值的阈值
    v = VAEInterface()
    if v.judge_is_normal(file):
        print('The audio file is abnormal.')
    else:
        print('The audio file is normal.')
