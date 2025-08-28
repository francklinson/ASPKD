# 分配ADer任务
import argparse
import os
import shutil

import librosa
import numpy as np
from matplotlib import pyplot as plt

from ADer.configs import get_cfg
from ADer.util.net import init_training
from ADer.util.util import run_pre, init_checkpoint
from ADer.trainer import get_trainer


def plot_spectrogram(audio_path, output_path):
    """
    读取音频文件，绘制时频图并保存为.png

    参数:
        audio_path (str): 输入音频文件路径
        output_path (str): 输出图像保存路径
    """
    # 加载音频文件
    y, sr = librosa.load(audio_path)
    # 计算短时傅里叶变换(STFT)
    D = librosa.stft(y)
    # 转换为分贝标度
    DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # 创建图形
    # 图形大小
    plt.figure(figsize=(6.3, 6.3))
    # 绘制时频图
    # librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
    librosa.display.specshow(DB, sr=sr)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Log-frequency power spectrogram')
    # 不需要边缘
    plt.axis('off')
    plt.tight_layout()
    # 保存图像，没有白边
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


class ADerTaskAssigner:
    def __init__(self, method):
        assert method in ['MambaAD', 'InVad', 'ViTAD', 'DiAD', 'UniAD']
        self.method = method
        if self.method == 'MambaAD':
            self.cfg_path = 'ADer/configs/mambaad/mambaad_spk.py'
        elif self.method == 'InVad':
            self.cfg_path = 'ADer/configs/invad/invad_spk.py'
        elif self.method == 'ViTAD':
            self.cfg_path = 'ADer/configs/vitad/vitad_spk.py'
        elif self.method == 'DiAD':
            # 暂未实现
            self.cfg_path = 'ADer/configs/diad/diad_spk.py'
        elif self.method == 'UniAD':
            self.cfg_path = 'ADer/configs/uniad/uniad_spk.py'
        else:
            raise NotImplementedError

    def test(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--cfg_path', default=self.cfg_path)
        parser.add_argument('-m', '--mode', default='test', choices=['train', 'test'])
        parser.add_argument('--sleep', type=int, default=-1)
        parser.add_argument('--memory', type=int, default=-1)
        parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
        parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
        parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER, )
        cfg_terminal = parser.parse_args()
        cfg = get_cfg(cfg_terminal)

        # 添加可视化配置
        cfg.vis = True
        cfg.vis_dir = 'vis'

        run_pre(cfg)
        init_training(cfg)
        init_checkpoint(cfg)
        i_trainer = get_trainer(cfg)
        i_trainer.run()

    def train(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--cfg_path', default=self.cfg_path)
        parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
        parser.add_argument('--sleep', type=int, default=-1)
        parser.add_argument('--memory', type=int, default=-1)
        parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
        parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
        parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER, )
        cfg_terminal = parser.parse_args()
        cfg = get_cfg(cfg_terminal)
        run_pre(cfg)
        init_training(cfg)
        init_checkpoint(cfg)
        i_trainer = get_trainer(cfg)
        i_trainer.run()

    def inference(self, inference_dir='inference_dir'):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--cfg_path', default=self.cfg_path)
        parser.add_argument('-m', '--mode', default='inference', choices=['train', 'test', 'inference'])
        parser.add_argument('--sleep', type=int, default=-1)
        parser.add_argument('--memory', type=int, default=-1)
        parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
        parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
        parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER, )
        # 处理输入音频数据，默认存放在inference_dir目录下
        # 先清理一下原来的数据，删除'data/spk/INF/test/bad'目录下的wav文件
        print("Clear cache...")
        for root, dirs, files in os.walk('data/spk/INF/test/bad'):
            for file in files:
                if file.endswith('.png'):
                    os.remove(os.path.join(root, file))
        for root, dirs, files in os.walk('data/spk/INF/ground_truth/bad'):
            for file in files:
                if file.endswith('.png'):
                    os.remove(os.path.join(root, file))
        # wav转png
        for root, dirs, files in os.walk(inference_dir):
            for file in files:
                if file.endswith('.wav'):
                    # 处理音频文件
                    print(f"Processing: {file}")
                    plot_spectrogram(audio_path=os.path.join(root, file),
                                     output_path=os.path.join('data/spk/INF/test/bad', file.replace('wav', 'png')))
                    # 生成ghost数据
                    shutil.copy('ghost.png',
                                os.path.join('data/spk/INF/ground_truth/bad', file.split('.')[0] + '_mask.png'))
        # 更新meta.json文件
        # 调用 python data/gen_benchmark/spk.py
        print("Generating meta.json...")
        os.system('python data/gen_benchmark/spk.py')
        cfg_terminal = parser.parse_args()
        cfg = get_cfg(cfg_terminal)
        # 添加可视化配置
        cfg.vis = True
        cfg.vis_dir = 'vis'
        run_pre(cfg)
        init_training(cfg)
        init_checkpoint(cfg)
        i_trainer = get_trainer(cfg)
        i_trainer.run()



class MambaAD(ADerTaskAssigner):
    def __init__(self, method='MambaAD'):
        super().__init__(method)


class InVad(ADerTaskAssigner):
    def __init__(self, method='InVad'):
        super().__init__(method)


class ViTAD(ADerTaskAssigner):
    def __init__(self, method='ViTAD'):
        super().__init__(method)


class DiAD(ADerTaskAssigner):
    def __init__(self, method='DiAD'):
        super().__init__(method)


class UniAD(ADerTaskAssigner):
    def __init__(self, method='UniAD'):
        super().__init__(method)
