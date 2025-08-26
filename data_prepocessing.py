# 将数据修改为normal_id_01_00000000.wav格式
import os
import shutil
import re
import argparse
import glob
import time

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm, trange

"""处理文件名"""
raw_normal_data_dir = r"E:\音频素材\异音检测\split\normal"
raw_anomaly_data_dir = r"E:\音频素材\异音检测\split\abnormal"
save_train_data_dir = r"E:\音频素材\异音检测\dev_data\spk\train"
save_test_data_dir = r"E:\音频素材\异音检测\dev_data\spk\test"


def generate_normal_data():
    NSPK320_TRAIN_NUM = 0
    NSPK320_TEST_NUM = 0
    SPK3200_TRAIN_NUM = 0
    SPK3200_TEST_NUM = 0
    SPK301_TRAIN_NUM = 0
    SPK301_TEST_NUM = 0
    SPK603_TRAIN_NUM = 0
    SPK603_TEST_NUM = 0
    for root, dirs, files in os.walk(raw_normal_data_dir):
        for i in trange(len(files)):
            file = files[i]
            if not file.endswith(".wav"):
                continue
            new_file_name = ""
            if 'NSPK320' in file:
                # 重命名  normal_id_01_00000000.wav
                if NSPK320_TRAIN_NUM < 100:
                    new_file_name = 'normal_id_01_{:08d}.wav'.format(NSPK320_TRAIN_NUM)
                    shutil.copy(os.path.join(root, file), os.path.join(save_train_data_dir, new_file_name))
                    NSPK320_TRAIN_NUM += 1
                else:
                    new_file_name = 'normal_id_01_{:08d}.wav'.format(NSPK320_TEST_NUM)
                    shutil.copy(os.path.join(root, file), os.path.join(save_test_data_dir, new_file_name))
                    NSPK320_TEST_NUM += 1

            elif 'SPK3200' in file:
                # 重命名  normal_id_02_00000000.wav
                if SPK3200_TRAIN_NUM < 170:
                    new_file_name = 'normal_id_02_{:08d}.wav'.format(SPK3200_TRAIN_NUM)
                    shutil.copy(os.path.join(root, file), os.path.join(save_train_data_dir, new_file_name))
                    SPK3200_TRAIN_NUM += 1
                else:
                    new_file_name = 'normal_id_02_{:08d}.wav'.format(SPK3200_TEST_NUM)
                    shutil.copy(os.path.join(root, file), os.path.join(save_test_data_dir, new_file_name))
                    SPK3200_TEST_NUM += 1

            elif 'SPK301' in file:
                # 重命名  normal_id_03_00000000.wav
                if SPK301_TRAIN_NUM < 150:
                    new_file_name = 'normal_id_03_{:08d}.wav'.format(SPK301_TRAIN_NUM)
                    shutil.copy(os.path.join(root, file), os.path.join(save_train_data_dir, new_file_name))
                    SPK301_TRAIN_NUM += 1
                else:
                    new_file_name = 'normal_id_03_{:08d}.wav'.format(SPK301_TEST_NUM)
                    shutil.copy(os.path.join(root, file), os.path.join(save_test_data_dir, new_file_name))
                    SPK301_TEST_NUM += 1

            elif 'SPK603' in file:
                # 重命名  normal_id_04_00000000.wav
                if SPK603_TRAIN_NUM < 50:
                    new_file_name = 'normal_id_04_{:08d}.wav'.format(SPK603_TRAIN_NUM)
                    shutil.copy(os.path.join(root, file), os.path.join(save_train_data_dir, new_file_name))
                    SPK603_TRAIN_NUM += 1
                else:
                    new_file_name = 'normal_id_04_{:08d}.wav'.format(SPK603_TEST_NUM)
                    shutil.copy(os.path.join(root, file), os.path.join(save_test_data_dir, new_file_name))
                    SPK603_TEST_NUM += 1


def generate_anomaly_data():
    NSPK320_TEST_NUM = 0
    SPK3200_TEST_NUM = 0
    SPK301_TEST_NUM = 0
    SPK603_TEST_NUM = 0
    for root, dirs, files in os.walk(raw_anomaly_data_dir):
        for i in trange(len(files)):
            file = files[i]
            if not file.endswith(".wav"):
                continue
            new_file_name = ""
            if 'NSPK320' in file:
                # 重命名  anomaly_id_01_00000000.wav
                new_file_name = 'anomaly_id_01_{:08d}.wav'.format(NSPK320_TEST_NUM)
                shutil.copy(os.path.join(root, file), os.path.join(save_test_data_dir, new_file_name))
                NSPK320_TEST_NUM += 1
            elif 'SPK3200' in file:
                # 重命名  anomaly_id_02_00000000.wav
                new_file_name = 'anomaly_id_02_{:08d}.wav'.format(SPK3200_TEST_NUM)
                shutil.copy(os.path.join(root, file), os.path.join(save_test_data_dir, new_file_name))
                SPK3200_TEST_NUM += 1
            elif 'SPK301' in file:
                # 重命名  anomaly_id_03_00000000.wav
                new_file_name = 'anomaly_id_03_{:08d}.wav'.format(SPK301_TEST_NUM)
                shutil.copy(os.path.join(root, file), os.path.join(save_test_data_dir, new_file_name))
                SPK301_TEST_NUM += 1
            elif 'SPK603' in file:
                # 重命名  anomaly_id_04_00000000.wav
                new_file_name = 'anomaly_id_04_{:08d}.wav'.format(SPK603_TEST_NUM)
                shutil.copy(os.path.join(root, file), os.path.join(save_test_data_dir, new_file_name))
                SPK603_TEST_NUM += 1


def pad_audio_to_10s():
    """将音频补齐到10s，16kHz wav"""

    def pad_audio(audio, sr, target_length):
        if len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            return audio[:target_length]

    for wav_dir in [save_train_data_dir, save_test_data_dir]:
        for root, dirs, files in os.walk(wav_dir):
            for file in tqdm(files):
                if not file.endswith(".wav"):
                    continue
                audio, sr = librosa.load(os.path.join(root, file), sr=16000)
                audio = pad_audio(audio, sr, 10 * sr)
                sf.write(os.path.join(root, file), audio, sr)


import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


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


def plot_ghost_ground_truth():
    """
    绘制幽灵数据
    Returns:
    """
    # 生成600 * 600 全黑的数据
    ghost_data = np.zeros((600, 600, 3), dtype=np.uint8)
    # 保存为png
    plt.imsave('ghost.png', ghost_data)


def generate_ghost_ground_truth():
    """
    生成幽灵 ground truth数据
    Returns:
    """
    test_dir_list = ["ADer/data/spk/N32", "ADer/data/spk/IP", "ADer/data/spk/TRY"]
    for test_dir in test_dir_list:
        for root, dirs, files in os.walk(os.path.join(test_dir, 'test')):
            # 只处理bad文件夹
            if 'bad' not in root:
                continue
            for file in tqdm(files):
                # 生成幽灵数据，复制'ghost.png' 到ground_truth文件夹
                shutil.copy('ghost.png',
                            os.path.join(test_dir, 'ground_truth', 'bad', file.split('.')[0] + '_mask.png'))


def generate_dataset():
    DATA_SRC_INFO = {"n32": (["原始数据/标记后/auto_test/48k",
                              "原始数据/标记后/manual_record/250626/201P/split/48k",
                              "原始数据/标记后/manual_record/250626/3200wg 1.0/48k",
                              "原始数据/标记后/manual_record/250721/NSPK320/N32/split",
                              "原始数据/标记后/manual_record/250721/SPK201/N32/split",
                              "原始数据/标记后/manual_record/250721/SPK3200/N32/split",
                              "原始数据/标记后/N32MPT/48k"],
                             "ADer/data/spk/N32/train",
                             "ADer/data/spk/N32/test"),
                     "ip": (["原始数据/标记后/manual_record/250721/NSPK320/IP/split",
                             "原始数据/标记后/manual_record/250721/SPK201/IP/split",
                             "原始数据/标记后/manual_record/250721/SPK3200/IP/split"],
                            "ADer/data/spk/IP/train",
                            "ADer/data/spk/IP/test"),
                     "try": (["原始数据/标记后/manual_record/250721/NSPK320/TRY/split",
                              "原始数据/标记后/manual_record/250721/SPK201/TRY/split",
                              "原始数据/标记后/manual_record/250721/SPK3200/TRY/split"],
                             "ADer/data/spk/TRY/train",
                             "ADer/data/spk/TRY/test")
                     }
    for key in DATA_SRC_INFO.keys():
        audio_good_num = 0
        audio_bad_num = 0
        IP_SRC_DIR_LIST = DATA_SRC_INFO[key][0]
        save_train_data_dir = DATA_SRC_INFO[key][1]
        save_test_data_dir = DATA_SRC_INFO[key][2]

        for ip_src_dir in IP_SRC_DIR_LIST:
            for root, dirs, files in os.walk(ip_src_dir):
                # 若目标文件夹不存在，则新建
                if not os.path.exists(os.path.join(save_train_data_dir, 'good')):
                    os.makedirs(os.path.join(save_train_data_dir, 'good'))
                # else:
                #     # 删除所有内容后新建
                #     # shutil.rmtree(os.path.join(save_train_data_dir, 'good'))
                #     os.makedirs(os.path.join(save_train_data_dir, 'good'))
                for _root, _dirs, _files in os.walk(os.path.join(root, 'good')):
                    for file in tqdm(_files):
                        if not file.endswith(".wav"):
                            continue
                        plot_spectrogram(os.path.join(_root, file),
                                         os.path.join(save_train_data_dir, 'good',
                                                      '{:03d}.png'.format(audio_good_num + 1)))
                        audio_good_num += 1
                # 若目标文件夹不存在，则新建
                if not os.path.exists(os.path.join(save_test_data_dir, 'bad')):
                    os.makedirs(os.path.join(save_test_data_dir, 'bad'))
                # else:
                #     # 删除所有内容后新建
                #     # shutil.rmtree(os.path.join(save_test_data_dir, 'bad'))
                #     os.makedirs(os.path.join(save_test_data_dir, 'bad'))
                for _root, _dirs, _files in os.walk(os.path.join(root, 'bad')):
                    for file in tqdm(_files):
                        if not file.endswith(".wav"):
                            continue
                        plot_spectrogram(os.path.join(_root, file),
                                         os.path.join(save_test_data_dir, 'bad',
                                                      '{:03d}.png'.format(audio_bad_num + 1)))
                        audio_bad_num += 1
        # 等待3s
        time.sleep(3)
        # 选择10%的good样本，移动到test中
        # 若目标文件夹不存在，则新建
        if not os.path.exists(os.path.join(save_test_data_dir, 'good')):
            os.makedirs(os.path.join(save_test_data_dir, 'good'))
        # else:
        #     # 删除所有内容后新建
        #     # shutil.rmtree(os.path.join(save_test_data_dir, 'good'))
        #     os.makedirs(os.path.join(save_test_data_dir, 'good'))
        print("Save to {} and {}".format(save_train_data_dir, save_test_data_dir))
        for file in tqdm(os.listdir(os.path.join(save_train_data_dir, 'good'))):
            if np.random.rand() < 0.1:
                shutil.move(os.path.join(save_train_data_dir, 'good', file),
                            os.path.join(save_test_data_dir, 'good', file))


if __name__ == '__main__':
    # generate_normal_data()
    # generate_anomaly_data()
    # pad_audio_to_10s()
    # generate_dataset()
    # plot_ghost_ground_truth()
    generate_ghost_ground_truth()
