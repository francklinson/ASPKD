import heapq
import json
import math
import os
import secrets
import shutil
import time

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from dtw import dtw
from tqdm import tqdm, trange


def plot_spectrogram(audio_path, output_path, offset=0.0, duration=None):
    """
    读取音频文件，绘制时频图并保存为.png

    参数:
        audio_path (str): 输入音频文件路径
        output_path (str): 输出图像保存路径
        offset (float, optional): 开始绘制的时间偏移量，单位为秒。默认为None，表示从音频开始绘制
        duration (float, optional): 绘制音频的持续时间，单位为秒。默认为None，表示绘制整个音频
    """
    # 加载音频文件
    y, sr = librosa.load(audio_path, offset=offset, duration=duration, sr=22050)
    # 幅值归一化
    y = librosa.util.normalize(y)
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
    # 转成RGB格式
    # 保存图像，没有白边
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close()


def plot_ghost_ground_truth():
    """
    绘制幽灵数据
    Returns:
    """
    # 生成600 * 600 全黑的数据
    ghost_data = np.zeros((600, 600, 3), dtype=np.uint8)
    # 中间随机构造一些白色
    for i in range(600):
        for j in range(300):
            ghost_data[i, j, :] = 255

    # 保存为png
    plt.imsave('all_black.png', ghost_data)


def convert_to_gray():
    from PIL import Image

    # 读取彩色图像并转换为灰度
    img = Image.open('all_black.png').convert('L')  # 'L'模式表示8位灰度
    # 修改为600*600大小
    img = img.resize((600, 600))
    # 保存为灰度图像
    img.save('ghost.png')


def generate_ghost_ground_truth():
    """
    生成幽灵 ground truth数据
    Returns:
    """
    test_dir_list = ["data/spk_251210/dk_22050","data/spk_251210/qzgy_22050"]
    for test_dir in test_dir_list:
        for root, dirs, files in os.walk(os.path.join(test_dir, 'test')):
            # 只处理bad文件夹
            if 'bad' not in root:
                continue
            for file in tqdm(files):
                # 如果没有ground_truth文件夹，则创建
                if not os.path.exists(os.path.join(test_dir, 'ground_truth')):
                    os.makedirs(os.path.join(test_dir, 'ground_truth'))
                # 如果没有bad文件夹，则创建
                if not os.path.exists(os.path.join(test_dir, 'ground_truth', 'bad')):
                    os.makedirs(os.path.join(test_dir, 'ground_truth', 'bad'))
                # 生成幽灵数据，复制'ghost.png' 到ground_truth文件夹
                shutil.copy('ghost.png',
                            os.path.join(test_dir, 'ground_truth', 'bad', file.split('.')[0] + '_mask.png'))


def generate_dataset():
    DATA_SRC_INFO = {"n32": ([
                                 r"E:\异音检测\raw\产测N32汇总\48k",
                                 r"E:\异音检测\raw\自动化产出音频数据",
                                 # r"E:\异音检测\raw\手动录制\1\201P\split\8k",
                                 # r"E:\异音检测\raw\手动录制\1\201P\split\16k",
                                 # r"E:\异音检测\raw\手动录制\1\201P\split\22.05k",
                                 # r"E:\异音检测\raw\手动录制\1\201P\split\32k",
                                 # r"E:\异音检测\raw\手动录制\1\201P\split\44.1k",
                                 # r"E:\异音检测\raw\手动录制\1\201P\split\48k",
                                 r"E:\异音检测\raw\手动录制\1\3200wg 1.0\8k",
                                 r"E:\异音检测\raw\手动录制\1\3200wg 1.0\16k",
                                 # r"E:\异音检测\raw\手动录制\1\3200wg 1.0\22.05k",
                                 # r"E:\异音检测\raw\手动录制\1\3200wg 1.0\32k",
                                 # r"E:\异音检测\raw\手动录制\1\3200wg 1.0\44.1k",
                                 # r"E:\异音检测\raw\手动录制\1\3200wg 1.0\48k",
                                 # r"E:\异音检测\raw\手动录制\2\201P\N32\split",
                                 r"E:\异音检测\raw\手动录制\2\3200WG\N32\split",
                                 # r"E:\异音检测\raw\手动录制\2\NSPK320\N32\split",
                             ],
                             r"C:\Users\W0401544_ZCH\PycharmProjects\ASPKD\data\spk\N32\train",
                             r"C:\Users\W0401544_ZCH\PycharmProjects\ASPKD\data\spk\N32\test"),
        # "ip": ([r"E:\异音检测\raw\手动录制\2\201P\IP",
        #         r"E:\异音检测\raw\手动录制\2\3200WG\IP\split",
        #         r"E:\异音检测\raw\手动录制\2\NSPK320\IP\split"],
        #        r"C:\Users\W0401544_ZCH\PycharmProjects\ASPKD\data\spk\IP\train",
        #        r"C:\Users\W0401544_ZCH\PycharmProjects\ASPKD\data\spk\IP\test"),
        # "try": ([r"E:\异音检测\raw\手动录制\2\201P\TRY\split",
        #          r"E:\异音检测\raw\手动录制\2\3200WG\TRY\split",
        #          r"E:\异音检测\raw\手动录制\2\NSPK320\TRY\split"],
        #         r"C:\Users\W0401544_ZCH\PycharmProjects\ASPKD\data\spk\TRY\train",
        #         r"C:\Users\W0401544_ZCH\PycharmProjects\ASPKD\data\spk\TRY\test")
    }
    for key in DATA_SRC_INFO.keys():
        print(f"Processing class: {key}")
        SRC_DIR_LIST = DATA_SRC_INFO[key][0]
        save_train_data_dir = DATA_SRC_INFO[key][1]
        save_test_data_dir = DATA_SRC_INFO[key][2]
        # 保存原始数据和生成数据之间的对应关系
        src_audio_gen_pic_map = dict()
        for src_dir in SRC_DIR_LIST:
            for root, dirs, files in os.walk(src_dir):
                # 若目标文件夹不存在，则新建
                if not os.path.exists(os.path.join(save_train_data_dir, 'good')):
                    os.makedirs(os.path.join(save_train_data_dir, 'good'))
                for _root, _dirs, _files in os.walk(os.path.join(root, 'good')):
                    for file in tqdm(_files):
                        if not file.endswith(".wav"):
                            continue
                        print(f"Generating spectrogram for {file} in class {key} train dataset")
                        new_file_name = secrets.token_hex(16)
                        plot_spectrogram(os.path.join(_root, file),
                                         os.path.join(save_train_data_dir, 'good',
                                                      '{}.png'.format(new_file_name)))
                        src_audio_gen_pic_map[os.path.join(_root, file)] = '{}.png'.format(new_file_name)
                # 若目标文件夹不存在，则新建
                if not os.path.exists(os.path.join(save_test_data_dir, 'bad')):
                    os.makedirs(os.path.join(save_test_data_dir, 'bad'))
                for _root, _dirs, _files in os.walk(os.path.join(root, 'bad')):
                    for file in tqdm(_files):
                        if not file.endswith(".wav"):
                            continue
                        print(f"Generating spectrogram for {file} in class {key} test dataset")
                        new_file_name = secrets.token_hex(16)
                        plot_spectrogram(os.path.join(_root, file),
                                         os.path.join(save_test_data_dir, 'bad',
                                                      '{}.png'.format(new_file_name)))
                        src_audio_gen_pic_map[os.path.join(_root, file)] = '{}.png'.format(new_file_name)
        # 保存对应关系到文件中
        with open('src_audio_gen_pic_map.json', 'w', encoding="utf-8") as f:
            json.dump(src_audio_gen_pic_map, f, ensure_ascii=False, indent=4)

        # 等待3s
        time.sleep(3)
        # 选择20%的good样本，移动到test中
        # 若目标文件夹不存在，则新建
        if not os.path.exists(os.path.join(save_test_data_dir, 'good')):
            os.makedirs(os.path.join(save_test_data_dir, 'good'))
        print("Save to {} and {}".format(save_train_data_dir, save_test_data_dir))
        for file in tqdm(os.listdir(os.path.join(save_train_data_dir, 'good'))):
            if np.random.rand() < 0.2:
                shutil.move(os.path.join(save_train_data_dir, 'good', file),
                            os.path.join(save_test_data_dir, 'good', file))


class FixedPriorityQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = []  # 最小堆 (优先级, 数据)

    def push(self, priority, item):
        """添加元素，保持队列不超过最大长度"""
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (priority, item))
        else:
            # 只保留优先级更高的元素（数值更小）
            if priority < self.heap[0][0]:
                heapq.heapreplace(self.heap, (priority, item))

    def pop(self):
        """弹出优先级最高的元素"""
        return heapq.heappop(self.heap) if self.heap else None

    def get_all(self):
        """获取所有元素（按优先级升序排序）"""
        return sorted(self.heap, key=lambda x: x[0])

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return str([f"({p}, {i})" for p, i in self.get_all()])

    def clear(self):
        while not self.empty():
            self.heap.pop()

    def empty(self):
        return len(self.heap) == 0


class MFCCLocate:
    def __init__(self, ref_file):
        self.ref_file = ref_file
        self.sr = 12000  # 低一点采样率就可以了
        self.hop_size = 512
        self.nfft = 2048
        # 按照这个配置，每一帧是间隔：512/16000=32ms 帧长：2048/16000=128ms
        # 使用示例
        self.pq = FixedPriorityQueue(10)  # 最大长度

    def extract_mfcc(self, audio_path, sr=16000, n_mfcc=13):
        """
        提取mfcc特征，调用librosa的方法
        Args:
            audio_path:
            sr:
            n_mfcc:

        Returns:

        """
        y, sr = librosa.load(audio_path, sr=sr)
        # nfft = 2048, hop_length = 512
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.T  # 转置为(时间帧, 特征维度)
        # return mfcc.mean(axis=1).flatten()

    def index2time(self, index):
        """
        index ——> time stamp
        Args:
            index:

        Returns:

        """
        _time = index * (self.hop_size / self.sr)
        return _time

    def time2index(self, _time):
        """
        time stamp ——> index
        Args:
            _time:

        Returns:
        """
        _index = math.floor(_time * self.sr / self.hop_size)
        return _index

    def audio_locate(self, long_audio_path):
        riddle_start_time = self._locate_short_audio_with_dtw(long_audio_path,
                                                              search_start_time=0,
                                                              search_stop_time=1e4,
                                                              n_mfcc=40,
                                                              jump_step=15)
        # print(f"riddle_start_time:{riddle_start_time}")
        # 精筛，在粗筛结果的基础上，前后1s以内
        # 用更大的n_mfcc和更小的step
        fine_start_time = self._locate_short_audio_with_dtw(long_audio_path,
                                                            search_start_time=riddle_start_time - 1,
                                                            search_stop_time=riddle_start_time + 1,
                                                            n_mfcc=40,
                                                            jump_step=1)
        # print(f"fine_start_time:{fine_start_time}")
        return fine_start_time

    def _locate_short_audio_with_dtw(self, long_audio_path, search_start_time, search_stop_time, n_mfcc,
                                     jump_step):
        """
        筛选
        Args:
            long_audio_path:
            search_start_time:
            search_stop_time:
            n_mfcc:
            jump_step:

        Returns:
        """
        # 提取特征
        long_mfcc = self.extract_mfcc(long_audio_path, sr=self.sr, n_mfcc=n_mfcc)
        short_mfcc = self.extract_mfcc(self.ref_file, sr=self.sr, n_mfcc=n_mfcc)
        # 滑动窗口对比
        min_distance = float('inf')
        window_size = short_mfcc.shape[0]
        best_start_index = -1

        start_index = max(0, self.time2index(search_start_time))
        stop_index = min(len(long_mfcc) - window_size + 1, self.time2index(search_stop_time))

        for i in range(start_index, stop_index, jump_step):
            window = long_mfcc[i:i + window_size]

            # dtw法
            distance = dtw(window, short_mfcc, dist_method='euclidean')
            self.pq.push(priority=distance.distance, item=i)
            # if distance.distance < min_distance:
            #     min_distance = distance.distance
            #     best_start_index = i

            # 余弦距离
            # distance = cosine_distances(window, short_mfcc)
            # if distance[0][0] < min_distance:
            #     min_distance = distance[0][0]
            #     best_start = i
        if not self.pq.empty():
            _, best_start_index = self.pq.pop()
        else:
            best_start_index = 0
        start_time = self.index2time(best_start_index)
        return start_time


class Preprocessor:
    def __init__(self, ref_file, split_method='mfcc_dtw'):
        self.ref_file = ref_file

        # 获取目标音频时长
        self.target_segment_duration = librosa.get_duration(path=self.ref_file)
        self.y_target, self.sr_target = librosa.load(self.ref_file)

        self.src_audio_gen_pic_map_file = "src_audio_gen_pic_map.json"
        self.src_audio_gen_pic_map = dict()
        # 若存在则加载
        if os.path.exists(self.src_audio_gen_pic_map_file):
            with open(self.src_audio_gen_pic_map_file, 'r', encoding="utf-8") as f:
                self.src_audio_gen_pic_map = json.load(f)
        assert split_method in ['mfcc_dtw', 'corr']
        self.split_method = split_method
        self.mfcc_finder = MFCCLocate(ref_file=self.ref_file)

    def find_audio_segment(self, audio_path, threshold=0.7):
        """
        在音频文件中查找特定片段

        参数:
            audio_path: 完整音频文件路径
            target_segment_path: 要查找的片段音频路径
            threshold: 相似度阈值(0-1)，越高越严格

        返回:
            找到的片段位置列表(以秒为单位)
        """
        # 加载音频文件
        y_full, sr_full = librosa.load(audio_path)
        # y_target, sr_target = librosa.load(self.ref_file)  # 放到类的初始化中去读取

        # 确保采样率一致
        if sr_full != self.sr_target:
            print(f"Found different sample rate, resample to {self.sr_target}Hz")
            y_full = librosa.resample(y_full, orig_sr=sr_full, target_sr=self.sr_target)

        # 计算互相关
        correlation = np.correlate(y_full, self.y_target, mode='valid')
        # 归一化互相关
        correlation = correlation / (np.max(np.abs(correlation)) + 1e-10)
        # 找到超过阈值的位置
        matches = np.where(correlation > threshold)[0]
        # 转换为秒
        positions = matches / sr_full
        # 取平均值
        if len(positions) > 0:
            return positions.mean()
        else:
            return -1

    def process_audio(self, file_list, save_dir):
        """
        处理file_list中的所有音频文件，并保存到save_dir文件夹下
        Args:
            file_list:
            save_dir:
        Returns:
        """
        # 检查输入数据类型正确，分别为文件夹和音频文件
        if not isinstance(file_list, list) or len(file_list) == 0:
            raise ValueError("输入的音频列表不存在")

        if not os.path.isfile(self.ref_file):
            raise ValueError("目标片段音频文件不存在")

        # 遍历音频目录中的所有音频文件
        for i in trange(len(file_list)):
            _file = file_list[i]
            if _file.endswith(".wav"):
                print(f"正在处理文件: {_file}")
                # 查找目标片段在音频中的位置
                found_position = 0

                if self.split_method == 'corr':
                    found_position = self.find_audio_segment(_file)
                elif self.split_method == 'mfcc_dtw':
                    found_position = self.mfcc_finder.audio_locate(_file)

                if found_position > 0:
                    print(f"找到片段起始位置(秒): {found_position}")
                    # 提取目标片段
                    new_file_name = secrets.token_hex(16)
                    # 保存文件
                    output_path = os.path.join(save_dir, f'{new_file_name}.png')
                    try:
                        plot_spectrogram(_file, output_path, offset=found_position,
                                         duration=min(10.0, self.target_segment_duration))
                        print(f"保存图像到: {output_path}")
                        self.src_audio_gen_pic_map[_file] = '{}.png'.format(new_file_name)
                    except Exception as e:
                        print(f"Data transform failed! Error: {e} \nPlease check file:{_file}")
                else:
                    print("未找到指定片段")
            else:
                print("---跳过非音频文件: {}".format(_file))

        # 保存对应关系到文件中
        with open(self.src_audio_gen_pic_map_file, 'w', encoding="utf-8") as f:
            json.dump(self.src_audio_gen_pic_map, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # generate_dataset()
    # plot_ghost_ground_truth()
    # convert_to_gray()
    # generate_ghost_ground_truth()

    # p = Preprocessor(ref_file="ref/渡口片段10s.wav")
    p = Preprocessor(ref_file="ref/青藏高原片段_10s.wav")
    # single
    # predict_file_list = [r"E:\异音检测\raw\手动录制\2\201P\TRY\split\bad\1.wav"]

    # batch
    predict_file_list = list()
    predict_dir = r"E:\异音检测\raw\自动化产出音频数据\251210更新"
    for root, dirs, files in os.walk(predict_dir):
        for file in files:
            if file.endswith(".wav"):
                predict_file_list.append(os.path.join(root, file))
    p.process_audio(
        file_list=predict_file_list,
        save_dir="slice")
