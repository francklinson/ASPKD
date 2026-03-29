import heapq
import json
import math
import os
import secrets
import shutil
import sys
import time

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from dtw import dtw
from tqdm import tqdm, trange

# 添加项目根目录到路径，确保可以导入 Shazam 模块
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 尝试导入 Shazam 模块
try:
    from Shazam.api import AudioFingerprinter
    SHAZAM_AVAILABLE = True
except ImportError as e:
    SHAZAM_AVAILABLE = False
    # 只在详细模式下打印错误
    import importlib
    try:
        importlib.import_module('Shazam')
    except ImportError as inner_e:
        print(f"[警告] Shazam 模块未找到，shazam 定位方法不可用")
        # 调试信息：可以取消注释以下行查看详细错误
        # print(f"[调试] 导入错误详情: {inner_e}")


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
    # 等待1s
    time.sleep(1)
    # 转换为灰度图
    convert_to_gray()
    # 检查是否生成幽灵数据,有没有ghost.png
    if not os.path.exists('ghost.png'):
        raise FileNotFoundError("ghost.png not generated successfully!!! check the code!!!")


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
    test_dir_list = ["data/spk/qzgy", "data/spk/dk"]
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


class ShazamLocate:
    """
    使用 Shazam 音频指纹算法进行音频定位

    基于音频指纹技术，通过匹配频谱峰值特征来定位参考片段在长音频中的位置。
    相比 MFCC+DTW 具有更好的抗噪能力和更快的匹配速度。
    """

    def __init__(self, ref_file, sr=22050, threshold=10, auto_add_ref=True):
        """
        初始化 Shazam 定位器

        Args:
            ref_file: 参考音频文件路径
            sr: 采样率（默认22050，与主项目一致）
            threshold: Shazam匹配阈值，低于此值认为定位失败
            auto_add_ref: 是否自动将参考音频添加到指纹库
        """
        if not SHAZAM_AVAILABLE:
            raise ImportError("Shazam 模块未安装或初始化失败，无法使用 shazam 定位方法")

        self.ref_file = ref_file
        self.sr = sr
        self.threshold = threshold
        self.auto_add_ref = auto_add_ref

        # 初始化指纹识别器
        self._fingerprinter = None
        self._ref_added = False

        # 验证参考音频
        if not os.path.exists(ref_file):
            raise FileNotFoundError(f"参考音频不存在: {ref_file}")

    def _get_fingerprinter(self):
        """获取指纹识别器（懒加载）"""
        if self._fingerprinter is None:
            self._fingerprinter = AudioFingerprinter()
            # 确保数据库已初始化
            self._fingerprinter.init_database()
            if self.auto_add_ref and not self._ref_added:
                self._fingerprinter.add_reference(self.ref_file, name="shazam_ref")
                self._ref_added = True
        return self._fingerprinter

    def audio_locate(self, long_audio_path):
        """
        在长音频中定位参考片段的位置

        Args:
            long_audio_path: 长音频文件路径

        Returns:
            float: 片段起始时间（秒），定位失败返回 -1
        """
        if not os.path.exists(long_audio_path):
            print(f"[Shazam] 音频文件不存在: {long_audio_path}")
            return -1

        try:
            fp = self._get_fingerprinter()

            # 获取参考音频时长
            ref_duration = librosa.get_duration(path=self.ref_file, sr=self.sr)

            # 使用 Shazam 定位
            location = fp.locate(
                long_audio_path=long_audio_path,
                reference_path=self.ref_file,
                threshold=self.threshold
            )

            if location.found:
                print(f"[Shazam] 定位成功: {location.start_time:.2f}s, 置信度: {location.confidence}")
                return location.start_time
            else:
                print(f"[Shazam] 定位失败: 未找到匹配片段")
                return -1

        except Exception as e:
            print(f"[Shazam] 定位出错: {e}")
            return -1

    def close(self):
        """释放资源"""
        if self._fingerprinter:
            self._fingerprinter.close()
            self._fingerprinter = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


class Preprocessor:
    def __init__(self, ref_file, split_method='shazam', shazam_threshold=10):
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

        # 支持的方法: mfcc_dtw, corr, shazam
        valid_methods = ['mfcc_dtw', 'corr']
        if SHAZAM_AVAILABLE:
            valid_methods.append('shazam')

        if split_method not in valid_methods:
            raise ValueError(f"不支持的 split_method: {split_method}，可用选项: {valid_methods}")

        self.split_method = split_method
        self.mfcc_finder = MFCCLocate(ref_file=self.ref_file)

        # Shazam 定位器（懒加载）
        self._shazam_finder = None
        self._shazam_threshold = shazam_threshold

    def _get_shazam_finder(self):
        """获取 Shazam 定位器（懒加载）"""
        if self._shazam_finder is None:
            self._shazam_finder = ShazamLocate(
                ref_file=self.ref_file,
                sr=self.sr_target,
                threshold=self._shazam_threshold,
                auto_add_ref=True
            )
        return self._shazam_finder

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
        图片保存到 save_dir/picture/ 目录
        音频保存到 save_dir/audio/ 目录
        Args:
            file_list: 音频文件路径列表
            save_dir: 保存目录
        Returns:
            dict: 处理结果字典，格式为 {原始音频路径: {"dk": 图片路径, "qzgy": 图片路径}}
        """
        # 检查输入数据类型正确，分别为文件夹和音频文件
        if not isinstance(file_list, list) or len(file_list) == 0:
            raise ValueError("输入的音频列表不存在")

        if not os.path.isfile(self.ref_file):
            raise ValueError("目标片段音频文件不存在")

        # 创建picture和audio子目录
        pic_dir = os.path.join(save_dir, 'picture')
        audio_dir = os.path.join(save_dir, 'audio')
        os.makedirs(pic_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        # 初始化结果字典
        result_dict = {}

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
                elif self.split_method == 'shazam':
                    found_position = self._get_shazam_finder().audio_locate(_file)

                if found_position > 0:
                    print(f"找到片段起始位置(秒): {found_position}")
                    # 提取目标片段
                    new_file_name = secrets.token_hex(16)

                    # 计算切片时长
                    slice_duration = min(10.0, self.target_segment_duration)

                    # 保存图片到picture目录
                    pic_output_path = os.path.join(pic_dir, f'{new_file_name}.png')
                    try:
                        plot_spectrogram(_file, pic_output_path, offset=found_position,
                                         duration=slice_duration)
                        print(f"保存图像到: {pic_output_path}")
                        self.src_audio_gen_pic_map[_file] = '{}.png'.format(new_file_name)

                        # 记录到结果字典
                        if _file not in result_dict:
                            result_dict[_file] = {"dk": None, "qzgy": None}
                        # 根据参考音频类型判断是dk还是qzgy
                        if "qzgy" in self.ref_file.lower() or "青藏高原" in self.ref_file:
                            result_dict[_file]["qzgy"] = pic_output_path
                        else:
                            result_dict[_file]["dk"] = pic_output_path

                    except Exception as e:
                        print(f"Data transform failed! Error: {e} \nPlease check file:{_file}")
                        continue

                    # 保存音频切片到audio目录
                    audio_output_path = os.path.join(audio_dir, f'{new_file_name}.wav')
                    try:
                        # 加载音频切片
                        y_slice, sr_slice = librosa.load(_file, offset=found_position,
                                                         duration=slice_duration, sr=22050)
                        # 保存音频文件
                        sf.write(audio_output_path, y_slice, sr_slice)
                        print(f"保存音频到: {audio_output_path}")
                    except Exception as e:
                        print(f"Audio save failed! Error: {e} \nPlease check file:{_file}")
                else:
                    print("未找到指定片段")
            else:
                print("---跳过非音频文件: {}".format(_file))

        # 保存对应关系到文件中
        # with open(self.src_audio_gen_pic_map_file, 'w', encoding="utf-8") as f:
        #     json.dump(self.src_audio_gen_pic_map, f, ensure_ascii=False, indent=4)

        # 清理 Shazam 资源
        if self._shazam_finder is not None:
            self._shazam_finder.close()

        return result_dict

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        # 清理 Shazam 资源
        if self._shazam_finder is not None:
            self._shazam_finder.close()


if __name__ == '__main__':
    # plot_ghost_ground_truth()
    # convert_to_gray()

    # ==================== 使用示例 ====================

    # 示例1: 使用 MFCC+DTW 方法（默认）
    # p = Preprocessor(ref_file="ref/渡口片段10s.wav", split_method='mfcc_dtw')

    # 示例2: 使用互相关方法
    # p = Preprocessor(ref_file="ref/渡口片段10s.wav", split_method='corr')

    # 示例3: 使用 Shazam 音频指纹方法（推荐，需要 MySQL 数据库）
    # p = Preprocessor(ref_file="ref/渡口片段10s.wav", split_method='shazam', shazam_threshold=10)

    p = Preprocessor(ref_file="ref/渡口片段10s.wav")
    # p = Preprocessor(ref_file="ref/青藏高原片段_10s.wav")
    # single
    # predict_file_list = [r"E:\异音检测\raw\手动录制\2\201P\TRY\split\bad\1.wav"]

    # batch
    predict_file_list = list()
    predict_dir = "/home/zhouchenghao/PycharmProjects/ASD_for_SPK/原始数据/歌曲/TL-SPK3A20PG 1.0"
    for root, dirs, files in os.walk(predict_dir):
        for file in files:
            if file.endswith(".wav"):
                predict_file_list.append(os.path.join(root, file))

    # 使用上下文管理器确保资源释放
    with p:
        p.process_audio(
            file_list=predict_file_list,
            save_dir="slice")

    # generate_ghost_ground_truth()
