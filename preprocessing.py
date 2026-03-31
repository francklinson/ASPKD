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
    from core.shazam import AudioFingerprinter

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

    注意: Shazam 使用自己的采样率配置（默认16kHz），与主项目（22.05kHz）不同，
    但时间计算会自动处理转换。

    支持两种模式:
    1. 指定参考音频: 在长音频中定位特定参考片段的位置
    2. 仅数据库匹配: 不指定参考音频，自动从数据库中找到最匹配的参考音频并定位
    """

    def __init__(self, ref_file=None, threshold=10, auto_add_ref=True, auto_match=False):
        """
        初始化 Shazam 定位器

        Args:
            ref_file: 参考音频文件路径（可选，若不提供则需设置 auto_match=True）
            threshold: Shazam匹配阈值，低于此值认为定位失败
            auto_add_ref: 是否自动将参考音频添加到指纹库
            auto_match: 是否自动从数据库匹配参考音频（无需提供 ref_file）
        """
        if not SHAZAM_AVAILABLE:
            raise ImportError("Shazam 模块未安装或初始化失败，无法使用 shazam 定位方法")

        self.ref_file = ref_file
        self.threshold = threshold
        self.auto_add_ref = auto_add_ref
        self.auto_match = auto_match

        # 初始化指纹识别器
        self._fingerprinter = None
        self._ref_added = False

        # 验证参考音频（如果提供了）
        if ref_file and not os.path.exists(ref_file):
            raise FileNotFoundError(f"参考音频不存在: {ref_file}")

        # 检查模式
        if not ref_file and not auto_match:
            raise ValueError("必须提供 ref_file 或设置 auto_match=True")

        # 保存最后一次匹配的参考音频名称（用于 auto_match 模式下的分类）
        self._last_matched_name = ""

    def _get_fingerprinter(self):
        """获取指纹识别器（懒加载）"""
        if self._fingerprinter is None:
            self._fingerprinter = AudioFingerprinter()
            # 确保数据库已初始化
            self._fingerprinter.init_database()
            if self.auto_add_ref and not self._ref_added and self.ref_file:
                self._fingerprinter.add_reference(self.ref_file, name="shazam_ref")
                self._ref_added = True
        return self._fingerprinter

    def audio_locate(self, long_audio_path, debug=False):
        """
        在长音频中定位参考片段的位置

        Args:
            long_audio_path: 长音频文件路径
            debug: 是否输出详细调试信息

        Returns:
            float: 片段起始时间（秒），定位失败返回 -1
        """
        if not os.path.exists(long_audio_path):
            print(f"[Shazam] 音频文件不存在: {long_audio_path}")
            return -1

        try:
            fp = self._get_fingerprinter()

            if debug:
                if self.auto_match:
                    print(f"[Shazam调试] 模式: 自动数据库匹配")
                else:
                    print(f"[Shazam调试] 参考音频: {self.ref_file}")
                print(f"[Shazam调试] 查询音频: {long_audio_path}")
                print(f"[Shazam调试] 匹配阈值: {self.threshold}")

            # 使用 Shazam 定位
            if self.auto_match:
                # 自动匹配模式：从数据库中找到最匹配的参考音频
                location = fp.locate(
                    long_audio_path=long_audio_path,
                    auto_match=True,
                    threshold=self.threshold
                )
            else:
                # 指定参考音频模式
                location = fp.locate(
                    long_audio_path=long_audio_path,
                    reference_path=self.ref_file,
                    threshold=self.threshold
                )

            if debug:
                print(
                    f"[Shazam调试] 匹配结果: found={location.found}, offset={location.start_time:.3f}s, confidence={location.confidence}")
                if location.music_name:
                    print(f"[Shazam调试] 匹配到的参考音频: {location.music_name}")

            if location.found and location.start_time != -1.0:  # found=True 且 start_time 有效
                # 保存匹配到的参考音频名称
                self._last_matched_name = location.music_name
                # offset 解释: offset = 参考帧 - 查询帧
                # 负值表示参考音频出现在查询音频的 |offset| 秒处
                if location.start_time < 0:
                    actual_pos = -location.start_time
                    if self.auto_match:
                        print(
                            f"[Shazam] 定位成功: 匹配到 '{location.music_name}'，在查询音频的 {actual_pos:.2f}s 处出现 (置信度: {location.confidence})")
                    else:
                        print(
                            f"[Shazam] 定位成功: 参考音频在查询音频的 {actual_pos:.2f}s 处出现 (offset={location.start_time:.2f}s), 置信度: {location.confidence}")
                    return actual_pos  # 返回实际位置（取绝对值）
                else:
                    print(f"[Shazam] 定位成功: {location.start_time:.2f}s, 置信度: {location.confidence}")
                    return location.start_time
            else:
                print(f"[Shazam] 定位失败: 未找到匹配片段（置信度 {location.confidence} < 阈值 {self.threshold}）")
                if not self.auto_match:
                    print(f"[Shazam建议] 1. 先用 quickstart.py add 添加参考音频到指纹库")
                    print(f"[Shazam建议] 2. 尝试降低 shazam_threshold（当前 {self.threshold}）")
                return -1

        except Exception as e:
            print(f"[Shazam] 定位出错: {e}")
            import traceback
            traceback.print_exc()
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
    def __init__(self, ref_file=None, split_method='shazam', shazam_threshold=10, shazam_auto_match=False,
                 max_workers=1):
        """
        预处理器

        Args:
            ref_file: 参考音频文件路径（可选，shazam_auto_match=True 时不需要）
            split_method: 分割方法 ('mfcc_dtw', 'corr', 'shazam')
            shazam_threshold: Shazam匹配阈值
            shazam_auto_match: 是否使用Shazam自动数据库匹配模式（无需ref_file）
            max_workers: 并行处理的线程数，默认为1（串行），>1时启用多线程
        """
        self.ref_file = ref_file
        self.shazam_auto_match = shazam_auto_match
        self.max_workers = max_workers

        # 获取目标音频时长
        if ref_file:
            self.target_segment_duration = librosa.get_duration(path=self.ref_file)
            self.y_target, self.sr_target = librosa.load(self.ref_file)
        else:
            # 无参考音频模式：使用默认时长
            self.target_segment_duration = 10.0  # 默认10秒
            self.y_target, self.sr_target = None, 22050

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

        # 检查参数
        if split_method in ['mfcc_dtw', 'corr'] and not ref_file:
            raise ValueError(f"方法 {split_method} 需要提供 ref_file")
        if split_method == 'shazam' and not ref_file and not shazam_auto_match:
            raise ValueError("Shazam 方法需要提供 ref_file 或设置 shazam_auto_match=True")

        self.split_method = split_method
        if ref_file:
            self.mfcc_finder = MFCCLocate(ref_file=self.ref_file)
        else:
            self.mfcc_finder = None

        # Shazam 定位器（懒加载）
        self._shazam_finder = None
        self._shazam_threshold = shazam_threshold

    def _get_shazam_finder(self):
        """获取 Shazam 定位器（懒加载）"""
        if self._shazam_finder is None:
            self._shazam_finder = ShazamLocate(
                ref_file=self.ref_file,
                threshold=self._shazam_threshold,
                auto_add_ref=True,
                auto_match=self.shazam_auto_match
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

    def process_audio(self, file_list, save_dir, original_names=None):
        """
        处理file_list中的所有音频文件，并保存到save_dir文件夹下
        图片保存到 save_dir/picture/ 目录
        音频保存到 save_dir/audio/ 目录
        Args:
            file_list: 音频文件路径列表
            save_dir: 保存目录
            original_names: 原始文件名映射 {临时路径: 原始文件名}，用于Gradio上传的文件
        Returns:
            dict: 处理结果字典，格式为 {原始音频路径: {"dk": 图片路径, "qzgy": 图片路径}}
        """
        # 检查输入数据类型正确，分别为文件夹和音频文件
        if not isinstance(file_list, list) or len(file_list) == 0:
            raise ValueError("输入的音频列表不存在")

        if not self.shazam_auto_match and not os.path.isfile(self.ref_file):
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
                    # Shazam offset 含义: offset = 参考帧 - 查询帧
                    # 负值表示参考音频出现在查询音频的 |offset| 秒处
                    # 例如 offset=-7.68s 表示参考音频在查询音频的 7.68s 处开始
                    # 负值是有效定位结果，不需要调整

                if found_position >= 0:
                    print(f"找到片段起始位置(秒): {found_position}")
                    # 提取目标片段

                    # 生成直观的文件名
                    # 格式: {父文件夹}_{原始文件名}_{参考音频名}_pos{定位位置}s
                    # 使用原始文件名（如果有映射），否则使用临时文件名
                    if hasattr(self, '_original_names') and self._original_names and _file in self._original_names:
                        original_name = os.path.splitext(self._original_names[_file])[0]
                        # 对于Gradio上传的文件，不使用额外的父目录前缀
                        parent_dir = ""
                    else:
                        original_name = os.path.splitext(os.path.basename(_file))[0]
                        parent_dir = os.path.basename(os.path.dirname(_file))

                    # 获取参考音频名称
                    if self.shazam_auto_match and self._shazam_finder:
                        ref_name = getattr(self._shazam_finder, '_last_matched_name', 'unknown')
                    elif self.ref_file:
                        ref_name = os.path.splitext(os.path.basename(self.ref_file))[0]
                    else:
                        ref_name = 'unknown'

                    # 清理名称中的非法字符
                    def sanitize_name(name):
                        import re
                        # 替换非法字符为下划线
                        name = re.sub(r'[\\/:*?"<>|\.\s]+', '_', name)
                        # 限制长度
                        return name[:30]  # 最多30字符

                    parent_dir_clean = sanitize_name(parent_dir)
                    original_name_clean = sanitize_name(original_name)
                    ref_name_clean = sanitize_name(ref_name)

                    # 组合文件名（去掉随机hash，使文件名更可读）
                    # 如果parent_dir为空，则不添加前缀
                    if parent_dir_clean:
                        new_file_name = f"{parent_dir_clean}_{original_name_clean}_{ref_name_clean}_pos{found_position:.2f}s"
                    else:
                        new_file_name = f"{original_name_clean}_{ref_name_clean}_pos{found_position:.2f}s"

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
                        # auto_match模式下根据匹配到的参考音频名称判断
                        if self.shazam_auto_match:
                            # 从Shazam定位器获取最后一次匹配的参考音频名称
                            matched_name = getattr(self._shazam_finder, '_last_matched_name', '')
                            if "qzgy" in matched_name.lower() or "青藏高原" in matched_name:
                                result_dict[_file]["qzgy"] = pic_output_path
                                print(f"[Shazam] 自动匹配到 '{matched_name}'，分类为 qzgy")
                            elif "dk" in matched_name.lower() or "渡口" in matched_name:
                                result_dict[_file]["dk"] = pic_output_path
                                print(f"[Shazam] 自动匹配到 '{matched_name}'，分类为 dk")
                            else:
                                # 无法确定分类，默认保存到dk
                                result_dict[_file]["dk"] = pic_output_path
                                print(f"[Shazam] 自动匹配到 '{matched_name}'，无法确定分类，默认保存到 dk")
                        elif "qzgy" in self.ref_file.lower() or "青藏高原" in self.ref_file:
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

    def process_audio_parallel(self, file_list, save_dir):
        """
        并行处理file_list中的所有音频文件（仅支持shazam方法）
        
        使用多线程并发加速Shazam音频指纹查询
        
        Args:
            file_list: 音频文件路径列表
            save_dir: 保存目录
        Returns:
            dict: 处理结果字典
        """
        if not SHAZAM_AVAILABLE:
            raise ImportError("Shazam 模块不可用，无法使用并行处理")

        if self.split_method != 'shazam':
            raise ValueError(f"并行处理仅支持 'shazam' 方法，当前方法为 '{self.split_method}'")

        if self.max_workers <= 1:
            print("[警告] max_workers <= 1，将使用串行处理")
            return self.process_audio(file_list, save_dir)

        # 检查输入
        if not isinstance(file_list, list) or len(file_list) == 0:
            raise ValueError("输入的音频列表不存在")

        # 创建输出目录
        pic_dir = os.path.join(save_dir, 'picture')
        audio_dir = os.path.join(save_dir, 'audio')
        os.makedirs(pic_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        # 过滤出wav文件
        wav_files = [f for f in file_list if f.endswith(".wav")]
        print(f"[并行处理] 总文件数: {len(file_list)}, WAV文件数: {len(wav_files)}, 线程数: {self.max_workers}")

        # 导入并行接口
        try:
            from core.shazam import ParallelAudioFingerprinter, ParallelResult
        except ImportError:
            print("[错误] 无法导入并行接口，回退到串行处理")
            return self.process_audio(file_list, save_dir)

        # 阶段1：并行定位所有音频
        print("[阶段1/2] 并行音频指纹查询...")
        parallel_fp = ParallelAudioFingerprinter(max_workers=self.max_workers)

        def progress_callback(completed, total):
            print(f"  进度: {completed}/{total} ({completed / total * 100:.1f}%)")

        locate_results = parallel_fp.batch_locate(
            long_audio_paths=wav_files,
            reference_path=self.ref_file if not self.shazam_auto_match else None,
            threshold=self._shazam_threshold,
            auto_match=self.shazam_auto_match,
            progress_callback=progress_callback,
            use_parallel=True
        )

        stats = parallel_fp.get_stats()
        print(f"[阶段1完成] 成功: {stats['success']}, 失败: {stats['failed']}, 总耗时: {stats['total_time']:.2f}s")

        # 阶段2：串行处理切片和保存（IO操作，不需要并行）
        print("[阶段2/2] 生成时频图和音频切片...")
        result_dict = {}

        for i, result in enumerate(locate_results):
            if not result.success or not result.result or not result.result.matched:
                print(f"[跳过] {result.file_path}: {result.error or '匹配失败'}")
                continue

            _file = result.file_path
            # 修复：offset 为负值时表示参考音频在查询音频的 |offset| 秒处出现
            # 需要转换为正数作为实际切片的起始位置
            raw_offset = result.result.offset
            found_position = -raw_offset if raw_offset < 0 else raw_offset
            matched_name = result.result.name

            print(f"[{i + 1}/{len(locate_results)}] 处理: {_file}")
            print(f"       位置: {found_position:.2f}s, 匹配: {matched_name}")

            # 生成文件名
            if hasattr(self, '_original_names') and self._original_names and _file in self._original_names:
                original_name = os.path.splitext(self._original_names[_file])[0]
                parent_dir = ""
            else:
                original_name = os.path.splitext(os.path.basename(_file))[0]
                parent_dir = os.path.basename(os.path.dirname(_file))

            if self.shazam_auto_match:
                ref_name = matched_name or 'unknown'
            elif self.ref_file:
                ref_name = os.path.splitext(os.path.basename(self.ref_file))[0]
            else:
                ref_name = 'unknown'

            def sanitize_name(name):
                import re
                name = re.sub(r'[\\/:*?"<>|\.\s]+', '_', name)
                return name[:30]

            parent_dir_clean = sanitize_name(parent_dir)
            original_name_clean = sanitize_name(original_name)
            ref_name_clean = sanitize_name(ref_name)

            # 如果parent_dir为空，则不添加前缀
            if parent_dir_clean:
                new_file_name = f"{parent_dir_clean}_{original_name_clean}_{ref_name_clean}_pos{found_position:.2f}s"
            else:
                new_file_name = f"{original_name_clean}_{ref_name_clean}_pos{found_position:.2f}s"
            slice_duration = min(10.0, self.target_segment_duration)

            # 保存图片
            pic_output_path = os.path.join(pic_dir, f'{new_file_name}.png')
            try:
                plot_spectrogram(_file, pic_output_path, offset=found_position, duration=slice_duration)
                self.src_audio_gen_pic_map[_file] = f'{new_file_name}.png'

                if _file not in result_dict:
                    result_dict[_file] = {"dk": None, "qzgy": None}

                # 分类
                if self.shazam_auto_match:
                    if "qzgy" in ref_name_clean.lower() or "青藏高原" in ref_name_clean:
                        result_dict[_file]["qzgy"] = pic_output_path
                    else:
                        result_dict[_file]["dk"] = pic_output_path
                elif "qzgy" in self.ref_file.lower() or "青藏高原" in self.ref_file:
                    result_dict[_file]["qzgy"] = pic_output_path
                else:
                    result_dict[_file]["dk"] = pic_output_path

            except Exception as e:
                print(f"[错误] 生成图片失败: {e}")
                continue

            # 保存音频
            audio_output_path = os.path.join(audio_dir, f'{new_file_name}.wav')
            try:
                y_slice, sr_slice = librosa.load(_file, offset=found_position, duration=slice_duration, sr=22050)
                sf.write(audio_output_path, y_slice, sr_slice)
            except Exception as e:
                print(f"[错误] 保存音频失败: {e}")

        parallel_fp.close()
        print(f"[并行处理完成] 成功处理 {len(result_dict)} 个文件")
        return result_dict

    def process_audio(self, file_list, save_dir, use_parallel=None, original_names=None):
        """
        处理file_list中的所有音频文件，并保存到save_dir文件夹下
        
        自动根据max_workers选择串行或并行处理
        
        Args:
            file_list: 音频文件路径列表
            save_dir: 保存目录
            use_parallel: 是否强制使用并行处理，None表示自动决定
            original_names: 原始文件名映射 {临时路径: 原始文件名}
        Returns:
            dict: 处理结果字典
        """
        # 保存original_names供内部方法使用
        self._original_names = original_names or {}
        
        # 决定是否使用并行处理
        should_use_parallel = (
                use_parallel is True or
                (use_parallel is None and self.max_workers > 1 and self.split_method == 'shazam')
        )

        if should_use_parallel and self.split_method == 'shazam' and SHAZAM_AVAILABLE:
            return self.process_audio_parallel(file_list, save_dir)
        else:
            return self._process_audio_serial(file_list, save_dir)

    def _process_audio_serial(self, file_list, save_dir):
        """串行处理（原process_audio逻辑）"""
        # 检查输入数据类型正确，分别为文件夹和音频文件
        if not isinstance(file_list, list) or len(file_list) == 0:
            raise ValueError("输入的音频列表不存在")

        if not self.shazam_auto_match and not os.path.isfile(self.ref_file):
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

                if found_position >= 0:
                    print(f"找到片段起始位置(秒): {found_position}")
                    # 提取目标片段

                    # 生成直观的文件名
                    if hasattr(self, '_original_names') and self._original_names and _file in self._original_names:
                        original_name = os.path.splitext(self._original_names[_file])[0]
                        parent_dir = ""
                    else:
                        original_name = os.path.splitext(os.path.basename(_file))[0]
                        parent_dir = os.path.basename(os.path.dirname(_file))

                    if self.shazam_auto_match and self._shazam_finder:
                        ref_name = getattr(self._shazam_finder, '_last_matched_name', 'unknown')
                    elif self.ref_file:
                        ref_name = os.path.splitext(os.path.basename(self.ref_file))[0]
                    else:
                        ref_name = 'unknown'

                    def sanitize_name(name):
                        import re
                        name = re.sub(r'[\\/:*?"<>|\.\s]+', '_', name)
                        return name[:30]

                    parent_dir_clean = sanitize_name(parent_dir)
                    original_name_clean = sanitize_name(original_name)
                    ref_name_clean = sanitize_name(ref_name)

                    # 如果parent_dir为空，则不添加前缀
                    if parent_dir_clean:
                        new_file_name = f"{parent_dir_clean}_{original_name_clean}_{ref_name_clean}_pos{found_position:.2f}s"
                    else:
                        new_file_name = f"{original_name_clean}_{ref_name_clean}_pos{found_position:.2f}s"

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

                        if self.shazam_auto_match:
                            matched_name = getattr(self._shazam_finder, '_last_matched_name', '')
                            if "qzgy" in matched_name.lower() or "青藏高原" in matched_name:
                                result_dict[_file]["qzgy"] = pic_output_path
                                print(f"[Shazam] 自动匹配到 '{matched_name}'，分类为 qzgy")
                            elif "dk" in matched_name.lower() or "渡口" in matched_name:
                                result_dict[_file]["dk"] = pic_output_path
                                print(f"[Shazam] 自动匹配到 '{matched_name}'，分类为 dk")
                            else:
                                result_dict[_file]["dk"] = pic_output_path
                                print(f"[Shazam] 自动匹配到 '{matched_name}'，无法确定分类，默认保存到 dk")
                        elif "qzgy" in self.ref_file.lower() or "青藏高原" in self.ref_file:
                            result_dict[_file]["qzgy"] = pic_output_path
                        else:
                            result_dict[_file]["dk"] = pic_output_path

                    except Exception as e:
                        print(f"Data transform failed! Error: {e} \nPlease check file:{_file}")
                        continue

                    # 保存音频切片到audio目录
                    audio_output_path = os.path.join(audio_dir, f'{new_file_name}.wav')
                    try:
                        y_slice, sr_slice = librosa.load(_file, offset=found_position,
                                                         duration=slice_duration, sr=22050)
                        sf.write(audio_output_path, y_slice, sr_slice)
                        print(f"保存音频到: {audio_output_path}")
                    except Exception as e:
                        print(f"Audio save failed! Error: {e} \nPlease check file:{_file}")
                else:
                    print("未找到指定片段")
            else:
                print("---跳过非音频文件: {}".format(_file))

        # 清理 Shazam 资源
        if self._shazam_finder is not None:
            self._shazam_finder.close()

        return result_dict

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
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

    # 示例4: Shazam 并行处理（4线程，提速约2-3倍）
    p = Preprocessor(ref_file="ref/渡口片段10s.wav", split_method='shazam',
                     shazam_threshold=10, max_workers=4)

    # 示例5: Shazam 自动匹配模式 + 并行处理
    # p = Preprocessor(split_method='shazam', shazam_threshold=5,
    #                  shazam_auto_match=True, max_workers=4)

    # p = Preprocessor(shazam_auto_match=True, max_workers=4)  # max_workers=1为串行，>1启用并行
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
