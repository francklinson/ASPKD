"""
外部调用接口，基于evaluate.py改写
"""
import librosa
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from BaseASD.ASDBase import AnomalySoundDetectionBase
from BaseASD.DifferNet.utils import *
from BaseASD.DifferNet.model import load_weights
from BaseASD.DifferNet.model import DifferNet
from PIL import Image


class DifferNetInterface(AnomalySoundDetectionBase):
    def __init__(self, model_name="32k"):
        super(DifferNetInterface, self).init()
        self.model = None
        self.model_name = model_name
        self.load_model()
        self.anomaly_threshold = 1
        self.eval_dir = os.path.join("BaseASD", "DifferNet", "eval", self.model_name)
        # 清空eval文件夹，删除已有文件
        self.clear_eval_dir()

    def load_model(self):
        """
        加载模型
        """
        self.model = DifferNet()
        load_weights(self.model, self.model_name)

    def predict(self, image_path, fixed_transforms=True):
        """
        计算异常得分
        """
        if fixed_transforms:
            fixed_degrees = [i * 360.0 / c.n_transforms_test for i in range(c.n_transforms_test)]
            transforms = [get_fixed_transforms(fd) for fd in fixed_degrees]
        else:
            transforms = [get_random_transforms()] * c.n_transforms_test
        img = Image.open(image_path).convert('RGB')
        transformed_imgs = torch.stack([tf(img) for tf in transforms])
        z = self.model(transformed_imgs)
        anomaly_score = torch.mean(z ** 2)
        print("image: %s, score: %.2f" % (image_path, anomaly_score))
        return anomaly_score

    def predict_1(self, eval_loader):
        """
        用dataloader方式写的预测函数，但是用来测试的时候没有label信息，看起来不能用
        """
        test_loss = list()
        test_z = list()
        test_labels = list()
        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_loader, disable=c.hide_tqdm_bar)):
                inputs, _ = preprocess_batch(data)
                z = self.model(inputs)
                loss = get_loss(z, self.model.nf.jacobian(run_forward=False))
                test_z.append(z)
                test_loss.append(t2np(loss))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
        anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
        print("is_anomaly: ", is_anomaly)
        print("anomaly_score: ", anomaly_score)
        return anomaly_score

    @staticmethod
    def convert_wav_to_spec(file_path, output_path):
        """
        将音频文件转换为时频谱图
        """
        # 读取音频文件
        print("Processing file: ", file_path)
        # 判断是不是正常的音频文件
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        # 判断是不是wav文件
        if not file_path.endswith(".wav"):
            raise ValueError(f"File {file_path} is not a wav file.")
        y, sr = librosa.load(file_path)
        # 计算时频谱图
        D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
        # 绘制时频谱图
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('时频谱图')
        plt.axis('off')
        plt.tight_layout()
        # 保存图像文件
        plt.savefig(output_path)
        # plt.show()

    def clear_eval_dir(self):
        """
        清空eval文件夹
        """
        # 若不存在，新建
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        else:
            # 清空eval文件夹，删除已有文件
            for file_name in os.listdir(self.eval_dir):
                file_path = os.path.join(self.eval_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def judge_is_normal(self, file_path):
        """
        判断是否为正常样本
        输入是wav文件，要转成图片
        """
        # 转换后的文件放在eval目录下
        # 获取文件名，不含后缀
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        # 转换后的文件名
        output_path = os.path.join(self.eval_dir, file_name + ".png")
        # 转换音频文件为时频谱图
        self.convert_wav_to_spec(file_path, output_path)
        # # 构建dataloader
        # eval_set = ImageFolderMultiTransform(self.eval_dir, transform=get_random_transforms(),
        #                                      n_transforms=c.n_transforms_test)
        # eval_loader = torch.utils.data.DataLoader(eval_set, pin_memory=True, batch_size=c.batch_size_test, shuffle=True,
        #                                           drop_last=False)
        # 预测异常得分
        anomaly_score = self.predict(output_path)
        # 判断是否为正常样本
        return anomaly_score < self.anomaly_threshold, anomaly_score
