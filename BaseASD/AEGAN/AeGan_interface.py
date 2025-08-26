"""
外部调用接口类，基于 aegan_test.py 改写
"""
import os
import shutil

import torch
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import BaseASD.AEGAN.emb_distance as EDIS
from BaseASD.AEGAN.aegan_net import Discriminator, AEDC
from BaseASD.AEGAN.datasets import SegSet, ClipSet
from BaseASD.ASDBase import AnomalySoundDetectionBase


class AEGANInterface(AnomalySoundDetectionBase):
    def __init__(self):
        super(AEGANInterface, self).__init__()
        self.param = None
        with open('BaseASD/AEGAN/config.yaml', encoding='utf-8') as fp:
            self.param = yaml.safe_load(fp)
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        self.netD = None
        self.netG = None
        self.pth_file = None
        self.load_model()
        self.D_metric = ['D_maha', 'D_knn', 'D_lof', 'D_cos']
        self.G_metric = ['G_x_2_sum', 'G_x_2_min', 'G_x_2_max', 'G_x_1_sum', 'G_x_1_min', 'G_x_1_max',
                         'G_z_2_sum', 'G_z_2_min', 'G_z_2_max', 'G_z_1_sum', 'G_z_1_min', 'G_z_1_max',
                         'G_z_cos_sum', 'G_z_cos_min', 'G_z_cos_max']
        self.all_metric = self.D_metric + self.G_metric

    def load_model(self):
        """
        Load the pre-trained model
        """
        self.netD = Discriminator(self.param)
        self.netG = AEDC(self.param)
        self.pth_file = torch.load(r"BaseASD/AEGAN/model/spk.pth", map_location=torch.device('cpu'), weights_only=False)
        self.netD.load_state_dict(self.pth_file['netD'])
        self.netG.load_state_dict(self.pth_file['netG'])
        self.netD.to(self.device)
        self.netG.to(self.device)

    @staticmethod
    def check_file_path(input_file_path):
        """
        检查输入的音频文件格式是否正确
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError("file not found!!: {}".format(input_file_path))
        # 确定是文件
        if not os.path.isfile(input_file_path):
            raise FileNotFoundError("This is not a file!!: {}".format(input_file_path))
        # 检查文件格式，只支持wav
        if not input_file_path.lower().endswith(".wav"):
            raise FileNotFoundError("This is not a wav file!!: {}".format(input_file_path))

    def predict(self, file_path):
        """
        predict 方法用于预测给定音频文件是否为正常声音。
        首先加载训练数据，然后检查输入文件路径，将文件复制到测试目录，并重命名。
        使用 get_d_aver_emb 方法获取训练集的嵌入特征，然后使用 gan_test 方法进行测试。
        最后，删除临时文件并返回预测结果。
        """
        train_data = SegSet(self.param, self.param['train_set'], 'train')
        self.param['all_mid'] = train_data.get_mid()
        print(f"=> Recorded best performance: {self.pth_file['best_aver']:.4f}")
        mt = "spk"
        self.check_file_path(file_path)
        # 测试只使用一个文件，因此把测试文件挪到dataset/devdata/spk/test目录下，命名为anomaly_id_01_00000000.wav
        self.param["dataset_dir"] = 'BaseASD/AEGAN/dataset'
        target_dir = os.path.join(self.param["dataset_dir"], 'dev_data', mt, 'test')
        # 复制文件
        shutil.copy(file_path, target_dir)
        # 修改文件名
        old_file_name = os.path.split(file_path)[-1]
        new_file_name = 'anomaly_id_01_00000000.wav'
        old_file_path = os.path.join(target_dir, old_file_name)
        new_file_path = os.path.join(target_dir, new_file_name)
        if os.path.exists(new_file_path):
            os.remove(new_file_path)
        os.rename(old_file_path, new_file_path)

        mt_test_set = ClipSet(self.param, mt, 'dev', 'test')

        te_ld = DataLoader(mt_test_set,
                           batch_size=1,
                           shuffle=False,
                           num_workers=0)
        print("Test dataloader prepared!")
        train_embs = self.get_d_aver_emb(self.netD, train_data, self.device)
        y_true_all, y_score_all = self.gan_test(te_ld, train_embs)

        # 计算完成后删除文件
        os.remove(new_file_path)
        return y_true_all, y_score_all

    def get_d_aver_emb(self, netD, train_set, device):
        """
        获取训练集的嵌入特征
        遍历数据集，并将每个样本的特征存储在字典中。
        """
        netD.eval()
        train_embs = {mid: [] for mid in self.param['all_mid']}
        with torch.no_grad():
            for idx in range(train_set.get_clip_num()):
                mel, mid, _ = train_set.get_clip_data(idx)
                mel = torch.from_numpy(mel).to(device)
                _, feat_real = netD(mel)
                feat_real = feat_real.squeeze().mean(dim=0).cpu().numpy()
                train_embs[mid].append(feat_real)
        for mid in train_embs.keys():
            train_embs[mid] = np.array(train_embs[mid], dtype=np.float32)
        return train_embs

    def gan_test(self, te_ld, train_embs):
        """
        测试模型性能
        """
        print("============== MODEL TESTING ==============")
        # detect_location, score_type, score_comb= ('x', 'z'), ('2', '1'), ('sum', 'min', 'max')
        """
        D_metric：判别网络的评价指标，包括马氏距离（maha）、最近邻（knn）、局部离群因子（lof）和余弦相似度（cos）。
        G_metric：生成网络的评价指标，包括不同操作（如加、减、乘、除）和不同距离度量（如欧氏距离、余弦相似度）的结果。
        all_metric：所有评价指标的集合。
        """

        """
        初始化嵌入检测器
        edetect：嵌入检测器，用于计算嵌入表示的各种距离度量。
        edfunc：距离度量函数的字典。
        metric2id 和 id2metric：评价指标的映射关系。
        """
        edetect = EDIS.EmbeddingDetector(train_embs)
        edfunc = {'maha': edetect.maha_score, 'knn': edetect.knn_score,
                  'lof': edetect.lof_score, 'cos': edetect.cos_score}
        metric2id = {m: meid for m, meid in zip(self.all_metric, range(len(self.all_metric)))}
        id2metric = {v: k for k, v in metric2id.items()}

        """
        定义辅助函数
        specfunc：对张量进行求和操作。
        stfunc：定义不同操作（如平方、绝对值、余弦相似度）的函数。
        scfunc：定义不同统计操作（如求和、最小值、最大值）的函数。
        """

        def specfunc(x):
            return x.sum(axis=tuple(list(range(1, x.ndim))))

        stfunc = {'2': lambda x, y: (x - y).pow(2),
                  '1': lambda x, y: (x - y).abs(),
                  'cos': lambda x, y: 1 - F.cosine_similarity(x, y)}
        scfunc = {'sum': lambda x: x.sum().item(),
                  'min': lambda x: x.min().item(),
                  'max': lambda x: x.max().item()}
        """
        将判别网络和生成网络设置为评估模式。
        初始化真实标签和得分字典。
        使用 torch.no_grad() 禁用梯度计算，以节省内存。
        遍历测试数据集，计算每个样本的判别和生成得分。
        将得分和标签存储在字典中。
        """
        self.netD.eval()
        self.netG.eval()
        # {mid: []}
        y_true_all, y_score_all = [{} for _ in metric2id.keys()], [{} for _ in metric2id.keys()]
        with torch.no_grad():
            with tqdm(total=len(te_ld)) as pbar:
                for mel, mid, status in te_ld:  # mel: 1*186*1*128*128
                    mel = mel.squeeze(0).to(self.device)
                    _, feat_t = self.netD(mel)
                    recon = self.netG(mel)
                    melz = self.netG(mel, outz=True)
                    reconz = self.netG(recon, outz=True)
                    feat_t = feat_t.mean(axis=0, keepdim=True).cpu().numpy()
                    mid, status = mid.item(), status.item()
                    # 遍历评价指标
                    for idx, metric in id2metric.items():
                        wn = metric.split('_')[0]
                        if wn == 'D':
                            dname = metric.split('_')[1]
                            score = edfunc[dname](feat_t)

                        elif wn == 'G':
                            dd, st, sc = tuple(metric.split('_')[1:])
                            ori = mel if dd == 'x' else melz
                            hat = recon if dd == 'x' else reconz
                            score = scfunc[sc](specfunc(stfunc[st](hat, ori)))

                        if mid not in y_true_all[idx].keys():
                            y_true_all[idx][mid] = []
                            y_score_all[idx][mid] = []
                        y_true_all[idx][mid].append(status)
                        y_score_all[idx][mid].append(score)
                    pbar.update(1)
        return y_true_all, y_score_all

    def judge_is_normal(self, file_path):
        """
        判断音频文件是否正常
        """
        _, y_score_all = self.predict(file_path)
        score_list = []
        for y_score in y_score_all:
            for k, v in y_score.items():
                score = v[0]
                if isinstance(score, np.ndarray):
                    score = score[0]
                score_list.append(score)
        # 一共19个metric，每个metric都会对应一个阈值
        # 根据19个中的多数给决策
        anomaly_threshold = self.param["anomaly_threshold"]
        count = 0
        for i in range(19):
            if score_list[i] < anomaly_threshold[i]:
                # 比阈值小，认为是正常的
                count += 1
        # 有10个指标任务是正常的，返回Ture
        return count >= 10


if __name__ == '__main__':
    a = AEGANInterface()
    print(a.judge_is_normal(r"E:\音频素材\异音检测\dev_data\spk\test\anomaly_id_02_00000001.wav"))
