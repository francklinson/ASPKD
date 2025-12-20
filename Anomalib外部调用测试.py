import os
import shutil
import time
from errno import ELOOP
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import librosa
import numpy as np
from sklearn import metrics
from Anomalib.data import MVTecAD
from Anomalib.models import Dinomaly
from Anomalib.engine import Engine
from Anomalib.data import PredictDataset
from Anomalib.visualization import ImageVisualizer
from data_prepocessing import Preprocessor

"""
Image Models:
    - CFA (:class:`Anomalib.models.image.Cfa`)
    - Cflow (:class:`Anomalib.models.image.Cflow`)
    - CSFlow (:class:`Anomalib.models.image.Csflow`)
    - DFKDE (:class:`Anomalib.models.image.Dfkde`)
    - DFM (:class:`Anomalib.models.image.Dfm`)
    - DRAEM (:class:`Anomalib.models.image.Draem`)
    - DSR (:class:`Anomalib.models.image.Dsr`)
    - EfficientAd (:class:`Anomalib.models.image.EfficientAd`)
    - FastFlow (:class:`Anomalib.models.image.Fastflow`)
    - FRE (:class:`Anomalib.models.image.Fre`)
    - GANomaly (:class:`Anomalib.models.image.Ganomaly`)
    - PaDiM (:class:`Anomalib.models.image.Padim`)
    - PatchCore (:class:`Anomalib.models.image.Patchcore`)
    - Reverse Distillation (:class:`Anomalib.models.image.ReverseDistillation`)
    - STFPM (:class:`Anomalib.models.image.Stfpm`)
    - SuperSimpleNet (:class:`Anomalib.models.image.Supersimplenet`)
    - UFlow (:class:`Anomalib.models.image.Uflow`)
    - VLM-AD (:class:`Anomalib.models.image.VlmAd`)
    - WinCLIP (:class:`Anomalib.models.image.WinClip`)

"""

# 定义数据集模块
datamodule = MVTecAD(root="data/spk_251031", category="qzgy_22050", train_batch_size=16, eval_batch_size=16)

# 定义检测模型
model = Dinomaly()

# 定义训练引擎
engine = Engine(max_epochs=1000)


class ModelTest:
    def __init__(self, ckpt_path):
        self.ckpt = ckpt_path

    def model_batch_test(self):
        """
        对模型在测试数据集上进行测试
        测试数据集已经全都是处理后的图像了，就不需要另外处理
        """
        # Load model and make predictions
        predictions = engine.predict(
            datamodule=datamodule,
            model=model,
            ckpt_path=self.ckpt,
        )
        if predictions is not None:
            total_pred_label_list = []
            total_pred_score_list = []
            total_gt_label_list = []

            # 根据预测结果，收集数据
            for prediction in predictions:
                image_path = prediction.image_path
                anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
                pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
                pred_score = prediction.pred_score  # Image-level anomaly score
                gt_label = prediction.gt_label
                total_pred_label_list.extend(pred_label.tolist())
                total_pred_score_list.extend(pred_score.tolist())
                total_gt_label_list.extend(gt_label.tolist())

            # 处理结果
            print("total_pred_label_list: ", total_pred_label_list)
            print("total_pred_score_list: ", total_pred_score_list)
            print("total_gt_label_list: ", total_gt_label_list)
            cnt = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            for i in range(len(total_pred_label_list)):
                # TPR = TP / (TP + FN)
                # FPR = FP / (FP + TN)
                # 判断为真
                if total_pred_label_list[i] is True:
                    # 实际为真
                    if total_gt_label_list[i] is True:
                        TP += 1
                    # 实际为假
                    else:
                        FP += 1
                # 判断为假
                else:
                    # 实际为真
                    if total_gt_label_list[i] is True:
                        FN += 1
                    # 实际为假
                    else:
                        TN += 1
            print("测试集总样本数：", len(total_pred_label_list))
            print("TP：", TP)
            print("TN：", TN)
            print("FP：", FP)
            print("FN：", FN)
            print("TPR：", TP / (TP + FN))
            print("FPR：", FP / (FP + TN))
            print("ACC：", (TP + TN) / (TP + TN + FP + FN))
            print("Precision：", TP / (TP + FP))
            print("Recall：", TP / (TP + FN))

            # 计算AUROC
            total_gt_label_list_to_num = [1 if i is True else 0 for i in total_gt_label_list]
            fpr, tpr, thresholds = metrics.roc_curve(total_gt_label_list_to_num, total_pred_score_list)
            print("fpr: ", fpr)
            print("tpr: ", tpr)
            print("thresholds: ", thresholds)
            print("AUROC: ", metrics.auc(fpr, tpr))
            # # 绘制ROC曲线
            # plt.plot(fpr, tpr)
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('ROC Curve')
            # # 保存
            # plt.savefig('roc_curve.png')

        return predictions


class ModelPredict:
    """
    执行模型预测功能
    包含：
        - 对单个音频文件执行预测；
        - 对批量音频文件执行预测；
    需要执行音频——>图像的预处理过程
    """

    def __init__(self, ckpt_path, ref_file):
        """
        初始化
        Args:
            ckpt_path:
            ref_file:
        """
        self.ckpt_path = ckpt_path
        self.preprocessor = Preprocessor(ref_file=ref_file)
        self.predict_dir = "predict"

    def clean_predict_dir(self):
        """
        清空predict文件夹
        Returns:

        """
        print("---Clean predict dir---")
        if os.path.exists(self.predict_dir):
            shutil.rmtree(self.predict_dir)
            os.mkdir(self.predict_dir)
        else:
            os.mkdir(self.predict_dir)
        print("---Clean done---")
        return

    def check_predict_dir(self):
        # 检查一下处理的结果，在self.predict_dir目录下
        # 获取文件列表
        return len(os.listdir(self.predict_dir)) != 0

    def __model_dir_predict(self, dir_path):
        """
        对文件夹内的批量音频文件进行预测
        首先提取出目标音频段
        Args:
            dir_path:
        Returns:
        """
        # 整理出文件夹下所有的音频文件列表
        predict_audio_file_list = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith(".wav")]
        if len(predict_audio_file_list) == 0:
            print("No wav file in folder!")
            return
        # 清空predict文件夹
        self.clean_predict_dir()

        # 执行音频预处理过程，提取目标音频段，并保存到 predict文件夹下
        print("Process audio, save to predict folder")
        self.preprocessor.process_audio(file_list=predict_audio_file_list, save_dir=self.predict_dir)

        if not self.check_predict_dir():
            assert RuntimeError("Predict folder is empty!")

        # 执行模型预测
        prediction = engine.predict(
            model=model,
            ckpt_path=self.ckpt_path,
            data_path=self.predict_dir,
        )
        res = list()
        if prediction is None :
            return res
        for _p in prediction:
            res.append([_p.image_path[0], _p.pred_label, _p.pred_score])
        return res

    def __model_file_predict(self, file_path):
        """
        对单个音频文件进行预测
        首先提取出目标音频段
        Args:
            file_path:
        Returns:
        """
        if not os.path.exists(file_path):
            print("Pic to infer does not exist!")
            return
        # 确认是wav音频文件
        if not file_path.endswith(".wav"):
            print("Pic to infer is not wav file!")
            return
        # 清空predict文件夹
        self.clean_predict_dir()

        # 执行音频预处理过程，提取目标音频段，并保存到 predict文件夹下
        print("Process audio, save to predict folder")
        self.preprocessor.process_audio(file_list=[file_path], save_dir=self.predict_dir)
        if not self.check_predict_dir():
            assert RuntimeError("Predict folder is empty!")

        # 执行模型预测
        prediction = engine.predict(
            model=model,
            ckpt_path=self.ckpt_path,
            data_path=self.predict_dir,
        )
        res = list()
        for _p in prediction:
            res.append([_p.image_path[0], _p.pred_label, _p.pred_score])
        return res

    def __model_list_predict(self, predict_file_list):
        """
        对文件列表内的音频文件进行预测
        Args:
            predict_file_list:
        Returns:
        """
        # 整理出文件夹下所有的音频文件列表
        if len(predict_file_list) == 0:
            print("No wav file in list!")
            return
        # 清空predict文件夹
        self.clean_predict_dir()

        # 执行音频预处理过程，提取目标音频段，并保存到 predict文件夹下
        print("Process audio, save to predict folder")
        self.preprocessor.process_audio(file_list=predict_file_list, save_dir=self.predict_dir)

        assert self.check_predict_dir(), "Predict folder is empty!"

        # 执行模型预测
        prediction = engine.predict(
            model=model,
            ckpt_path=self.ckpt_path,
            data_path=self.predict_dir,
        )
        res = list()
        for _p in prediction:
            res.append([_p.image_path[0], _p.pred_label, _p.pred_score])
        return res

    def predict(self, predict_file):
        """
        对外的预测功能接口
        Args:
            predict_file: 可以是单个音频文件、文件夹、音频文件列表list
        Returns:
        """
        # 判断输入的是音频文件夹、文件列表还是单个音频文件
        if isinstance(predict_file, str):
            if os.path.isdir(predict_file):
                # 是文件夹
                return self.__model_dir_predict(predict_file)
            elif os.path.isfile(predict_file):
                # 是文件
                return self.__model_file_predict(predict_file)
            else:
                print("File to predict does not exist!")
                return -1
        elif isinstance(predict_file, list):
            # 是文件列表
            return self.__model_list_predict(predict_file)

        else:
            raise TypeError(f'Invalid predict_file type: {type(predict_file)}. Expected str or list')


def train():
    # Train the model
    engine.train(datamodule=datamodule, model=model)


if __name__ == '__main__':
    t1 = time.time()

    train()
    # mt = ModelTest(
    #     ckpt_path="/mnt/test/scripts/asd_for_spk/results/Dinomaly/MVTecAD/qzgy_32000/v8/weights/lightning/model.ckpt"
    # )
    # mp = ModelPredict(
    #     ckpt_path="model_ckpts/dinomaly_qzgy_22050_e1000.ckpt",
    #     ref_file="ref/青藏高原片段.wav")
    #
    # t = mt.model_batch_test()
    #
    # p = mp.predict(predict_file=r"E:\异音检测\raw\手动录制\2\3200WG\N32\split\bad\bad")
    # print(p)

    t2 = time.time()
    print("程序运行时间：", t2 - t1)
