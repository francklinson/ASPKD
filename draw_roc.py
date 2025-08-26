import numpy as np
import pandas
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def draw_roc(y_true, y_score):
    """
    绘制roc图
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("Ture Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc='lower right')
    plt.show()


def draw_multi_rocs(y_true, y_score_matrix: pd.DataFrame):
    """
    绘制多条roc图
    """

    plt.figure()
    for i in y_score_matrix.columns:
        y_score = y_score_matrix[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        # 用不同的线条颜色
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve {i} (area = %0.2f)' % roc_auc, color=np.random.rand(3, ))
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlim([-0.05, 1.1])
        plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("Ture Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # pd = pandas.read_csv("DenseAE/result/anomaly_score_spk_id_02.csv")
    # pd = pandas.read_csv("Convolutional AE/result/anomaly_score_spk_id_02.csv")
    # pd = pandas.read_csv("STgram-MFN/results/STgram-MFN(m=0.7,s=30)/anomaly_score_spk_id_02.csv")
    # score = pd.iloc[:, 1]
    # file = pd.iloc[:, 0]
    # gt = []
    # for f in file:
    #     if 'normal' in f:
    #         gt.append(0)
    #     else:
    #         gt.append(1)
    # draw_roc(y_true=gt, y_score=score)

    # pd = pandas.read_csv("GenRepASD/out/dcase2020_test/patch_diff_beats_ft1/GenRep_model_namebeats_ft1_topk1_trainsplittrain_evalsplitvalid_poolingtemporal_num_samples50_seed42/log_20250408-135613/layer10result.csv")
    # gt = pd.iloc[:,0]
    # score_w_norm = pd.iloc[:,1]
    # scores_wo_norm = pd.iloc[:,2]
    # draw_roc(y_true=gt, y_score=score_w_norm)
    # draw_roc(y_true=gt, y_score=scores_wo_norm)

    pd = pandas.read_csv("2020/04-AEGAN-AD/results/spk_id_spk_id_1_results.csv")
    gt = pd.iloc[:, 0]
    score = pd.iloc[:, 1:]
    draw_multi_rocs(gt, score)
