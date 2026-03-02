import numpy as np
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
from skimage import measure


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    """
    Compute the PRO score.

    PRO (Region Overlap) score is a metric used for evaluating anomaly detection models.
    It calculates the area under the curve (AUC) of the PRO score versus the false positive rate (FPR).
    Args:
        masks (np.ndarray): Ground truth masks of the anomalies.
        amaps (np.ndarray): Anomaly maps predicted by the model.
        max_step (int): Maximum number of threshold steps to evaluate. Default is 200.
        expect_fpr (float): Expected false positive rate. Default is 0.3.
    Returns:
        float: The PRO AUC score.
    """
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def compute_metrics(gt_sp=None, pr_sp=None, gt_px=None, pr_px=None):
    """
    计算分类和分割任务的评估指标
    参数:
        gt_sp: 图像级别的真实标签 (ground truth for image-level)
        pr_sp: 图像级别的预测概率 (predicted probabilities for image-level)
        gt_px: 像素级别的真实标签 (ground truth for pixel-level)
        pr_px: 像素级别的预测概率 (predicted probabilities for pixel-level)
    返回:
        image_metric: 图像级别的评估指标列表 [AUROC, F1, AP]
        pixel_metric: 像素级别的评估指标列表 [AUROC, F1, AP, AUPRO]
    """
    # classification
    if gt_sp is None or pr_sp is None or gt_sp.sum() == 0 or gt_sp.sum() == gt_sp.shape[0]:
        auroc_sp, f1_sp, ap_sp = 0, 0, 0
    else:
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp)
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])

    # segmentation
    if gt_px is None or pr_px is None or gt_px.sum() == 0:
        auroc_px, f1_px, ap_px, aupro = 0, 0, 0, 0
    else:
        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        aupro = cal_pro_score(gt_px.squeeze(), pr_px.squeeze())

    image_metric = [auroc_sp, f1_sp, ap_sp]
    pixel_metric = [auroc_px, f1_px, ap_px, aupro]

    return image_metric, pixel_metric
