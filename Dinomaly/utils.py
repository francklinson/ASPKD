import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from dataset import MVTecDataset
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from functools import partial
import math

import pickle


def modify_grad(x, inds, factor=0.):
    """
    修改梯度值的函数
    参数:
        x: 输入的张量，将被修改梯度
        inds: 用于选择x中特定位置的索引
        factor: 用于缩放梯度的因子，默认为0.0
    返回:
        修改后的张量x
    """
    # 将索引张量扩展到与输入张量x相同的形状
    inds = inds.expand_as(x)
    # 根据索引选择的位置，将对应的梯度值乘以因子
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):
    """
    对输入张量x的值进行修改，乘以一个因子factor

    参数:
        x (Tensor): 需要被修改的张量
        factor (Tensor): 用于修改x的因子张量

    返回:
        Tensor: 修改后的张量x
    """
    factor = factor.expand_as(x)  # 将factor的形状扩展到与x相同
    x *= factor  # 将x的每个元素乘以factor中对应的元素
    return x  # 返回修改后的x


def global_cosine(a, b, stop_grad=True):
    """
    计算两个张量之间的全局余弦相似度损失
    参数:
        a: 第一个张量列表
        b: 第二个张量列表
        stop_grad: 是否在计算损失时停止梯度传播，默认为True
    返回:
        计算得到的平均余弦相似度损失
    """
    # 初始化余弦相似度计算函数
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0  # 初始化损失值为0
    # 遍历输入张量列表中的每个元素
    for item in range(len(a)):
        # 根据stop_grad参数决定是否停止梯度传播
        if stop_grad:
            # 计算当前元素的余弦相似度损失，并停止a[item]的梯度传播
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                            b[item].view(b[item].shape[0], -1)))
        else:
            # 计算当前元素的余弦相似度损失，保留梯度信息
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
    # 计算所有元素损失的平均值
    loss = loss / len(a)
    return loss


def global_cosine_hm(a, b, alpha=1., factor=0.):
    """
    计算全局余弦相似度损失函数，并根据距离阈值调整梯度
    参数:
        a: 第一个输入张量
        b: 第二个输入张量
        alpha: 控制距离阈值的标准差系数，默认为1.0
        factor: 用于修改梯度的因子，默认为0.0
    返回:
        计算得到的损失值
    """
    # 初始化余弦相似度计算函数
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    # 遍历输入张量的每个元素
    for item in range(len(a)):
        # 分离a的当前元素，防止梯度传播
        a_ = a[item].detach()
        b_ = b[item]
        # 在不计算梯度的情况下计算点间距离
        with torch.no_grad():
            # 计算余弦相似度并转换为距离(1-相似度)
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
            # 计算距离的平均值
        mean_dist = point_dist.mean()
        # 计算距离的标准差
        std_dist = point_dist.reshape(-1).std()

        # 累加全局余弦相似度损失
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        # 计算阈值：平均距离 + alpha倍标准差
        thresh = mean_dist + alpha * std_dist
        # 使用partial函数创建修改梯度的函数，并注册为b_的反向传播钩子
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)
    # 注意：这行代码被注释掉了，原本用于计算平均损失
    # loss = loss / len(a)
    return loss


def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    """
    计算全局余弦相似度损失，并根据距离百分比调整梯度
    参数:
        a: 第一个输入张量
        b: 第二个输入张量
        p: 用于确定距离阈值的百分比 (默认0.9)
        factor: 用于调整梯度的因子 (默认0.)
    返回:
        计算得到的损失值
    """
    # 创建余弦相似度计算对象
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    # 遍历输入张量的每个元素
    for item in range(len(a)):
        # 分离a的梯度，防止反向传播
        a_ = a[item].detach()
        b_ = b[item]
        # 计算点之间的距离 (1 - 余弦相似度)
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def regional_cosine_hm_percent(a, b, p=0.9, factor=0.):
    """
    计算区域余弦相似度损失函数，并根据百分比阈值修改梯度
    参数:
        a: 第一个输入张量，通常为特征图
        b: 第二个输入张量，通常为特征图
        p: 百分比阈值参数，用于确定距离阈值，默认为0.9
        factor: 梯度修改因子，默认为0
    返回:
        计算得到的平均损失值
    """
    # 初始化余弦相似度计算器
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    # 遍历输入张量的每个元素
    for item in range(len(a)):
        # 分离a的当前项，防止梯度回传
        a_ = a[item].detach()
        # 获取b的当前项
        b_ = b[item]
        # 计算点之间的距离（1 - 余弦相似度）
        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # 计算距离的平均值（注释掉的代码）
        # mean_dist = point_dist.mean()
        # 计算距离的标准差（注释掉的代码）
        # std_dist = point_dist.reshape(-1).std()
        # 根据百分比p确定距离阈值
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        # 累加当前项的平均距离
        loss += point_dist.mean()

        # 创建部分函数，用于修改梯度
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        # 为b_注册梯度修改钩子
        b_.register_hook(partial_func)

    # 计算平均损失
    loss = loss / len(a)
    return loss


def global_cosine_focal(a, b, p=0.9, alpha=2., min_grad=0.):
    """
    计算全局余弦焦点损失函数，用于深度学习模型中的损失计算。
    该函数结合了余弦相似度和焦点损失的思想，通过调整梯度来关注难样本。
    参数:
        a (torch.Tensor): 第一个输入张量，通常为模型输出的特征表示
        b (torch.Tensor): 第二个输入张量，通常为目标特征表示
        p (float): 用于确定焦点区域的参数，范围在0到1之间，默认为0.9
        alpha (float): 焦点损失的幂参数，用于调整焦点强度，默认为2.0
        min_grad (float): 梯度的最小值，用于防止梯度过小，默认为0.0
    返回:
        torch.Tensor: 计算得到的损失值
    """
    # 初始化余弦相似度计算器
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0  # 初始化损失值为0
    # 遍历输入张量的每个元素
    for item in range(len(a)):
        # 将a中的当前项分离计算图，防止梯度回传
        a_ = a[item].detach()
        b_ = b[item]
        # 在不计算梯度的情况下计算点之间的距离
        with torch.no_grad():
            # 计算余弦相似度并转换为距离（1-相似度）
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()

        # 根据参数p确定阈值
        if p < 1.:
            # 找到前(1-p)%百分位的距离作为阈值
            thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        else:
            # 如果p=1，则使用最大距离作为阈值
            thresh = point_dist.max()
        # 计算焦点因子，基于距离与阈值的关系
        focal_factor = torch.clip(point_dist, max=thresh) / thresh

        # 应用alpha参数调整焦点因子
        focal_factor = focal_factor ** alpha
        # 设置焦点因子的最小值
        focal_factor = torch.clip(focal_factor, min=min_grad)

        # 计算当前项的余弦相似度损失
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        # 使用partial函数创建修改梯度的函数，并注册为b_的反向传播钩子
        partial_func = partial(modify_grad_v2, factor=focal_factor)
        b_.register_hook(partial_func)

    return loss


def regional_cosine_focal(a, b, p=0.9, alpha=2.):
    """
    计算区域余弦焦点损失函数
    参数:
        a (torch.Tensor): 第一个输入张量
        b (torch.Tensor): 第二个输入张量
        p (float): 控制考虑区域的参数，默认值为0.9
        alpha (float): 焦点损失中的alpha参数，默认值为2
    返回:
        torch.Tensor: 计算得到的损失值
    """
    # 创建余弦相似度计算对象
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0  # 初始化损失值为0
    # 遍历输入张量的每个元素
    for item in range(len(a)):
        # 将a的当前项分离计算图，不参与梯度计算
        a_ = a[item].detach()
        b_ = b[item]  # 获取b的当前项

        # 计算点之间的距离（1减去余弦相似度）
        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # 如果p小于1，计算距离阈值
        if p < 1.:
            # 获取距离的前(1-p)百分位的最大值作为阈值
            thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        else:
            # 如果p大于等于1，使用最大距离作为阈值
            thresh = point_dist.max()
        # 计算焦点因子
        focal_factor = torch.clip(point_dist, max=thresh) / thresh
        focal_factor = focal_factor ** alpha  # 应用alpha次方

        # 累加损失值
        loss += (point_dist * focal_factor.detach()).mean()

    return loss


def regional_cosine_hm(a, b, p=0.9):
    """
    计算区域余弦相似度损失函数
    参数:
        a (torch.Tensor): 第一个输入张量，通常为预测特征
        b (torch.Tensor): 第二个输入张量，通常为目标特征
        p (float): 保留相似度的比例，默认为0.9
    返回:
        torch.Tensor: 计算得到的损失值
    """
    # 创建余弦相似度计算对象
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    # 遍历输入张量的每个元素
    for item in range(len(a)):
        # 分离预测特征，防止梯度回传
        a_ = a[item].detach()
        b_ = b[item]

        # 计算点之间的距离（1减去余弦相似度）
        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # 计算阈值，保留前p比例的距离值
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        # 只保留大于等于阈值的距离值，并计算均值
        L = point_dist[point_dist >= thresh]
        loss += L.mean()

    return loss


def region_cosine(a, b, stop_grad=True):
    """
    计算两个区域特征之间的余弦相似度损失
    参数:
        a (torch.Tensor): 第一个区域的特征表示
        b (torch.Tensor): 第二个区域的特征表示
        stop_grad (bool): 是否停止梯度传播，默认为True
    返回:
        torch.Tensor: 计算得到的损失值
    """
    # 初始化余弦相似度计算函数
    cos_loss = torch.nn.CosineSimilarity()
    # 初始化损失值为0
    loss = 0
    # 遍历所有区域对
    for item in range(len(a)):
        # 计算当前区域对的余弦相似度，并转换为损失值（1-相似度）
        # 使用detach()停止梯度传播（如果stop_grad为True）
        loss += 1 - cos_loss(a[item].detach(), b[item]).mean()
    # 返回总损失值
    return loss


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='add', norm_factor=None):
    """
    计算异常映射图
    参数:
        fs_list: 源特征图列表
        ft_list: 目标特征图列表
        out_size: 输出图像大小，默认为224
        amap_mode: 异常图合并模式，'add'或'mul'，默认为'add'
        norm_factor: 归一化因子，可选参数
    返回:
        anomaly_map: 合并后的异常映射图
        a_map_list: 各层异常映射图列表
    """
    # 如果out_size不是元组，则转换为正方形尺寸
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    # 根据模式初始化异常映射图
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)  # 乘法模式初始化为1
    else:
        anomaly_map = np.zeros(out_size)  # 加法模式初始化为0

    a_map_list = []
    # 遍历特征图列表
    for i in range(len(ft_list)):
        fs = fs_list[i]  # 获取源特征
        ft = ft_list[i]  # 获取目标特征
        # 计算余弦相似度并生成异常图
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)  # 增加维度
        # 双线性插值调整大小
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        # 如果提供了归一化因子，则进行归一化处理
        if norm_factor is not None:
            a_map = 0.1 * (a_map - norm_factor[0][i]) / (norm_factor[1][i] - norm_factor[0][i])

        # 转换为numpy数组并添加到列表
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        # 根据模式合并异常图
        if amap_mode == 'mul':
            anomaly_map *= a_map  # 乘法合并
        else:
            anomaly_map += a_map  # 加法合并
    return anomaly_map, a_map_list  # 返回合并后的异常图和各层异常图列表


def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    """
    计算异常映射图
    参数:
        fs_list (list): 源特征图列表
        ft_list (list): 目标特征图列表
        out_size (int or tuple): 输出图像的大小，默认为224
    返回:
        anomaly_map (torch.Tensor): 计算得到的异常映射图
        a_map_list (list): 每层特征图的异常映射图列表
    """
    # 如果out_size不是元组，则将其转换为正方形尺寸的元组
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []  # 用于存储每层特征图的异常映射图
    for i in range(len(ft_list)):
        fs = fs_list[i]  # 获取源特征图
        ft = ft_list[i]  # 获取目标特征图
        # 计算余弦相似度，并得到异常分数（1-余弦相似度）
        a_map = 1 - F.cosine_similarity(fs, ft)
        # 增加维度，以便进行上采样操作
        a_map = torch.unsqueeze(a_map, dim=1)
        # 使用双线性插值调整异常映射图的大小
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        # 将处理后的异常映射图添加到列表中
        a_map_list.append(a_map)
    # 将所有层的异常映射图沿通道维度拼接，并计算平均值
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list  # 返回最终的异常映射图和各层的异常映射图列表


def map_normalization(fs_list, ft_list, start=0.5, end=0.95):
    """
    对特征图进行归一化处理，计算每个特征图的起始和结束量化值

    参数:
        fs_list (list): 源域特征图列表
        ft_list (list): 目标域特征图列表
        start (float): 起始分位数，默认为0.5
        end (float): 结束分位数，默认为0.95

    返回:
        list: 包含两个列表的列表，分别是每个特征图的起始值列表和结束值列表
    """
    start_list = []  # 存储每个特征图的起始量化值
    end_list = []  # 存储每个特征图的结束量化值
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for i in range(len(ft_list)):  # 遍历特征图列表
            fs = fs_list[i]  # 获取源域特征图
            ft = ft_list[i]  # 获取目标域特征图
            # 计算余弦相似度的补数，值越大表示差异越大
            a_map = 1 - F.cosine_similarity(fs, ft)
            # 计算并存储指定分位数的值
            start_list.append(torch.quantile(a_map, q=start).item())
            end_list.append(torch.quantile(a_map, q=end).item())

    return [start_list, end_list]  # 返回起始值和结束值列表


def cal_anomaly_map_v2(fs_list, ft_list, out_size=224, amap_mode='add'):
    """
    计算异常图，通过比较特征图之间的余弦相似度

    参数:
        fs_list (list): 源特征图列表
        ft_list (list): 目标特征图列表
        out_size (int): 输出异常图的大小，默认为224
        amap_mode (str): 异常图合并模式，默认为'add'

    返回:
        tuple: 包含两个元素的元组
            - anomaly_map (numpy.ndarray): 最终生成的异常图
            - a_map_list (list): 中间过程生成的异常图列表
    """
    a_map_list = []  # 用于存储每个特征对生成的异常图
    for i in range(len(ft_list)):
        fs = fs_list[i]  # 获取源特征图
        ft = ft_list[i]  # 获取目标特征图
        # 计算余弦相似度并转换为异常图(1-相似度)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)  # 增加维度以便进行插值
        # 使用双线性插值调整异常图大小
        a_map = F.interpolate(a_map, size=out_size // 4, mode='bilinear', align_corners=False)
        a_map_list.append(a_map)  # 将生成的异常图添加到列表中

    # 将所有异常图堆叠并沿最后一个维度求和
    anomaly_map = torch.stack(a_map_list, dim=-1).sum(-1)
    # 再次使用双线性插值调整异常图大小到最终输出尺寸
    anomaly_map = F.interpolate(anomaly_map, size=out_size, mode='bilinear', align_corners=False)
    # 提取第一个通道的异常图，并转换为CPU上的numpy数组
    anomaly_map = anomaly_map[0, 0, :, :].to('cpu').detach().numpy()

    return anomaly_map, a_map_list  # 返回最终异常图和中间过程异常图列表


def show_cam_on_image(img, anomaly_map):
    """
    将异常热力图叠加在原始图像上并可视化
    参数:
        img: 原始图像，numpy数组格式
        anomaly_map: 异常热力图，numpy数组格式
    返回:
        叠加后的图像，numpy数组格式，数值范围为0-255
    """
    # 将异常热力图和原始图像都归一化到0-1范围并相加
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    # 对叠加后的图像进行归一化处理，确保最大值为1
    cam = cam / np.max(cam)
    # 将图像缩放到0-255范围并转换为uint8格式
    return np.uint8(255 * cam)


def min_max_norm(image):
    """
    对输入图像进行最小-最大归一化处理，将像素值缩放到[0,1]范围内
    参数:
        image: 输入图像数组，可以是numpy数组或其他支持min/max操作的数组
    返回:
        归一化后的图像数组，像素值范围在[0,1]之间
    """
    # 计算图像中的最小像素值
    a_min, a_max = image.min(), image.max()
    # 应用最小-最大归一化公式: (image - min) / (max - min)
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    """
    将灰度图像转换为热力图
    参数:
        gray: 输入的灰度图像数据
    返回:
        heatmap: 应用热力图颜色映射后的图像
    """
    # 使用OpenCV的applyColorMap函数将灰度图像转换为热力图
    # np.uint8(gray)确保输入数据是8位无符号整数格式
    # cv2.COLORMAP_JET使用Jet颜色映射方案，这是一种常用的热力图配色方案
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def return_best_thr(y_true, y_score):
    """
    根据真实标签和预测分数，找到使F1分数最大的最佳阈值
    参数:
        y_true (array-like): 真实标签数组，通常为0和1
        y_score (array-like): 预测分数数组，表示样本为正类的概率或置信度
    返回:
        float: 使F1分数最大的最佳阈值
    计算过程:
        1. 使用precision_recall_curve计算精确率、召回率和对应的阈值
        2. 根据精确率和召回率计算F1分数
        3. 去除无效的F1分数(NaN值)及其对应的阈值
        4. 找到使F1分数最大的阈值并返回
    """
    # 计算精确率、召回率和对应的阈值
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    # 计算F1分数，添加1e-7避免除以0
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    # 去除最后一个F1分数(对应阈值列表之外的值)
    f1s = f1s[:-1]
    # 去除F1分数为NaN的值及其对应的阈值
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    # 找到使F1分数最大的阈值
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def f1_score_max(y_true, y_score):
    """
    计算F1分数的最大值
    参数:
    y_true: 真实标签数组
    y_score: 预测得分数组
    返回:
    float: F1分数的最大值
    """
    # 计算精确率和召回率，同时获得阈值
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    # 计算F1分数，添加1e-7防止除以0
    # 公式: F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    # 去掉最后一个元素(对应阈值为1的情况)
    f1s = f1s[:-1]
    # 返回F1分数的最大值
    return f1s.max()


def specificity_score(y_true, y_score):
    """
    计算特异性分数(Specificity)，也称为真负率(True Negative Rate)
    特异性衡量的是在所有实际为负类的样本中，被正确预测为负类的比例
    参数:
    y_true -- 真实标签数组，包含实际的类别标签(0和1)
    y_score -- 预测分数数组，包含模型预测的分数(0表示负类，1表示正类)
    返回:
    float -- 特异性分数，范围在[0,1]之间
    """
    # 将输入转换为numpy数组以便进行数组操作
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # 计算真负值(TN)数量：预测为负且实际也为负的样本数
    TN = (y_true[y_score == 0] == 0).sum()
    # 计算实际负值(N)总数：所有实际为负的样本数
    N = (y_true == 0).sum()
    # 返回特异性分数：真负值数量除以实际负值总数
    return TN / N


def evaluation(model, dataloader, device, _class_=None, calc_pro=True, norm_factor=None, feature_used='all',
               max_ratio=0):
    """
    模型评估函数，用于计算异常检测模型的性能指标

    参数:
        model: 待评估的模型
        dataloader: 数据加载器
        device: 计算设备 (CPU/GPU)
        _class_: 类别信息 (未使用)
        calc_pro: 是否计算PRO指标
        norm_factor: 异常图归一化因子
        feature_used: 使用哪些特征层 ('trained', 'freezed', 'all')
        max_ratio: 计算sp_score时的最大比例

    返回:
        auroc_px: 像素级AUROC
        auroc_sp: 图像级AUROC
        aupro: 平均PRO得分
    """
    model.eval()  # 将模型设置为评估模式
    gt_list_px = []  # 存储像素级真实标签
    pr_list_px = []  # 存储像素级预测分数
    gt_list_sp = []  # 存储图像级真实标签
    pr_list_sp = []  # 存储图像级预测分数
    aupro_list = []  # 存储PRO得分列表

    with torch.no_grad():  # 禁用梯度计算
        for img, gt, label, _ in dataloader:  # 遍历数据加载器
            img = img.to(device)  # 将图像数据移动到指定设备

            en, de = model(img)  # 获取模型编码器和解码器的输出

            # 根据feature_used参数选择不同的特征层计算异常图
            if feature_used == 'trained':
                anomaly_map, _ = cal_anomaly_map(en[3:], de[3:], img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            elif feature_used == 'freezed':
                anomaly_map, _ = cal_anomaly_map(en[:3], de[:3], img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            else:
                anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # 对异常图进行高斯滤波
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()

            if calc_pro:
                if label.item() != 0:
                    aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                                  anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            if max_ratio <= 0:
                sp_score = anomaly_map.max()
            else:
                anomaly_map = anomaly_map.ravel()
                sp_score = np.sort(anomaly_map)[-int(anomaly_map.shape[0] * max_ratio):]
                sp_score = sp_score.mean()
            pr_list_sp.append(sp_score)
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 4)


def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None):
    """
    执行批量分析，评估模型在给定数据上的性能

    :param model: 模型
    :param dataloader: 数据加载器
    :param device: 设备
    :param _class_: 类别
    :param max_ratio: 最大比例
    :param resize_mask: 调整大小
    :return:

    """
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            # starter.record()
            output = model(img)
            # ender.record()
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            en, de = output[0], output[1]

            # 计算anomaly_maps
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])

            # anomaly_map = anomaly_map - anomaly_map.mean(dim=[1, 2, 3]).view(-1, 1, 1, 1)
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        aupro_px = compute_pro(gt_list_px, pr_list_px)

        gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

        auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_px = average_precision_score(gt_list_px, pr_list_px)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, gt_list_sp, pr_list_sp]


def evaluation_batch_loco(model, dataloader, device, _class_=None, max_ratio=0):
    """
    在LOCO数据集上批量评估模型的性能

    参数:
        model: 要评估的模型
        dataloader: 数据加载器，包含图像、标签等信息
        device: 计算设备(CPU/GPU)
        _class_: 类别参数(未使用)
        max_ratio: 计算sp_score时考虑的最大像素比例

    返回:
        auroc_sp: 整体图像级AUROC分数
        auroc_logic: 逻辑异常类的AUROC分数
        auroc_struct: 结构异常类的AUROC分数
        auroc_both: 逻辑和结构异常类的平均AUROC分数
    """
    model.eval()  # 将模型设置为评估模式

    # 初始化各种评估指标的列表
    gt_list_px = []  # 像素级真实标签列表
    pr_list_px = []  # 像素级预测结果列表
    gt_list_sp = []  # 图像级真实标签列表
    pr_list_sp = []  # 图像级预测结果列表
    defect_type_list = []  # 缺陷类型列表
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)  # 获取高斯核

    with torch.no_grad():  # 不计算梯度，节省内存
        for img, gt, label, path, defect_type, size in dataloader:
            img = img.to(device)  # 将图像移动到指定设备

            output = model(img)  # 模型前向传播
            en, de = output[0], output[1]  # 获取编码器和解码器的输出

            # 计算异常图
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)  # 应用高斯平滑

            gt = gt.bool()  # 将真实标签转换为布尔类型

            # 收集像素级评估数据
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.extend(label.cpu().numpy().astype(int))

            # 计算图像级预测分数
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0].cpu().numpy()
            else:
                anomaly_map = anomaly_map.flatten(1)
                # 只考虑前max_ratio比例的最高分数
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1).cpu().numpy()
            pr_list_sp.extend(sp_score)
            defect_type_list.extend(defect_type)

        # 计算各种评估指标
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)  # 像素级AUROC
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)  # 图像级AUROC
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)  # 像素级AP
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)  # 图像级AP

        # 按缺陷类型计算AUROC
        defect_type_list = np.array(defect_type_list)
        auroc_logic = roc_auc_score(
            np.array(gt_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')],
            np.array(pr_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')])
        auroc_struct = roc_auc_score(
            np.array(gt_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')],
            np.array(pr_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')])
        auroc_both = (auroc_logic + auroc_struct) / 2  # 计算平均AUROC

    return auroc_sp, auroc_logic, auroc_struct, auroc_both


def evaluation_uniad(model, dataloader, device, _class_=None, reg_calib=False, max_ratio=0):
    """
    评估UniAD模型的性能函数

    参数:
        model: 待评估的模型
        dataloader: 数据加载器，包含测试数据
        device: 计算设备(CPU/GPU)
        _class_: 类别信息(本实现中未使用)
        reg_calib: 是否使用校准回归
        max_ratio: 用于计算空间得分的最大比例

    返回:
        auroc_px: 像素级AUROC分数
        auroc_sp: 图像级AUROC分数
        ap_px: 像素级AP分数
        ap_sp: 图像级AP分数
        [gt_list_px, pr_list_px, gt_list_sp, pr_list_sp]: 真实值和预测值列表
    """
    model.eval()  # 将模型设置为评估模式

    # 初始化用于存储真实值和预测值的列表
    gt_list_px = []  # 像素级真实值
    pr_list_px = []  # 像素级预测值
    gt_list_sp = []  # 图像级真实值
    pr_list_sp = []  # 图像级预测值
    aupro_list = []  # AUROC列表(本实现中未使用)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)  # 创建高斯核

    with torch.no_grad():  # 禁用梯度计算
        for img, gt, label, _ in dataloader:  # 遍历数据加载器
            img = img.to(device)  # 将图像移动到指定设备
            # 根据是否校准选择不同的模型输出
            if reg_calib:
                en, de, reg = model({'image': img})  # 获取编码器、解码器和回归器输出
            else:
                en, de = model({'image': img})  # 仅获取编码器和解码器输出

            # 计算异常图(MSE损失)
            anomaly_map = torch.mean(F.mse_loss(de, en, reduction='none'), dim=1, keepdim=True)
            # 双线性插值调整异常图大小
            anomaly_map = F.interpolate(anomaly_map, size=(img.shape[-1], img.shape[-1]), mode='bilinear',
                                        align_corners=False)

            # 如果启用校准，则对异常图进行校准
            if reg_calib:
                if reg.shape[1] == 2:  # 如果回归器输出两个值(均值和最大值)
                    reg_mean = reg[:, 0].view(-1, 1, 1, 1)
                    reg_max = reg[:, 1].view(-1, 1, 1, 1)
                    anomaly_map = (anomaly_map - reg_mean) / (reg_max - reg_mean)
                    # anomaly_map = anomaly_map - reg_max

                else:
                    reg = F.interpolate(reg, size=img.shape[-1], mode='bilinear', align_corners=True)
                    anomaly_map = anomaly_map - reg

            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.extend(label.cpu().numpy().astype(int))

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0].cpu().numpy()
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1).cpu().numpy()
            pr_list_sp.extend(sp_score)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, ap_px, ap_sp, [gt_list_px, pr_list_px, gt_list_sp, pr_list_sp]


def save_feature(model, dataloader, device, _class_='None', save_name='save'):
    """
    保存模型提取的特征

    参数:
        model: 要提取特征的模型
        dataloader: 数据加载器，包含图像、标签等信息
        device: 计算设备(CPU/GPU)
        _class_: 类别标签，默认为'None'
        save_name: 保存特征的目录名称，默认为'save'
    """
    model.eval()  # 将模型设置为评估模式
    # 创建特征保存目录
    save_dir = os.path.join('./feature', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 关闭梯度计算，节省内存
    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        for img, gt, label, img_path in dataloader:
            img = img.to(device)  # 将图像移动到指定设备
            en, de = model(img)  # 通过模型获取编码器和解码器的输出

            # 初始化用于存储异常和正常特征的列表
            en_abnorm_list = []
            en_normal_list = []
            de_abnorm_list = []
            de_normal_list = []

            # 遍历3个不同尺度的特征
            for i in range(3):
                # 获取编码器和解码器的特征
                en_feat = en[0 + i]
                de_feat = de[0 + i]

                # 调整ground truth尺寸以匹配特征图尺寸
                gt_resize = F.interpolate(gt, size=en_feat.shape[2], mode='bilinear') > 0

                # 提取异常和正常区域的特征
                en_abnorm = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                en_normal = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                de_abnorm = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                de_normal = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                # 标准化特征并移动到CPU，然后转换为numpy数组
                en_abnorm_list.append(F.normalize(en_abnorm, dim=1).cpu().numpy())
                en_normal_list.append(F.normalize(en_normal, dim=1).cpu().numpy())
                de_abnorm_list.append(F.normalize(de_abnorm, dim=1).cpu().numpy())
                de_normal_list.append(F.normalize(de_normal, dim=1).cpu().numpy())

            # 创建类别特定的保存目录
            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            # 从图像路径生成文件名
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')

            # 准备要保存的字典
            saved_dict = {'en_abnorm_list': en_abnorm_list, 'en_normal_list': en_normal_list,
                          'de_abnorm_list': de_abnorm_list, 'de_normal_list': de_normal_list}

            # 将特征保存到pickle文件
            with open(save_dir_class + '/' + name + '.pkl', 'wb') as f:
                pickle.dump(saved_dict, f)

    return


def visualize(model, dataloader, device, _class_='None', save_name='save'):
    """
    可视化模型预测结果并保存图像

    参数:
        model: 要评估的模型
        dataloader: 数据加载器
        device: 计算设备 (CPU/GPU)
        _class_: 数据类别，默认为'None'
        save_name: 保存结果的目录名，默认为'save'
    """
    model.eval()  # 将模型设置为评估模式
    save_dir = os.path.join('./visualize', save_name)  # 设置保存目录
    if not os.path.exists(save_dir):  # 如果目录不存在则创建
        os.makedirs(save_dir)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)  # 获取高斯核并移动到指定设备

    with torch.no_grad():  # 禁用梯度计算
        for img, gt, label, img_path in dataloader:  # 遍历数据加载器
            img = img.to(device)  # 将图像移动到指定设备
            output = model(img)  # 获取模型输出
            en, de = output[0], output[1]  # 分离编码器和解码器的输出
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])  # 计算异常图
            anomaly_map = gaussian_kernel(anomaly_map)  # 应用高斯核平滑异常图

            # 批量处理图像，每次处理8张
            for i in range(0, anomaly_map.shape[0], 8):
                # 归一化异常图并转换为热力图
                heatmap = min_max_norm(anomaly_map[i, 0].cpu().numpy())
                heatmap = cvt2heatmap(heatmap * 255)

                # 处理原始图像
                im = img[i].permute(1, 2, 0).cpu().numpy()  # 调整维度并转到CPU
                im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # 反标准化
                im = (im * 255).astype('uint8')  # 转换为8位整数
                im = im[:, :, ::-1]  # BGR转RGB
                # 将热力图叠加到原始图像上
                hm_on_img = show_cam_on_image(im, heatmap)
                # 处理真实掩码
                mask = (gt[i][0].numpy() * 255).astype('uint8')
                # 创建类别保存目录
                save_dir_class = os.path.join(save_dir, str(_class_))
                if not os.path.exists(save_dir_class):
                    os.mkdir(save_dir_class)
                # 生成文件名
                name = img_path[i].split('/')[-2] + '_' + img_path[i].split('/')[-1].replace('.png', '')
                # 保存原始图像、热力图和真实掩码
                cv2.imwrite(save_dir_class + '/' + name + '_img.png', im)
                cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)
                cv2.imwrite(save_dir_class + '/' + name + '_gt.png', mask)
    return


def visualize_noseg(model, dataloader, device, _class_='None', save_name='save'):
    """
    可视化无分割结果的异常检测模型输出

    参数:
        model: 要可视化的模型
        dataloader: 数据加载器，包含要可视化的图像
        device: 计算设备 (CPU/GPU)
        _class_: 类别名称，用于保存结果的文件夹命名
        save_name: 保存结果的根目录名称

    返回:
        None
    """
    model.eval()  # 将模型设置为评估模式，关闭dropout和batch normalization
    save_dir = os.path.join('./visualize', save_name)  # 创建保存结果的目录路径
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建
        os.mkdir(save_dir)
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for img, label, img_path in dataloader:  # 遍历数据加载器中的图像
            img = img.to(device)  # 将图像移动到指定设备
            en, de = model(img)  # 通过模型获取编码器和解码器的输出

            # 计算异常图，使用a模式
            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # 对异常图进行高斯滤波

            heatmap = min_max_norm(anomaly_map)  # 对异常图进行最小-最大归一化
            heatmap = cvt2heatmap(heatmap * 255)  # 将归一化的异常图转换为热力图格式
            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]  # 调整图像维度并转换为numpy数组
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # 反标准化处理
            img = (img * 255).astype('uint8')  # 将图像像素值转换为0-255范围的无符号8位整数
            hm_on_img = show_cam_on_image(img, heatmap)  # 将热力图叠加到原始图像上

            save_dir_class = os.path.join(save_dir, str(_class_))  # 创建按类别分类的保存目录
            if not os.path.exists(save_dir_class):  # 如果目录不存在，则创建
                os.mkdir(save_dir_class)
            # 从图像路径中提取名称，用于保存文件
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')
            # 保存热力图和叠加热力图的图像
            cv2.imwrite(save_dir_class + '/' + name + '_seg.png', heatmap)
            cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)

    return


def visualize_loco(model, dataloader, device, _class_='None', save_name='save'):
    """
    可视化LOCO数据集中的异常检测结果

    参数:
        model: 要评估的模型
        dataloader: 数据加载器，包含测试图像
        device: 计算设备(CPU或GPU)
        _class_: 类别名称，默认为'None'
        save_name: 保存结果的目录名，默认为'save'
    """
    model.eval()  # 将模型设置为评估模式
    save_dir = os.path.join('./visualize', save_name)  # 设置保存结果的根目录
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for img, gt, label, img_path, defect_type, size in dataloader:  # 遍历数据加载器
            img = img.to(device)  # 将图像移动到指定设备
            en, de = model(img)  # 前向传播，获取编码器和解码器的输出

            # 计算异常图
            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # 应用高斯滤波
            # 调整异常图大小以匹配原始图像尺寸
            anomaly_map = cv2.resize(anomaly_map, dsize=(size[0].item(), size[1].item()),
                                     interpolation=cv2.INTER_NEAREST)

            # 设置保存路径
            save_dir_class = os.path.join(save_dir, str(_class_), 'test', defect_type[0])
            if not os.path.exists(save_dir_class):  # 如果目录不存在则创建
                os.makedirs(save_dir_class)
            name = img_path[0].split('/')[-1].replace('.png', '')  # 从图像路径中提取文件名
            # 保存异常图
            cv2.imwrite(save_dir_class + '/' + name + '.tiff', anomaly_map)
    return


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df._append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def replace_layers(model, old, new):
    """
    递归遍历模型中的所有层，将指定类型的旧层替换为新层
    参数:
        model: 要处理的神经网络模型
        old: 需要被替换的旧层类型
        new: 用于替换的新层类型
    """
    for n, module in model.named_children():
        # 检查当前模块是否包含子模块
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)


class FeatureJitter(torch.nn.Module):
    def __init__(self, scale=1., p=0.25) -> None:

        """
        初始化特征抖动模块

        参数:
            scale (float): 抖动的强度缩放因子，默认为1.0
            p (float): 应用抖动的概率，默认为0.25
        """
        super(FeatureJitter, self).__init__()  # 调用父类的初始化方法
        self.scale = scale  # 设置抖动强度
        self.p = p  # 设置应用抖动的概率

    def add_jitter(self, feature):
        """
        为特征添加随机抖动
        参数:
            feature (torch.Tensor): 输入的特征张量，形状为(B, C, H, W)
        返回:
            torch.Tensor: 添加抖动后的特征张量
        """
        if self.scale > 0:
            B, C, H, W = feature.shape  # 获取特征图的批大小、通道数、高度和宽度
            # 计算每个空间位置的特征范数，并归一化
            feature_norms = feature.norm(dim=1).unsqueeze(1) / C  # B*1*H*W
            # 生成随机抖动噪声
            jitter = torch.randn((B, C, H, W), device=feature.device)
            # 对抖动噪声进行归一化
            jitter = F.normalize(jitter, dim=1)
            # 根据特征范数和缩放因子调整抖动强度
            jitter = jitter * feature_norms * self.scale
            # 创建随机掩码，决定哪些位置应用抖动
            mask = torch.rand((B, 1, H, W), device=feature.device) < self.p
            # 只在掩码为True的位置添加抖动
            feature = feature + jitter * mask
        return feature

    def forward(self, x):

        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入的特征张量

        返回:
            torch.Tensor: 处理后的特征张量，仅在训练模式下应用抖动
        """
        if self.training:  # 只在训练模式下应用特征抖动
            x = self.add_jitter(x)
        return x


class WarmCosineScheduler(_LRScheduler):
    """
    带有预热阶段和余弦退学习率的调度器类
    这种学习率调度器结合了线性预热和余弦退火两种策略，先进行线性预热，
    然后按照余弦函数曲线逐渐降低学习率到最终值。
    参数:
        optimizer: 优化的优化器
        base_value: 基础学习率值
        final_value: 最终学习率值
        total_iters: 总迭代次数
        warmup_iters: 预热阶段的迭代次数，默认为0
        start_warmup_value: 预热阶段的起始学习率值，默认为0
    """

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        # 设置最终学习率和总迭代次数
        self.final_value = final_value
        self.total_iters = total_iters
        # 创建预热阶段的学习率调度（线性增长）
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        # 创建余弦退火阶段的学习率调度
        iters = np.arange(total_iters - warmup_iters)
        # 使用余弦函数计算学习率变化
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        # 将预热阶段和余弦退火阶段的学习率调度合并
        self.schedule = np.concatenate((warmup_schedule, schedule))

        # 调用父类初始化方法
        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        """
        获取当前迭代次数对应的学习率
        返回:
            list: 当前每个参数组的学习率列表
        """
        # 如果当前迭代次数超过总迭代次数，返回最终学习率
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            # 返回当前迭代次数对应的学习率
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]
