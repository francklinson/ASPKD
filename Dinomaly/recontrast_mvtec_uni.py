import copy
import logging
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder

from dataset import MVTecDataset
from dataset import get_data_transforms
from models.de_resnet import de_wide_resnet50_2
from models.resnet import wide_resnet50_2
from models.uad import ReContrast
from optimizers import StableAdamW
from utils import evaluation_batch, global_cosine_hm, replace_layers

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    """
    创建并配置一个日志记录器
    参数:
        name (str): 日志记录器的名称
        save_path (str, optional): 日志文件的保存路径，如果为None则不保存到文件
        level (str, optional): 日志级别，默认为'INFO'
    返回:
        logging.Logger: 配置好的日志记录器
    """
    # 创建指定名称的日志记录器
    logger = logging.getLogger(name)
    # 设置日志级别，通过字符串获取对应的logging级别常量
    logger.setLevel(getattr(logging, level))

    # 定义日志格式，只输出消息内容
    log_format = logging.Formatter('%(message)s')
    # 创建控制台处理器
    streamHandler = logging.StreamHandler()
    # 设置日志格式到控制台处理器
    streamHandler.setFormatter(log_format)
    # 将控制台处理器添加到日志记录器
    logger.addHandler(streamHandler)

    # 如果提供了保存路径
    if not save_path is None:
        # 创建保存路径目录（如果不存在）
        os.makedirs(save_path, exist_ok=True)
        # 创建文件处理器，日志将保存到log.txt文件
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        # 设置日志格式到文件处理器
        fileHandler.setFormatter(log_format)
        # 将文件处理器添加到日志记录器
        logger.addHandler(fileHandler)

    # 返回配置好的日志记录器
    return logger


def count_parameters(model):
    """
    计算模型中需要训练的参数总数
    参数:
        model: 要计算参数的PyTorch模型
    返回:
        int: 模型中需要训练的参数总数
    """
    # 遍历模型中的所有参数，使用生成器表达式计算每个参数的元素数量
    # 只计算requires_grad=True的参数（即需要训练的参数）
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    """
    设置随机种子以确保实验结果的可重复性

    参数:
        seed (int): 随机种子值

    该函数会设置以下库的随机种子:
        - PyTorch (CPU和GPU)
        - NumPy
        - Python内置random模块
    """
    torch.manual_seed(seed)  # 设置PyTorch CPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python random模块的随机种子
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作是确定性的
    torch.backends.cudnn.benchmark = False  # 禁用自动寻找最优算法以提高可重复性


def train(item_list):
    """
    模型训练函数
    参数:
        item_list: 包含训练项目的列表
    """
    # 设置随机种子，确保实验可重复性
    setup_seed(1)

    # 定义训练总迭代次数、批次大小、图像尺寸和裁剪尺寸
    total_iters = 5000  # 总训练迭代次数
    batch_size = 16     # 每个批次的大小
    image_size = 256    # 输入图像的尺寸
    crop_size = 256     # 图像裁剪的尺寸

    # 获取数据转换函数，用于数据预处理
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    # 初始化训练和测试数据列表
    train_data_list = []
    test_data_list = []
    # 遍历item列表，加载每个item的训练和测试数据
    for i, item in enumerate(item_list):
        # 构建训练和测试数据路径
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)

        # 创建训练数据集，使用ImageFolder加载图像数据
        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        # 创建测试数据集，使用MVTecDataset加载测试数据
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    # 合并所有训练数据集
    train_data = ConcatDataset(train_data_list)
    # 创建训练数据加载器，设置批次大小、是否打乱、工作进程数等参数
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)

    # 加载预训练的编码器和批归一化层
    encoder, bn = wide_resnet50_2(pretrained=True)
    # 创建解码器
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)

    # 将ReLU激活函数替换为GELU
    replace_layers(decoder, nn.ReLU, nn.GELU())

    # 将模型移动到指定设备（GPU或CPU）
    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    # 创建编码器的深拷贝版本，用于冻结权重
    encoder_freeze = copy.deepcopy(encoder)

    # 创建ReContrast模型，包含编码器、冻结编码器、批归一化层和解码器
    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)

    # 创建优化器，设置学习率、权重衰减等参数
    optimizer = StableAdamW([{'params': decoder.parameters()}, {'params': bn.parameters()},
                             {'params': encoder.parameters(), 'lr': 1e-5}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-10, amsgrad=True)
    # 创建学习率调度器，在训练过程中调整学习率
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(total_iters * 0.8)], gamma=0.2)

    # 打印训练图像数量
    print_fn('train image number:{}'.format(len(train_data)))

    # 初始化迭代计数器
    it = 0
    # 训练循环
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        # 设置模型为训练模式，编码器和批归一化层不参与训练
        model.train(encoder_bn_train=False)

        # 初始化损失列表
        loss_list = []
        # 遍历训练数据加载器
        for img, label in train_dataloader:
            # 将数据移动到指定设备
            img = img.to(device)
            label = label.to(device)

            # 前向传播，获取编码器和解码器的输出
            en, de = model(img)
            # 计算损失，使用全局余弦相似度
            # loss = global_cosine(en, de)

            # 动态调整alpha参数，从-3逐渐增加到1
            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)
            # 计算带有alpha参数的全局余弦相似度损失
            loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 2 + \
                   global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.) / 2

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            # nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            # 定期评估模型性能
            if (it + 1) % 5000 == 0:
                # 保存模型
                # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

                # 初始化评估指标列表
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                # 评估每个item的性能
                for item, test_data in zip(item_list, test_data_list):
                    # 创建测试数据加载器
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)
                    # 执行评估，获取各项指标
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    # 收集评估结果
                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    # 打印每个item的评估结果
                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                # 打印平均评估结果
                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                # 重新设置模型为训练模式
                model.train(encoder_bn_train=False)

            it += 1
            if it == total_iters:
                break
        # 打印当前迭代次数和平均损失
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    # 保存训练好的模型
    # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../data/mvtec')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_mvtec_uni_max1_it5k_sams2e31e5_b16_s1')
    args = parser.parse_args()

    # item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
    #              'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    item_list = ['carpet', ]
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(f"Using device: {device}!")

    train(item_list)
