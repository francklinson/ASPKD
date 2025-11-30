import argparse
import logging
import os
import random
import time
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder

from dataset import MVTecDataset
from dataset import get_data_transforms
from dinov1.utils import trunc_normal_
from dinov3.hub.backbones import load_dinov3_model
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from optimizers import StableAdamW
from utils import evaluation_batch, global_cosine_hm_percent, \
    WarmCosineScheduler
from utils import visualize

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
    # 设置日志记录器的级别
    logger.setLevel(getattr(logging, level))

    # 定义日志格式
    log_format = logging.Formatter('%(message)s')
    # 创建控制台处理器
    streamHandler = logging.StreamHandler()
    # 设置控制台处理器的日志格式
    streamHandler.setFormatter(log_format)
    # 将控制台处理器添加到日志记录器
    logger.addHandler(streamHandler)

    # 如果指定了日志保存路径
    if not save_path is None:
        # 创建保存路径目录（如果不存在）
        os.makedirs(save_path, exist_ok=True)
        # 创建文件处理器
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        # 设置文件处理器的日志格式
        fileHandler.setFormatter(log_format)
        # 将文件处理器添加到日志记录器
        logger.addHandler(fileHandler)

    # 返回配置好的日志记录器
    return logger


def count_parameters(model):
    """
    计算模型中需要训练的参数总数
    参数:
        model: 要计算参数的神经网络模型
    返回:
        int: 模型中需要训练的参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    """
    设置随机种子以确保实验结果的可复现性

    参数:
        seed (int): 随机种子值

    该函数会设置以下库的随机种子:
        - PyTorch (CPU和GPU)
        - NumPy
        - Python内置random模块
    """
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch的所有GPU随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python内置random模块的随机种子
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作是确定性的
    torch.backends.cudnn.benchmark = False  # 禁用自动寻找最优算法，以确保确定性


def train(item_list, model_size="base"):
    """
    训练模型
    Args:
        item_list:
        model_size:

    Returns:

    """
    # 设置随机种子，保证实验可重复性
    setup_seed(1)

    # 根据模型大小选择参数
    assert model_size in ["base", "large"]
    if model_size == "base":
        # 设置模型参数
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]  # 目标层列表
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 编码器融合层
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 解码器融合层
        batch_size = 16  # 批次大小

        # 设置编码器名称和权重路径
        encoder_name = 'dinov3_vitb16'
        encoder_weight = 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    elif model_size == "large":
        # 定义目标层和融合层
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        batch_size = 8  # 批次大小

        # 定义编码器名称和权重路径
        encoder_name = 'dinov3_vitl16'
        encoder_weight = 'weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'

    # 设置训练参数
    total_iters = 100  # 总训练迭代次数
    image_size = 512  # 输入图像尺寸
    crop_size = 448  # 裁剪尺寸

    # 获取数据变换方法
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    # 初始化训练和测试数据列表
    train_data_list = []
    test_data_list = []

    # 遍历数据集项目列表，加载训练和测试数据
    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')  # 训练数据路径
        test_path = os.path.join(args.data_path, item)  # 测试数据路径

        # 加载训练数据集
        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]
        train_data_list.append(train_data)

        # 加载测试数据集
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        test_data_list.append(test_data)

    # 合并训练数据集并创建数据加载器
    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)

    # 加载预训练编码器模型
    encoder = load_dinov3_model(encoder_name, layers_to_extract_from=target_layers,
                                pretrained_weight_path=encoder_weight)

    # 根据编码器名称设置模型参数
    if 'vits' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'vitb' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'vitl' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise "Architecture not in vits, vitb, vitl."

    # 初始化瓶颈层和解码器
    bottleneck = []
    decoder = []

    # 添加瓶颈层模块
    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)

    # 添加解码器模块
    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    # 创建完整模型并移动到指定设备
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)

    # 设置可训练模块
    trainable = nn.ModuleList([bottleneck, decoder])

    # 初始化模型参数
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # 设置优化器和学习率调度器
    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=False, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    # 打印训练数据数量
    print_fn('train image number:{}'.format(len(train_data)))

    # 开始训练循环
    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        model.encoder.eval()

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)
            en, de = model(img)

            # 计算损失，使用动态的百分比参数
            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            # 一定次数循环后进行模型效果验证
            if (it + 1) % 1000 == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)
                    # 调用evaluation_batch 函数进行evaluate
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, gt_list_sp, pr_list_sp = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                    # 做可视化,按照时间保存
                    visualize(model, test_dataloader, device, _class_=item,
                              save_name=f"model_size_{model_size}_epoch_{it + 1}")

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                model.train()
                model.encoder.eval()

            it += 1
            if it == total_iters:
                break

        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
    # 保存模型
    # 如果没有路径就新建一个
    model_save_pth = f"saved_results/{args.save_name}/Dinomaly_{model_size}_epoch_{total_iters}_{time.ctime()}.pth"
    if not os.path.exists(f"saved_results/{args.save_name}"):
        os.makedirs(f"saved_results/{args.save_name}")
    torch.save(model.state_dict(), model_save_pth)
    print_fn("Model saved to {}!".format(model_save_pth))


def model_test(model_path, model_size="base"):
    """
    测试模型
    Returns:

    """
    # 根据模型大小选择参数
    assert model_size in ["base", "large"]
    if model_size == "base":
        # 设置模型参数
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]  # 目标层列表
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 编码器融合层
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 解码器融合层
        batch_size = 16  # 批次大小

        # 设置编码器名称和权重路径
        encoder_name = 'dinov3_vitb16'
        encoder_weight = 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    elif model_size == "large":
        # 定义目标层和融合层
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        batch_size = 8  # 批次大小

        # 定义编码器名称和权重路径
        encoder_name = 'dinov3_vitl16'
        encoder_weight = 'weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'

    # 设置参数
    image_size = 512  # 输入图像尺寸
    crop_size = 448  # 裁剪尺寸

    # 获取数据变换方法
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    # 初始化测试数据列表
    test_data_list = []

    # 遍历数据集项目列表，加载训练和测试数据
    for i, item in enumerate(item_list):
        test_path = os.path.join(args.data_path, item)  # 测试数据路径

        # 加载测试数据集
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        test_data_list.append(test_data)

    # 加载预训练编码器模型
    encoder = load_dinov3_model(encoder_name, layers_to_extract_from=target_layers,
                                pretrained_weight_path=encoder_weight)

    # 根据编码器名称设置模型参数
    if 'vits' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'vitb' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'vitl' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise "Architecture not in vits, vitb, vitl."

    # 初始化瓶颈层和解码器
    bottleneck = []
    decoder = []

    # 添加瓶颈层模块
    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)

    # 添加解码器模块
    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    # 创建完整模型并移动到指定设备
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)

    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    for item, test_data in zip(item_list, test_data_list):
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                      num_workers=4)
        # 调用evaluation_batch 函数进行evaluate
        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, gt_list_sp, pr_list_sp = results

        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)

        print_fn(
            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

        print_fn('gt: {} \n pr: {}'.format(gt_list_sp, pr_list_sp))

    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
            np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../data/mvtec')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='vitill_mvtec_uni_dinov3')
    args = parser.parse_args()

    # item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
    #              'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    item_list = ['toothbrush', ]

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(f"Using device: {device}")

    # 执行训练
    train(item_list, model_size="base")

    # 执行测试
    # model_test(model_path="saved_results/vitill_mvtec_uni_dinov3/Dinomaly_base_epoch_100_Sun Nov 30 15:03:54 2025.pth",
    #            model_size="base")
