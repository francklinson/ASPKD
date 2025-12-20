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
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder

from dataset import MVTecDataset
from dataset import get_data_transforms
from dinov1.utils import trunc_normal_
from dinov3.hub.backbones import load_dinov3_model
from models.uad import ViTill, ViTillDinoV2
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from optimizers import StableAdamW
from utils import evaluation_batch, global_cosine_hm_percent, WarmCosineScheduler
from utils import visualize

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    """创建并配置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DinomalyBaseTrainer:
    def __init__(self, model_size="small", device='cuda:0'):
        self.model_size = model_size
        self.device = device
        self.logger = None
        self.encoder = None
        self.model = None
        self.trainable = None
        self.setup_model_config()

    def setup_model_config(self):
        """设置模型配置参数"""
        assert self.model_size in ["small", "base", "large"]

        if self.model_size == "small":
            self.embed_dim, self.num_heads = 384, 6
            self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        elif self.model_size == "base":
            self.embed_dim, self.num_heads = 768, 12
            self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        else:  # large
            self.embed_dim, self.num_heads = 1024, 16
            self.target_layers = [4, 6, 8, 10, 12, 14, 16, 18]

        self.fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    def create_model_components(self):
        """创建模型组件"""
        bottleneck = nn.ModuleList([bMlp(self.embed_dim, self.embed_dim * 4, self.embed_dim, drop=0.2)])

        decoder = nn.ModuleList([
            VitBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=4.,
                     qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                     attn=LinearAttention2)
            for _ in range(8)
        ])

        return bottleneck, decoder

    def initialize_weights(self):
        """初始化模型权重"""
        for m in self.trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def setup_optimizer(self, lr=2e-3):
        """设置优化器"""
        return StableAdamW(
            [{'params': self.trainable.parameters()}],
            lr=lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10
        )

    def setup_scheduler(self, optimizer, total_iters, warmup_iters=100):
        """设置学习率调度器"""
        return WarmCosineScheduler(
            optimizer, base_value=2e-3, final_value=2e-4,
            total_iters=total_iters, warmup_iters=warmup_iters
        )

    def evaluate_model(self, item_list, test_data_list, batch_size=16):
        """评估模型"""
        auroc_sp_list, ap_sp_list, f1_sp_list, gt_sp_list, pr_sp_list = [], [], [], [], []

        for item, test_data in zip(item_list, test_data_list):
            test_dataloader = torch.utils.data.DataLoader(
                test_data, batch_size=batch_size, shuffle=False, num_workers=4
            )
            results = evaluation_batch(self.model, test_dataloader, self.device, max_ratio=0.01, resize_mask=256)
            auroc_sp, ap_sp, f1_sp, gt_sp, pr_sp = results

            auroc_sp_list.append(auroc_sp)
            ap_sp_list.append(ap_sp)
            f1_sp_list.append(f1_sp)
            gt_sp_list.extend(gt_sp)
            pr_sp_list.extend(pr_sp)

            self.logger.info(f'{item}: I-Auroc:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f}')

        self.logger.info(f'Mean: I-Auroc:{np.mean(auroc_sp_list):.4f}, '
                         f'I-AP:{np.mean(ap_sp_list):.4f}, I-F1:{np.mean(f1_sp_list):.4f}')
        # 计算AUROC
        fpr, tpr, thresholds = metrics.roc_curve(gt_sp_list, pr_sp_list)
        print("fpr: ", fpr)
        print("tpr: ", tpr)
        print("thresholds: ", thresholds)
        print("AUROC: ", metrics.auc(fpr, tpr))
        # 绘制ROC曲线
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        # 保存
        plt.savefig('roc_curve.png')

        return auroc_sp_list, ap_sp_list, f1_sp_list, gt_sp_list, pr_sp_list

    def save_model(self, save_dir, save_name, item_list, total_iters):
        """保存模型"""
        item_list_str = "_".join(item_list) + "_"
        model_save_pth = os.path.join(
            save_dir, f"{save_name}_{self.model_size}_{item_list_str}epoch_{total_iters}_{time.ctime()}.pth"
        )
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), model_save_pth)
        self.logger.info(f"Model saved to {model_save_pth}!")

    def plot_roc_curve(self, gt_sp_list, pr_sp_list):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = metrics.roc_curve(gt_sp_list, pr_sp_list)
        self.logger.info(f"AUROC: {metrics.auc(fpr, tpr):.4f}")

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig('roc_curve.png')

    def prepare_test_data(self, item_list, data_path, data_transform, gt_transform):
        """准备测试数据"""
        test_data_list = []
        for item in item_list:
            test_path = os.path.join(data_path, item)
            test_data = MVTecDataset(root=test_path, transform=data_transform,
                                     gt_transform=gt_transform, phase="test")
            test_data_list.append(test_data)
        return test_data_list

    def prepare_data(self, item_list, data_path, data_transform, gt_transform):
        """准备训练和测试数据"""
        train_data_list, test_data_list = [], []

        for i, item in enumerate(item_list):
            train_path = os.path.join(data_path, item, 'train')
            test_path = os.path.join(data_path, item)

            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            train_data.samples = [(sample[0], i) for sample in train_data.samples]
            train_data_list.append(train_data)

            test_data = MVTecDataset(root=test_path, transform=data_transform,
                                     gt_transform=gt_transform, phase="test")
            test_data_list.append(test_data)

        return train_data_list, test_data_list


class DinomalyV2Trainer(DinomalyBaseTrainer):
    def __init__(self, model_size="small", device='cuda:0'):
        super().__init__(model_size, device)
        self.encoder_name = None
        self.setup_encoder_config()

    def setup_encoder_config(self):
        """设置编码器名称"""
        if self.model_size == "small":
            self.encoder_name = "dinov2reg_vit_small_14"
        elif self.model_size == "base":
            self.encoder_name = "dinov2reg_vit_base_14"
        else:
            self.encoder_name = "dinov2reg_vit_large_14"

    def create_model(self):
        """创建DinoV2模型"""
        self.encoder = vit_encoder.load(self.encoder_name)
        bottleneck, decoder = self.create_model_components()

        self.model = ViTillDinoV2(
            encoder=self.encoder, bottleneck=bottleneck, decoder=decoder,
            target_layers=self.target_layers, mask_neighbor_size=0,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder
        ).to(self.device)

        self.trainable = nn.ModuleList([bottleneck, decoder])
        self.initialize_weights()

    def train(self, item_list, data_path, save_dir, total_iters=10000, batch_size=16,
              image_size=448, crop_size=392):
        """训练模型"""
        self.logger.info("Begin DinomalyDinov2 model train!!!")
        setup_seed(1)
        data_transform, gt_transform = get_data_transforms(image_size, crop_size)

        # 准备数据
        train_data_list, test_data_list = self.prepare_data(item_list, data_path, data_transform, gt_transform)
        train_dataloader = torch.utils.data.DataLoader(
            ConcatDataset(train_data_list), batch_size=batch_size, shuffle=True,
            num_workers=4, drop_last=True
        )

        # 创建模型和优化器
        self.create_model()
        optimizer = self.setup_optimizer()
        scheduler = self.setup_scheduler(optimizer, total_iters)

        # 训练循环
        it = 0
        for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
            self.model.train()
            loss_list = []

            for img, label in train_dataloader:
                img, label = img.to(self.device), label.to(self.device)
                en, de = self.model(img)

                p = min(0.9 * it / 1000, 0.9)
                loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.trainable.parameters(), max_norm=0.1)
                optimizer.step()

                loss_list.append(loss.item())
                scheduler.step()

                if (it + 1) % 100 == 0:
                    self.logger.info("Begin model eval!!!")
                    self.evaluate_model(item_list, test_data_list, batch_size)
                    for item, test_data in zip(item_list, test_data_list):
                        test_dataloader = torch.utils.data.DataLoader(
                            test_data, batch_size=batch_size, shuffle=False, num_workers=4
                        )
                        visualize(
                            self.model, test_dataloader, self.device,
                            _class_=item,
                            save_name=f"dinov2_model_size_{self.model_size}_epoch_{it + 1}"
                        )
                    self.model.train()

                it += 1
                if it == total_iters:
                    break

            self.logger.info(f'iter [{it}/{total_iters}], loss:{np.mean(loss_list):.4f}')
        self.save_model(save_dir=save_dir, save_name="dinomaly_dinov2", item_list=item_list, total_iters=total_iters)

    def evaluate(self, model_path, item_list, data_path, batch_size=16,
                 image_size=448, crop_size=392):
        """评估模型"""
        data_transform, gt_transform = get_data_transforms(image_size, crop_size)
        test_data_list = self.prepare_test_data(item_list, data_path, data_transform, gt_transform)

        self.create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        self.logger.info("Model load finished!")
        return self.evaluate_model(item_list, test_data_list, batch_size)


class DinomalyV3Trainer(DinomalyBaseTrainer):
    def __init__(self, model_size="small", device='cuda:0'):
        super().__init__(model_size, device)
        self.encoder_name = None
        self.encoder_weight = None
        self.setup_encoder_config()

    def setup_encoder_config(self):
        """设置编码器配置"""
        if self.model_size == "small":
            self.encoder_name = 'dinov3_vits16'
            self.encoder_weight = 'weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
        elif self.model_size == "base":
            self.encoder_name = 'dinov3_vitb16'
            self.encoder_weight = 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        else:
            self.encoder_name = 'dinov3_vitl16'
            self.encoder_weight = 'weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'

    def create_model(self):
        """创建DinoV3模型"""
        self.encoder = load_dinov3_model(
            self.encoder_name,
            layers_to_extract_from=self.target_layers,
            pretrained_weight_path=self.encoder_weight
        )

        bottleneck, decoder = self.create_model_components()

        self.model = ViTill(
            encoder=self.encoder, bottleneck=bottleneck, decoder=decoder,
            target_layers=self.target_layers, mask_neighbor_size=0,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder
        ).to(self.device)

        self.trainable = nn.ModuleList([bottleneck, decoder])
        self.initialize_weights()

    def train(self, item_list, data_path, save_dir, total_iters=10000, batch_size=16,
              image_size=512, crop_size=448):
        """训练模型"""

        self.logger.info("Begin DinomalyDinov3 model train!!!")
        setup_seed(1)
        data_transform, gt_transform = get_data_transforms(image_size, crop_size)

        # 准备数据
        train_data_list, test_data_list = self.prepare_data(item_list, data_path, data_transform, gt_transform)
        train_dataloader = torch.utils.data.DataLoader(
            ConcatDataset(train_data_list), batch_size=batch_size, shuffle=True,
            num_workers=4, drop_last=True
        )

        # 创建模型和优化器
        self.create_model()
        optimizer = self.setup_optimizer()
        scheduler = self.setup_scheduler(optimizer, total_iters)

        # 训练循环
        it = 0
        for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
            self.model.train()
            self.model.encoder.eval()
            loss_list = []

            for img, label in train_dataloader:
                img, label = img.to(self.device), label.to(self.device)
                en, de = self.model(img)

                p = min(0.9 * it / 1000, 0.9)
                loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.trainable.parameters(), max_norm=0.1)
                optimizer.step()

                loss_list.append(loss.item())
                scheduler.step()

                if (it + 1) % 100 == 0:
                    self.logger.info("Begin model eval!!!")
                    self.evaluate_model(item_list, test_data_list, batch_size)

                    for item, test_data in zip(item_list, test_data_list):
                        test_dataloader = torch.utils.data.DataLoader(
                            test_data, batch_size=batch_size, shuffle=False, num_workers=4
                        )
                        visualize(
                            self.model, test_dataloader, self.device,
                            _class_=item,
                            save_name=f"dinov3_model_size_{self.model_size}_epoch_{it + 1}"
                        )

                    self.model.train()
                    self.model.encoder.eval()

                it += 1
                if it == total_iters:
                    break

            self.logger.info(f'iter [{it}/{total_iters}], loss:{np.mean(loss_list):.4f}')
        self.save_model(save_dir=save_dir, save_name="dinomaly_dinov3", item_list=item_list, total_iters=total_iters)

    def evaluate(self, model_path, item_list, data_path, batch_size=16,
                 image_size=512, crop_size=448):
        """评估模型"""
        data_transform, gt_transform = get_data_transforms(image_size, crop_size)
        test_data_list = self.prepare_test_data(item_list, data_path, data_transform, gt_transform)

        self.create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        self.logger.info("Model load finished!")
        return self.evaluate_model(item_list, test_data_list, batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../data/mvtec')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='vitill_mvtec_uni')
    args = parser.parse_args()

    item_list = ['carpet', 'grid']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))

    # 创建并训练DinoV2模型
    dinomaly_v2 = DinomalyV2Trainer(model_size="small", device=device)
    dinomaly_v2.logger = logger

    # 训练
    dinomaly_v2.train(item_list, args.data_path, save_dir=args.save_dir, total_iters=100)

    # 评估
    model_path = "saved_results/dinomaly_dinov2_small_carpet_grid_epoch_100_Sat Dec 20 18:14:09 2025.pth"
    dinomaly_v2.evaluate(model_path, item_list, args.data_path)

    # 创建并训练DinoV3模型
    dinomaly_v3 = DinomalyV3Trainer(model_size="small", device=device)
    dinomaly_v3.logger = logger

    # 训练
    dinomaly_v3.train(item_list, args.data_path, save_dir=args.save_dir, total_iters=100)

    # 评估
    model_path = "saved_results/dinomaly_dinov3_small_carpet_grid_epoch_100_Sat Dec 20 18:14:43 2025.pth"
    dinomaly_v3.evaluate(model_path, item_list, args.data_path)
