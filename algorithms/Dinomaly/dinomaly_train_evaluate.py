"""
Dinomaly 模型训练与评估脚本
支持 DINOv2 和 DINOv3 编码器的异常检测模型
"""

# =============================================================================
# 标准库导入
# =============================================================================
import argparse
import logging
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

# =============================================================================
# 第三方库导入
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder

# =============================================================================
# 本地模块导入 (使用相对导入)
# =============================================================================
from .dataset import MVTecDataset, get_data_transforms
from .dinov1.utils import trunc_normal_
from .dinov3.hub.backbones import load_dinov3_model
from .models.uad import ViTill, ViTillDinoV2
from .models import vit_encoder
from .models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from .optimizers import StableAdamW
from .utils import (
    evaluation_batch,
    global_cosine_hm_percent,
    WarmCosineScheduler,
    visualize,
)

warnings.filterwarnings("ignore")


# =============================================================================
# 配置类
# =============================================================================
@dataclass
class ModelConfig:
    """模型配置参数"""
    model_size: str = "small"  # small, base, large
    device: str = "cuda:0"
    image_size: int = 512
    crop_size: int = 448
    batch_size: int = 8
    total_iters: int = 20000
    warmup_iters: int = 100
    lr: float = 2e-3
    final_lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    eval_interval: int = 1000


@dataclass
class DataConfig:
    """数据配置参数"""
    data_path: str = "/home/zhouchenghao/PycharmProjects/ASD_for_SPK/data/spk"
    save_dir: str = "./saved_results"
    save_name: str = "vit_spk"
    item_list: List[str] = field(default_factory=lambda: ["dk", "qzgy"])
    num_workers: int = 4


# =============================================================================
# 工具函数
# =============================================================================
def get_logger(name: str, save_path: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """创建并配置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_path, "log.txt"))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def setup_seed(seed: int = 1) -> None:
    """设置随机种子保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_first_greater_than(sequence: np.ndarray, thresholds: List[float]) -> Dict[float, int]:
    """
    查找序列中第一个大于各阈值的索引
    
    Args:
        sequence: 输入序列
        thresholds: 阈值列表
        
    Returns:
        阈值到索引的映射字典
    """
    results = {}
    for threshold in thresholds:
        index = -1
        for i, num in enumerate(sequence):
            if num > threshold:
                index = i
                break
        results[threshold] = index
    return results


# =============================================================================
# 模型架构配置
# =============================================================================
MODEL_ARCHITECTURES = {
    "small": {
        "embed_dim": 384,
        "num_heads": 6,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
    },
    "base": {
        "embed_dim": 768,
        "num_heads": 12,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
    },
    "large": {
        "embed_dim": 1024,
        "num_heads": 16,
        "target_layers": [4, 6, 8, 10, 12, 14, 16, 18],
    },
}


ENCODER_CONFIGS = {
    "dinov2": {
        "small": "dinov2reg_vit_small_14",
        "base": "dinov2reg_vit_base_14",
        "large": "dinov2reg_vit_large_14",
    },
    "dinov3": {
        "small": {
            "name": "dinov3_vits16",
            "weight": "pre_trained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        },
        "base": {
            "name": "dinov3_vitb16",
            "weight": "pre_trained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        },
        "large": {
            "name": "dinov3_vitl16",
            "weight": "pre_trained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        },
    },
}


# =============================================================================
# 训练器基类
# =============================================================================
class BaseTrainer:
    """异常检测模型训练器基类"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger: Optional[logging.Logger] = None
        self.model: Optional[nn.Module] = None
        self.trainable: Optional[nn.Module] = None
        
        # 获取架构配置
        arch_config = MODEL_ARCHITECTURES[config.model_size]
        self.embed_dim = arch_config["embed_dim"]
        self.num_heads = arch_config["num_heads"]
        self.target_layers = arch_config["target_layers"]
        
        # 融合层配置
        self.fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    def create_model_components(self) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """创建模型组件（瓶颈层和解码器）"""
        bottleneck = nn.ModuleList([
            bMlp(self.embed_dim, self.embed_dim * 4, self.embed_dim, drop=0.2)
        ])

        decoder = nn.ModuleList([
            VitBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2,
            )
            for _ in range(8)
        ])

        return bottleneck, decoder

    def initialize_weights(self) -> None:
        """初始化可训练参数权重"""
        for m in self.trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def setup_optimizer(self) -> StableAdamW:
        """配置优化器"""
        return StableAdamW(
            [{"params": self.trainable.parameters()}],
            lr=self.config.lr,
            betas=(0.9, 0.999),
            weight_decay=self.config.weight_decay,
            amsgrad=True,
            eps=1e-10,
        )

    def setup_scheduler(self, optimizer: StableAdamW) -> WarmCosineScheduler:
        """配置学习率调度器"""
        return WarmCosineScheduler(
            optimizer,
            base_value=self.config.lr,
            final_value=self.config.final_lr,
            total_iters=self.config.total_iters,
            warmup_iters=self.config.warmup_iters,
        )

    def save_model(self, save_dir: str, save_name: str, item_list: List[str]) -> None:
        """保存模型权重"""
        item_list_str = "_".join(item_list) + "_"
        model_save_path = os.path.join(
            save_dir,
            f"{save_name}_{self.config.model_size}_{item_list_str}"
            f"epoch_{self.config.total_iters}_{time.ctime()}.pth"
        )
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), model_save_path)
        self.logger.info(f"Model saved to {model_save_path}!")


# =============================================================================
# 评估器混入类
# =============================================================================
class EvaluationMixin:
    """评估功能混入类"""

    def evaluate_model(
        self,
        item_list: List[str],
        test_data_list: List[MVTecDataset],
        batch_size: int = 16,
    ) -> Tuple[List[float], List[float], List[float], List, List]:
        """评估模型性能"""
        auroc_list, ap_list, f1_list = [], [], []
        gt_list, pr_list = [], []

        for item, test_data in zip(item_list, test_data_list):
            test_loader = DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
            results = evaluation_batch(
                self.model, test_loader, self.device, max_ratio=0.005, resize_mask=256
            )
            auroc, ap, f1, gt, pr = results

            auroc_list.append(auroc)
            ap_list.append(ap)
            f1_list.append(f1)
            gt_list.extend(gt)
            pr_list.extend(pr)

            self.logger.info(
                f"{item}: I-Auroc:{auroc:.4f}, I-AP:{ap:.4f}, I-F1:{f1:.4f}"
            )

        self.logger.info(
            f"Mean: I-Auroc:{np.mean(auroc_list):.4f}, "
            f"I-AP:{np.mean(ap_list):.4f}, I-F1:{np.mean(f1_list):.4f}"
        )
        
        self._plot_roc_curve(gt_list, pr_list)
        return auroc_list, ap_list, f1_list, gt_list, pr_list

    def _plot_roc_curve(self, gt_list: List, pr_list: List) -> None:
        """绘制并保存ROC曲线"""
        fpr, tpr, thresholds = metrics.roc_curve(gt_list, pr_list)
        auroc = metrics.auc(fpr, tpr)
        
        print(f"fpr: {fpr}")
        print(f"tpr: {tpr}")
        print(f"thresholds: {thresholds}")
        print(f"AUROC: {auroc}")

        # 输出关键阈值点
        targets = [0.005, 0.01, 0.05]
        indices = find_first_greater_than(fpr, targets)
        for target in targets:
            idx = indices[target]
            if idx >= 0:
                print(
                    f"When threshold={thresholds[idx]:.6f}, "
                    f"fpr={fpr[idx]:.6f}, tpr={tpr[idx]:.6f}"
                )

        # 绘制ROC曲线
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim((-0.01, 0.05))
        plt.ylim((0.7, 1))
        plt.grid(True)
        plt.title("ROC Curve")
        plt.savefig("roc_curve.png")
        plt.close()


# =============================================================================
# 数据准备混入类
# =============================================================================
class DataPreparationMixin:
    """数据准备功能混入类"""

    def prepare_data(
        self,
        item_list: List[str],
        data_path: str,
        data_transform,
        gt_transform,
    ) -> Tuple[List, List]:
        """准备训练和测试数据"""
        train_data_list, test_data_list = [], []

        for i, item in enumerate(item_list):
            train_path = os.path.join(data_path, item, "train")
            test_path = os.path.join(data_path, item)

            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            train_data.samples = [(sample[0], i) for sample in train_data.samples]
            train_data_list.append(train_data)

            test_data = MVTecDataset(
                root=test_path,
                transform=data_transform,
                gt_transform=gt_transform,
                phase="test",
            )
            test_data_list.append(test_data)

        return train_data_list, test_data_list

    def prepare_test_data(
        self,
        item_list: List[str],
        data_path: str,
        data_transform,
        gt_transform,
    ) -> List:
        """仅准备测试数据（用于评估模式）"""
        test_data_list = []
        for item in item_list:
            test_path = os.path.join(data_path, item)
            test_data = MVTecDataset(
                root=test_path,
                transform=data_transform,
                gt_transform=gt_transform,
                phase="test",
            )
            test_data_list.append(test_data)
        return test_data_list


# =============================================================================
# DINOv2 训练器
# =============================================================================
class DinomalyV2Trainer(BaseTrainer, EvaluationMixin, DataPreparationMixin):
    """基于 DINOv2 编码器的 Dinomaly 训练器"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.encoder_name = ENCODER_CONFIGS["dinov2"][config.model_size]

    def create_model(self) -> None:
        """创建 DINOv2 模型"""
        encoder = vit_encoder.load(self.encoder_name)
        bottleneck, decoder = self.create_model_components()

        self.model = ViTillDinoV2(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            target_layers=self.target_layers,
            mask_neighbor_size=0,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder,
        ).to(self.device)

        self.trainable = nn.ModuleList([bottleneck, decoder])
        self.initialize_weights()

    def train(self, data_config: DataConfig) -> None:
        """训练模型"""
        self.logger.info("Begin DinomalyDinov2 model train!!!")
        setup_seed(1)
        
        data_transform, gt_transform = get_data_transforms(
            self.config.image_size, self.config.crop_size
        )

        # 准备数据
        train_data_list, test_data_list = self.prepare_data(
            data_config.item_list,
            data_config.data_path,
            data_transform,
            gt_transform,
        )
        train_loader = DataLoader(
            ConcatDataset(train_data_list),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
            drop_last=True,
        )

        # 创建模型和优化器
        self.create_model()
        optimizer = self.setup_optimizer()
        scheduler = self.setup_scheduler(optimizer)

        # 训练循环
        iteration = 0
        num_epochs = int(np.ceil(self.config.total_iters / len(train_loader)))
        
        for epoch in range(num_epochs):
            self.model.train()
            loss_list = []

            for images, labels in train_loader:
                images = images.to(self.device)

                en, de = self.model(images)
                p = min(0.9 * iteration / 1000, 0.9)
                loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.trainable.parameters(), max_norm=self.config.grad_clip)
                optimizer.step()

                loss_list.append(loss.item())
                scheduler.step()

                # 评估和可视化
                if (iteration + 1) % self.config.eval_interval == 0:
                    self._evaluate_and_visualize(
                        data_config.item_list,
                        test_data_list,
                        iteration + 1,
                    )
                    self.model.train()

                iteration += 1
                if iteration >= self.config.total_iters:
                    break

            self.logger.info(
                f"iter [{iteration}/{self.config.total_iters}], loss:{np.mean(loss_list):.4f}"
            )

        self.save_model(
            data_config.save_dir,
            "dinomaly_dinov2",
            data_config.item_list,
        )

    def _evaluate_and_visualize(
        self,
        item_list: List[str],
        test_data_list: List,
        iteration: int,
    ) -> None:
        """执行评估和可视化"""
        self.logger.info("Begin model eval!!!")
        self.evaluate_model(item_list, test_data_list, self.config.batch_size)

        for item, test_data in zip(item_list, test_data_list):
            test_loader = DataLoader(
                test_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
            )
            visualize(
                self.model,
                test_loader,
                self.device,
                _class_=item,
                save_name=f"dinomaly_dinov2_{self.config.model_size}_epoch_{iteration}",
            )
        
        self.logger.info("Visualization done!")

    def evaluate(self, model_path: str, data_config: DataConfig) -> Tuple:
        """评估预训练模型"""
        data_transform, gt_transform = get_data_transforms(
            self.config.image_size, self.config.crop_size
        )
        test_data_list = self.prepare_test_data(
            data_config.item_list,
            data_config.data_path,
            data_transform,
            gt_transform,
        )

        self.create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        self.logger.info("Model load finished!")
        return self.evaluate_model(data_config.item_list, test_data_list, self.config.batch_size)


# =============================================================================
# DINOv3 训练器
# =============================================================================
class DinomalyV3Trainer(BaseTrainer, EvaluationMixin, DataPreparationMixin):
    """基于 DINOv3 编码器的 Dinomaly 训练器"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        v3_config = ENCODER_CONFIGS["dinov3"][config.model_size]
        self.encoder_name = v3_config["name"]
        self.encoder_weight = v3_config["weight"]

    def create_model(self) -> None:
        """创建 DINOv3 模型"""
        encoder = load_dinov3_model(
            self.encoder_name,
            layers_to_extract_from=self.target_layers,
            pretrained_weight_path=self.encoder_weight,
        )
        bottleneck, decoder = self.create_model_components()

        self.model = ViTill(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            target_layers=self.target_layers,
            mask_neighbor_size=0,
            fuse_layer_encoder=self.fuse_layer_encoder,
            fuse_layer_decoder=self.fuse_layer_decoder,
        ).to(self.device)

        self.trainable = nn.ModuleList([bottleneck, decoder])
        self.initialize_weights()

    def train(self, data_config: DataConfig) -> None:
        """训练模型"""
        self.logger.info("Begin DinomalyDinov3 model train!!!")
        setup_seed(1)
        
        data_transform, gt_transform = get_data_transforms(
            self.config.image_size, self.config.crop_size
        )

        # 准备数据
        train_data_list, test_data_list = self.prepare_data(
            data_config.item_list,
            data_config.data_path,
            data_transform,
            gt_transform,
        )
        train_loader = DataLoader(
            ConcatDataset(train_data_list),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
            drop_last=True,
        )

        # 创建模型和优化器
        self.create_model()
        optimizer = self.setup_optimizer()
        scheduler = self.setup_scheduler(optimizer)

        # 训练循环
        iteration = 0
        num_epochs = int(np.ceil(self.config.total_iters / len(train_loader)))
        
        for epoch in range(num_epochs):
            self.model.train()
            self.model.encoder.eval()  # 冻结编码器
            loss_list = []

            for images, labels in train_loader:
                images = images.to(self.device)

                en, de = self.model(images)
                p = min(0.9 * iteration / 1000, 0.9)
                loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.trainable.parameters(), max_norm=self.config.grad_clip)
                optimizer.step()

                loss_list.append(loss.item())
                scheduler.step()

                # 评估和可视化
                if (iteration + 1) % self.config.eval_interval == 0:
                    self._evaluate_and_visualize(
                        data_config.item_list,
                        test_data_list,
                        iteration + 1,
                    )
                    self.model.train()
                    self.model.encoder.eval()

                iteration += 1
                if iteration >= self.config.total_iters:
                    break

            self.logger.info(
                f"iter [{iteration}/{self.config.total_iters}], loss:{np.mean(loss_list):.4f}"
            )

        self.save_model(
            data_config.save_dir,
            "dinomaly_dinov3",
            data_config.item_list,
        )

    def _evaluate_and_visualize(
        self,
        item_list: List[str],
        test_data_list: List,
        iteration: int,
    ) -> None:
        """执行评估和可视化"""
        self.logger.info("Begin model eval!!!")
        self.evaluate_model(item_list, test_data_list, self.config.batch_size)

        for item, test_data in zip(item_list, test_data_list):
            test_loader = DataLoader(
                test_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
            )
            visualize(
                self.model,
                test_loader,
                self.device,
                _class_=item,
                save_name=f"dinomaly_dinov3_{self.config.model_size}_epoch_{iteration}",
            )
        
        self.logger.info("Visualization done!")

    def evaluate(self, model_path: str, data_config: DataConfig) -> Tuple:
        """评估预训练模型"""
        data_transform, gt_transform = get_data_transforms(
            self.config.image_size, self.config.crop_size
        )
        test_data_list = self.prepare_test_data(
            data_config.item_list,
            data_config.data_path,
            data_transform,
            gt_transform,
        )

        self.create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        self.logger.info("Model load finished!")
        return self.evaluate_model(data_config.item_list, test_data_list, self.config.batch_size)


# =============================================================================
# 主函数
# =============================================================================
def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Dinomaly 模型训练与评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/zhouchenghao/PycharmProjects/ASD_for_SPK/data/spk",
        help="数据集路径",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_results",
        help="模型保存目录",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="vit_spk",
        help="保存名称",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "base", "large"],
        help="模型大小",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="dinov3",
        choices=["dinov2", "dinov3"],
        help="编码器类型",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批次大小",
    )
    parser.add_argument(
        "--total_iters",
        type=int,
        default=1000,
        help="总训练迭代次数",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="仅执行评估",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="评估时加载的模型路径",
    )
    return parser.parse_args()


def main():
    """主入口函数"""
    args = parse_arguments()

    # 创建配置
    model_config = ModelConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        total_iters=args.total_iters,
    )
    data_config = DataConfig(
        data_path=args.data_path,
        save_dir=args.save_dir,
        save_name=args.save_name,
    )

    # 创建日志记录器
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger = get_logger(args.save_name, args.save_dir)
    logger.info(f"Using device: {device}")

    # 创建训练器
    if args.model_type == "dinov2":
        trainer = DinomalyV2Trainer(model_config)
    else:
        trainer = DinomalyV3Trainer(model_config)
    
    trainer.logger = logger

    # 训练或评估
    if args.eval_only:
        if args.model_path is None:
            raise ValueError("评估模式必须指定 --model_path")
        trainer.evaluate(args.model_path, data_config)
    else:
        trainer.train(data_config)


if __name__ == "__main__":
    main()
