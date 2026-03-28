"""
    PatchMaker, Preprocessing and MeanMapper are copied from https://github.com/amazon-science/patchcore-inspection.
"""

"""
translated by codegeex:

这段代码实现了一个完整的特征处理流程，包括补丁生成、特征对齐和多层聚合，适用于异常检测或特征提取任务。主要注意事项包括：

输入特征的形状和层数需与模型配置一致。
补丁生成和插值操作可能会影响特征的局部信息。
最终特征被移动到 CPU，适用于推理阶段，训练时需修改。
如果需要进一步优化，可以考虑：

使用更高效的插值方法（如最近邻插值）。
调整 PatchMaker 的 stride 参数以控制补丁重叠程度。
支持动态输入特征维度（如不同分辨率的输入）。

"""

import math

import torch
import torch.nn.functional as F


class PatchMaker:
    def __init__(self, patchsize, stride=None):

        """
        初始化PatchMaker类

        参数:
            patchsize: int, 生成的patch大小
            stride: int, stride大小，默认为None

        PatchMaker 类用于将输入的特征图分割成多个小的补丁（patches）。
        这在处理图像特征时非常有用，尤其是在 PatchCore 等异常检测方法中。
        实现原理
            初始化 (__init__):
                patchsize: 定义每个补丁的大小（例如 3x3）。
                stride: 定义补丁之间的步长，默认为 None（通常与 patchsize 相同）。
            补丁生成 (patchify):
                使用 torch.nn.Unfold 将输入特征图展开为补丁。
                计算填充（padding）以确保边缘补丁也能正确生成。
                返回补丁张量，形状为 (batch_size, num_patches, channels, patch_height, patch_width)。
        注意事项
            如果 stride 未设置，默认为 patchsize，可能导致补丁之间没有重叠。
            return_spatial_info 参数可以返回补丁的空间维度信息，便于后续处理。
        """
        self.patchsize = patchsize  # 设置patch大小
        self.stride = stride  # 设置stride大小

    def patchify(self, features, return_spatial_info=False):

        """
        将输入特征图转换为patch形式

        参数:
            features: 输入的特征图
            return_spatial_info: bool, 是否返回空间信息，默认为False

        返回:
            如果return_spatial_info为True，返回patch和空间信息
            否则只返回patch
        """
        padding = int((self.patchsize - 1) / 2)  # 计算填充大小
        # 创建Unfold层用于提取patch
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)  # 应用Unfold提取patch
        number_of_total_patches = []  # 用于存储每个维度上的patch数量
        # 计算每个维度上的patch数量
        for s in features.shape[-2:]:
            n_patches = (
                                s + 2 * padding - 1 * (self.patchsize - 1) - 1
                        ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        # 重塑张量形状并调整维度顺序
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        # 根据参数决定是否返回空间信息
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


class Preprocessing(torch.nn.Module):
    def __init__(self, input_layers, output_dim):
        """
        初始化预处理模块
        Args:
            input_layers: 输入层的列表
            output_dim: 输出维度

        Preprocessing 类用于对多个特征层进行预处理，确保它们的维度一致。

    实现原理
        初始化 (__init__):
            input_layers: 输入特征层的列表。
            output_dim: 预处理后的输出维度。
            为每个输入层创建一个 MeanMapper 模块。
        前向传播 (forward):
            对每个输入特征应用对应的 MeanMapper 模块。
            将处理后的特征在维度 1 上堆叠（torch.stack）。
    注意事项
        输入特征的数量必须与 input_layers 的长度一致。
        输出特征的形状为 (batch_size, num_layers, output_dim)。

        """
        super(Preprocessing, self).__init__()
        self.output_dim = output_dim  # 设置输出维度
        # 创建一个ModuleList来存储预处理模块
        self.preprocessing_modules = torch.nn.ModuleList()
        # 为每个输入层创建一个MeanMapper模块
        for input_layer in input_layers:
            module = MeanMapper(output_dim)  # 创建一个输出维度为output_dim的MeanMapper
            self.preprocessing_modules.append(module)  # 将模块添加到ModuleList中

    def forward(self, features):

        """
        前向传播函数
        Args:
            features: 输入特征列表
        Returns:
            处理后的特征堆叠结果
        """
        _features = []  # 用于存储处理后的特征
        # 遍历每个预处理模块和对应的输入特征
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))  # 对每个特征应用预处理模块
        # 将处理后的特征在维度1上堆叠
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        """
        初始化MeanMapper类

        参数:
            preprocessing_dim (int): 预处理维度，用于自适应平均池化操作的目标维度

        MeanMapper 类通过自适应平均池化将输入特征的维度调整为指定的 preprocessing_dim。

        实现原理
            初始化 (__init__):
                preprocessing_dim: 目标维度。
            前向传播 (forward):
                将输入特征重塑为 (batch_size, 1, feature_length)。
                使用 F.adaptive_avg_pool1d 将特征长度缩减到 preprocessing_dim。
                移除维度为 1 的维度。
        注意事项
            适用于特征长度不一致的情况，通过池化操作统一维度。
            池化操作会丢失部分信息，但能显著减少计算量。

        """
        super(MeanMapper, self).__init__()  # 调用父类(torch.nn.Module)的初始化方法
        self.preprocessing_dim = preprocessing_dim  # 保存预处理维度

    def forward(self, features):
        """
        前向传播函数
        参数:
            features (torch.Tensor): 输入特征张量
        返回:
            torch.Tensor: 经过自适应平均池化处理后的特征张量
        """
        # 将输入特征重塑为三维张量，形状为(batch_size, 1, feature_length)
        features = features.reshape(len(features), 1, -1)

        # 使用自适应平均池化将特征长度维度缩减到preprocessing_dim
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)  # 移除维度为1的维度


class LNAMD(torch.nn.Module):
    def __init__(self, device, feature_dim=1024, feature_layer=[1, 2, 3, 4], r=3, patchstride=1):
        """
        初始化LNAMD模块
        参数:
            device: 计算设备(CPU/GPU)
            feature_dim: 特征维度，默认为1024
            feature_layer: 特征层列表，默认为[1,2,3,4]
            r: 补丁大小，默认为3
            patchstride: 补丁步长，默认为1


        LNAMD 类是主模块，用于从多层特征中提取补丁并聚合为最终的特征表示。

        实现原理
            初始化 (__init__):

                device: 计算设备（CPU/GPU）。
                feature_dim: 特征维度，默认为 1024。
                feature_layer: 特征层列表，默认为 [1, 2, 3, 4]。
                r: 补丁大小，默认为 3。
                patchstride: 补丁步长，默认为 1。
            特征嵌入 (_embed):
                特征重塑和归一化:
                    移除 CLS token（ViT 中的分类 token）。
                    将特征重塑为 (batch_size, channels, height, width)。
                    应用 LayerNorm 归一化。
                补丁生成:
                    如果 r != 1，使用 PatchMaker 将特征分割为补丁。
                    否则，直接将特征重塑为 (batch_size, num_patches, channels, 1, 1)。
                补丁对齐:
                    如果不同层的补丁数量不一致，使用双线性插值对齐到参考层的补丁数量。
                特征聚合:
                    使用 Preprocessing 模块聚合多层特征。
                    返回最终的特征张量，形状为 (batch_size, num_patches, num_layers, feature_dim)。
        注意事项
            输入特征应为 ViT 的 patch tokens，形状为 (batch_size, num_tokens, feature_dim)。
            如果不同层的补丁数量不一致，插值操作可能会引入一定的误差。
            最终特征被移动到 CPU 并分离梯度（detach().cpu()），适用于推理阶段。
        """
        super(LNAMD, self).__init__()
        self.device = device
        self.r = r
        self.patch_maker = PatchMaker(r, stride=patchstride)  # 创建补丁生成器
        self.LNA = Preprocessing(feature_layer, feature_dim)  # 创建预处理器

    def _embed(self, features):

        """
        对输入特征进行嵌入处理

        参数:
            features: 输入特征列表

        返回:
            处理后的特征张量
        """
        B = features[0].shape[0]  # 获取批次大小

        features_layers = []
        for feature in features:
            # reshape and layer normalization
            feature = feature[:, 1:, :]  # remove the cls token
            feature = feature.reshape(feature.shape[0],
                                      int(math.sqrt(feature.shape[1])),
                                      int(math.sqrt(feature.shape[1])),
                                      feature.shape[2])
            feature = feature.permute(0, 3, 1, 2)
            feature = torch.nn.LayerNorm([feature.shape[1], feature.shape[2],
                                          feature.shape[3]]).to(self.device)(feature)
            features_layers.append(feature)

        if self.r != 1:
            # divide into patches
            features_layers = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features_layers]
            patch_shapes = [x[1] for x in features_layers]
            features_layers = [x[0] for x in features_layers]
        else:
            patch_shapes = [f.shape[-2:] for f in features_layers]
            features_layers = [f.reshape(f.shape[0], f.shape[1], -1, 1, 1).permute(0, 2, 1, 3, 4) for f in
                               features_layers]

        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features_layers)):
            patch_dims = patch_shapes[i]
            if patch_dims[0] == ref_num_patches[0] and patch_dims[1] == ref_num_patches[1]:
                continue
            _features = features_layers[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features_layers[i] = _features
        features_layers = [x.reshape(-1, *x.shape[-3:]) for x in features_layers]

        # aggregation
        features_layers = self.LNA(features_layers)
        features_layers = features_layers.reshape(B, -1, *features_layers.shape[-2:])  # (B, L, layer, C)

        return features_layers.detach().cpu()


if __name__ == "__main__":
    import time

    device = 'cuda:0'
    LNAMD_r = LNAMD(device=device, r=3, feature_dim=1024, feature_layer=[1, 2, 3, 4])
    B = 32
    patch_tokens = [torch.rand((B, 1370, 1024)), torch.rand((B, 1370, 1024)), torch.rand((B, 1370, 1024)),
                    torch.rand((B, 1370, 1024))]
    patch_tokens = [f.to('cuda:0') for f in patch_tokens]
    s = time.time()
    features = LNAMD_r._embed(patch_tokens)
    e = time.time()
    print((e - s) * 1000 / 32)
