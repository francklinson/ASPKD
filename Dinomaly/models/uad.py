import math

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


class ViTill(nn.Module):
    def __init__(
            self,
            encoder,          # 编码器网络，用于输入特征提取
            bottleneck,       # 瓶颈层，用于特征转换和降维
            decoder,          # 解码器网络，用于特征重建
            target_layers=None,  # 目标层列表，指定编码器中需要提取的特征层
            fuse_layer_encoder=None,  # 编码器特征融合层列表
            fuse_layer_decoder=None,  # 解码器特征融合层列表
            mask_neighbor_size=0,     # 掩码邻域大小，用于生成注意力掩码
            remove_class_token=False, # 是否移除类别标记
            encoder_require_grad_layer=None,  # 需要计算梯度的编码器层列表
    ) -> None:
        super(ViTill, self).__init__()
        """
        ViTill模型的初始化函数
        参数:
            encoder: 编码器网络
            bottleneck: 瓶颈层
            decoder: 解码器网络
            target_layers: 目标层列表
            fuse_layer_encoder: 编码器特征融合层列表
            fuse_layer_decoder: 解码器特征融合层列表
            mask_neighbor_size: 掩码邻域大小
            remove_class_token: 是否移除类别标记
            encoder_require_grad_layer: 需要计算梯度的编码器层列表
        """
        # 初始化默认参数
        if encoder_require_grad_layer is None:
            encoder_require_grad_layer = []
        if fuse_layer_decoder is None:
            fuse_layer_decoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
        if fuse_layer_encoder is None:
            fuse_layer_encoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
        if target_layers is None:
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        # 初始化网络组件
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        # 检查并设置注册令牌数量
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):

        """
        模型前向传播函数
        参数:
            x: 输入张量
        返回:
            en: 编码器特征列表
            de: 解码器特征列表
        """
        # 从编码器获取中间层特征
        en_list = self.encoder.get_intermediate_layers(x, n=self.target_layers, norm=False)

        # 计算特征图的空间尺寸
        side = int(math.sqrt(en_list[0].shape[1]))

        # 融合编码器特征
        x = self.fuse_feature(en_list)

        # 通过瓶颈层处理特征
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        # 生成注意力掩码（如果需要）
        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        # 通过解码器处理特征
        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]  # 反转解码器特征列表

        # 融合编码器和解码器特征
        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        # 重塑特征维度
        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        return en, de

    def fuse_feature(self, feat_list):
        """融合特征列表，通过堆叠和平均操作"""
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all


class ViTillCat(nn.Module):
    def __init__(
            self,
            encoder,          # 编码器，用于提取输入特征
            bottleneck,       # 瓶颈层，用于特征压缩
            decoder,          # 解码器，用于特征重建
            target_layers=None,   # 目标层列表，用于指定哪些编码器层的结果需要保留
            fuse_layer_encoder=None,  # 编码器中用于融合特征的层
            mask_neighbor_size=0,     # 掩码邻域大小
            remove_class_token=False, # 是否移除类别令牌
            encoder_require_grad_layer=[],  # 需要计算梯度的编码器层
    ) -> None:
        super(ViTillCat, self).__init__()
        if fuse_layer_encoder is None:
            fuse_layer_encoder = [1, 3, 5, 7]  # 默认融合第1,3,5,7层的特征
        if target_layers is None:
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]  # 默认目标层为2-9层
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0  # 设置注册令牌数量
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)  # 准备输入令牌
        en_list = []  # 用于存储目标层特征的列表
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:  # 只处理到目标层的最后一层
                if i in self.encoder_require_grad_layer:
                    x = blk(x)  # 计算梯度的层
                else:
                    with torch.no_grad():  # 不计算梯度的层
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))  # 计算特征图大小

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]  # 移除类别令牌

        x = self.fuse_feature(en_list)  # 融合特征
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)  # 通过瓶颈层

        for i, blk in enumerate(self.decoder):
            x = blk(x)  # 通过解码器

        en = [torch.cat([en_list[idx] for idx in self.fuse_layer_encoder], dim=2)]  # 编码器特征
        de = [x]  # 解码器特征

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class ViTAD(nn.Module):
    def __init__(
            self,  # 初始化方法
            encoder,  # 编码器组件
            bottleneck,  # 瓶颈层组件
            decoder,  # 解码器组件
            target_layers=None,  # 目标层，默认为None
            fuse_layer_encoder=None,  # 编码器融合层，默认为None
            fuse_layer_decoder=None,  # 解码器融合层，默认为None
            mask_neighbor_size=0,  # 掩码邻域大小，默认为0
            remove_class_token=False,  # 是否移除类别标记，默认为False
    ) -> None:  # 返回类型注解，表示不返回任何值
        super(ViTAD, self).__init__()  # 调用父类的初始化方法
        # 设置默认值，如果未提供
        if fuse_layer_decoder is None:
            fuse_layer_decoder = [2, 5, 8]  # 默认解码器融合层索引
        if fuse_layer_encoder is None:
            fuse_layer_encoder = [0, 1, 2]  # 默认编码器融合层索引
        if target_layers is None:
            target_layers = [2, 5, 8, 11]  # 默认目标层索引

        # 初始化模型组件
        self.encoder = encoder  # 设置编码器
        self.bottleneck = bottleneck  # 设置瓶颈层
        self.decoder = decoder  # 设置解码器
        self.target_layers = target_layers  # 设置目标层
        self.fuse_layer_encoder = fuse_layer_encoder  # 设置编码器融合层
        self.fuse_layer_decoder = fuse_layer_decoder  # 设置解码器融合层
        self.remove_class_token = remove_class_token  # 设置是否移除类别标记

        # 检查并设置编码器的注册令牌数量
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0  # 如果没有属性，则设置为0
        self.mask_neighbor_size = mask_neighbor_size  # 设置掩码邻域大小

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
            x = x[:, 1 + self.encoder.num_register_tokens:, :]

        # x = torch.cat(en_list, dim=2)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [en_list[idx] for idx in self.fuse_layer_encoder]
        de = [de_list[idx] for idx in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de


class ViTillv2(nn.Module):
    def __init__(
            self,  # 构造函数
            encoder,  # 编码器，用于输入数据的特征提取
            bottleneck,  # 瓶颈层，用于压缩和提取关键特征
            decoder,  # 解码器，用于从压缩特征中重建数据
            target_layers=None  # 目标层列表，用于指定关注的网络层，默认为None
    ) -> None:  # 函数返回类型注解，表示不返回任何值
        super(ViTillv2, self).__init__()  # 调用父类的构造函数，初始化继承的属性
        if target_layers is None:  # 检查是否提供了目标层列表
            target_layers = [2, 3, 4, 5, 6, 7]  # 如果未提供，则使用默认的目标层列表
        self.encoder = encoder  # 将传入的编码器赋值给实例变量
        self.bottleneck = bottleneck  # 将传入的瓶颈层赋值给实例变量
        self.decoder = decoder  # 将传入的解码器赋值给实例变量
        self.target_layers = target_layers  # 将目标层列表赋值给实例变量
        if not hasattr(self.encoder, 'num_register_tokens'):  # 检查编码器是否具有num_register_tokens属性
            self.encoder.num_register_tokens = 0  # 如果没有，则设置该属性为0

    def forward(self, x):
        """
        前向传播函数
        输入x: 输入数据
        返回: 编码器特征和解码器特征

        Args:
            x:

        Returns:
        """
        x = self.encoder.prepare_tokens(x)
        # 初始化编码器特征列表
        en = []
        # 遍历编码器块
        for i, blk in enumerate(self.encoder.blocks):
            # 只处理目标层之前的块
            if i <= self.target_layers[-1]:
                # 使用torch.no_grad()不计算梯度，节省内存
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            # 如果当前层在目标层中，则保存特征
            if i in self.target_layers:
                en.append(x)

        # 融合特征
        x = self.fuse_feature(en)
        # 通过瓶颈层
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        # 初始化解码器特征列表
        de = []
        # 遍历解码器块
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de.append(x)

        # 计算特征图的边长
        side = int(math.sqrt(x.shape[1]))

        # 移除register tokens，只保留特征部分
        en = [e[:, self.encoder.num_register_tokens + 1:, :] for e in en]
        de = [d[:, self.encoder.num_register_tokens + 1:, :] for d in de]

        # 调整特征维度并重塑为空间维度
        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        # 返回反转的编码器特征和解码器特征
        return en[::-1], de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class ViTillv3(nn.Module):
    def __init__(
            self,
            teacher,  # 教师模型
            student,  # 学生模型
            target_layers=None,  # 目标层列表，用于特征提取
            fuse_dropout=0.,  # 融合时的dropout率
    ) -> None:
        """
        初始化函数，用于设置教师模型、学生模型及相关参数
        参数:
            teacher: 教师模型，用于提供知识蒸馏的指导
            student: 学生模型，将被教师模型指导进行训练
            target_layers: 目标层列表，用于特征提取，默认为None
            fuse_dropout: 融合时的dropout率，默认为0
        """
        super(ViTillv3, self).__init__()
        if target_layers is None:
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]  # 默认目标层，用于特征提取的层索引
        self.teacher = teacher  # 保存教师模型引用
        self.student = student  # 保存学生模型引用
        if fuse_dropout > 0:
            self.fuse_dropout = nn.Dropout(fuse_dropout)  # 创建dropout层，用于防止过拟合
        else:
            self.fuse_dropout = nn.Identity()  # 如果dropout为0，使用恒等变换，不进行dropout操作
        self.target_layers = target_layers  # 保存目标层列表
        if not hasattr(self.teacher, 'num_register_tokens'):
            self.teacher.num_register_tokens = 0  # 如果教师模型没有注册令牌属性，则设为0，确保模型一致性

    def forward(self, x):
        # 教师模型前向传播，获取特征
        with torch.no_grad():
            patch = self.teacher.prepare_tokens(x)  # 准备输入令牌
            x = patch
            en = []  # 用于存储教师模型特征
            for i, blk in enumerate(self.teacher.blocks):
                if i <= self.target_layers[-1]:
                    x = blk(x)  # 通过教师模型的块
                else:
                    continue
                if i in self.target_layers:
                    en.append(x)  # 收集目标层特征
            en = self.fuse_feature(en, fuse_dropout=False)  # 融合特征

        # 学生模型前向传播，获取特征
        x = patch
        de = []  # 用于存储学生模型特征
        for i, blk in enumerate(self.student):
            x = blk(x)  # 通过学生模型的块
            if i in self.target_layers:
                de.append(x)  # 收集目标层特征
        de = self.fuse_feature(de, fuse_dropout=False)  # 融合特征

        # 处理特征，移除注册令牌并重塑形状
        en = en[:, 1 + self.teacher.num_register_tokens:, :]
        de = de[:, 1 + self.teacher.num_register_tokens:, :]
        side = int(math.sqrt(en.shape[1]))  # 计算特征图大小

        # 重塑特征为图像形式
        en = en.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        de = de.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        return [en.contiguous()], [de.contiguous()]  # 返回连续的特征图

    def fuse_feature(self, feat_list, fuse_dropout=False):
        """融合特征列表"""
        if fuse_dropout:
            feat = torch.stack(feat_list, dim=1)  # 堆叠特征
            feat = self.fuse_dropout(feat).mean(dim=1)  # 应用dropout并平均
            return feat
        else:
            return torch.stack(feat_list, dim=1).mean(dim=1)  # 直接平均特征


class ReContrast(nn.Module):
    def __init__(
            self,
            encoder,  # 编码器网络
            encoder_freeze,  # 冻结编码器网络
            bottleneck,  # 瓶颈层网络
            decoder,  # 解码器网络
    ) -> None:
        super(ReContrast, self).__init__()
        self.encoder = encoder
        self.encoder.layer4 = None  # 移除编码器的layer4层
        self.encoder.fc = None  # 移除编码器的全连接层

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.layer4 = None  # 移除冻结编码器的layer4层
        self.encoder_freeze.fc = None  # 移除冻结编码器的全连接层

        self.bottleneck = bottleneck  # 设置瓶颈层
        self.decoder = decoder  # 设置解码器

    def forward(self, x):
        """
        前向传播函数，处理输入数据并返回编码器和解码器的输出

        参数:
            x: 输入数据

        返回:
            en_freeze + en: 编码器输出列表，包含冻结编码器和正常编码器的输出
            de: 解码器输出列表，包含特定选择的输出部分
        """
        en = self.encoder(x)  # 通过编码器处理输入，获取编码器输出
        with torch.no_grad():  # 不计算梯度，冻结编码器的前向传播
            en_freeze = self.encoder_freeze(x)  # 使用冻结的编码器处理输入
        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]  # 拼接两个编码器的输出，沿第0维度连接
        de = self.decoder(self.bottleneck(en_2))  # 通过瓶颈层和解码器处理拼接后的特征
        de = [a.chunk(dim=0, chunks=2) for a in de]  # 将解码器输出分成两部分
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]  # 选择特定的输出部分
        return en_freeze + en, de  # 返回编码器输出和解码器输出

    def train(self, mode=True, encoder_bn_train=True):
        """
        设置模型及其子模块的训练模式

        参数:
            mode (bool): 是否设置为训练模式，True为训练模式，False为评估模式
            encoder_bn_train (bool): 是否将编码器的批归一化层设置为训练模式

        返回:
            self: 返回模型实例本身，支持链式调用
        """
        self.training = mode  # 设置模型的训练模式标志
        if mode is True:  # 如果设置为训练模式
            if encoder_bn_train:
                self.encoder.train(True)  # 设置编码器的训练模式
            else:
                self.encoder.train(False)  # 设置编码器为评估模式
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
        else:
            self.encoder.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
        return self


def update_moving_average(ma_model, current_model, momentum=0.99):
    """
    更新移动平均模型的参数和缓冲区
    参数:
        ma_model: 移动平均模型，其参数将被更新
        current_model: 当前模型，其参数将用于更新移动平均模型
        momentum: 动量参数，控制新旧参数的权重比例，默认为0.99
    该函数遍历模型的参数和缓冲区，使用update_average函数进行更新
    """
    # 遍历当前模型和移动平均模型的参数，并更新移动平均模型的参数
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data  # 获取旧参数和新参数
        ma_params.data = update_average(old_weight, up_weight)  # 使用update_average函数更新参数

    # 遍历当前模型和移动平均模型的缓冲区，并更新移动平均模型的缓冲区
    for current_buffers, ma_buffers in zip(current_model.buffers(), ma_model.buffers()):
        old_buffer, up_buffer = ma_buffers.data, current_buffers.data  # 获取旧缓冲区和新缓冲区
        ma_buffers.data = update_average(old_buffer, up_buffer, momentum)  # 使用update_average函数更新缓冲区


def update_average(old, new, momentum=0.99):
    """
    计算移动平均值

    参数:
        old: 旧的平均值，如果为None则返回新值
        new: 新的数值
        momentum: 动量参数，默认值为0.99，用于控制旧值和新值的权重比例

    返回:
        计算后的移动平均值
    """
    if old is None:  # 如果旧值为None，直接返回新值
        return new
    return old * momentum + (1 - momentum) * new  # 使用动量公式计算移动平均值


def disable_running_stats(model):
    """
    禁用模型中所有批归一化(BatchNorm)层的运行统计信息
    参数:
        model: 要处理的神经网络模型
    这个函数通过将BatchNorm层的momentum参数设置为0来实现禁用运行统计的目的。
    momentum=0意味着不会更新均值和方差等统计信息，从而保持初始状态。
    """

    def _disable(module):
        """
        内部辅助函数，用于禁止单个模块的运行统计
        参数:
            module: 网络中的单个模块/层
        判断如果模块是BatchNorm层(_BatchNorm)，则备份其原始momentum值，
        并将momentum设置为0以禁用统计信息的更新
        """
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum  # 备份原始momentum值
            module.momentum = 0  # 设置momentum为0，禁用统计更新

    model.apply(_disable)  # 对模型中的所有模块应用_disable函数


def enable_running_stats(model):
    """
    启用模型中所有BatchNorm层的running stats（运行时的均值和方差）
    参数:
        model: 要处理的神经网络模型
    这个函数会遍历模型中的所有模块，如果模块是BatchNorm层并且有备份的momentum值，
    则恢复其原始的momentum值，从而启用running stats的更新。
    """

    def _enable(module):
        """
        内部辅助函数，用于处理单个模块
        参数:
            module: 网络中的单个模块
        如果模块是BatchNorm层并且有backup_momentum属性，
        则将备份的momentum值恢复到当前的momentum属性
        """
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum  # 恢复原始momentum值

    model.apply(_enable)  # 对模型中的所有模块应用_enable函数
