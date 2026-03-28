import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class DINOHead(nn.Module):
    def __init__(
            self,  # 初始化函数
            in_dim,  # 输入维度
            out_dim,  # 输出维度
            use_bn=False,  # 是否使用批归一化(Batch Normalization)
            nlayers=3,  # MLP层数
            hidden_dim=2048,  # 隐藏层维度
            bottleneck_dim=256,  # 瓶颈层维度
            mlp_bias=True,  # MLP层是否使用偏置项
    ):
        super().__init__()  # 调用父类的初始化方法
        nlayers = max(nlayers, 1)  # 确保层数至少为1层
        # 构建多层感知机(MLP)结构
        self.mlp = _build_mlp(
            nlayers,  # 层数
            in_dim,  # 输入维度
            bottleneck_dim,  # 瓶颈层维度
            hidden_dim=hidden_dim,  # 隐藏层维度
            use_bn=use_bn,  # 是否使用批归一化
            bias=mlp_bias,  # 是否使用偏置项
        )
        # 定义最后一层线性变换层，将瓶颈层维度映射到输出维度，不使用偏置项
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, no_last_layer=False, only_last_layer=False):
        if not only_last_layer:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        if not no_last_layer:
            x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    """
    构建多层感知机(MLP)的函数
    参数:
        nlayers (int): 网络层数
        in_dim (int): 输入维度
        bottleneck_dim (int): 输出维度/瓶颈层维度
        hidden_dim (int, optional): 隐藏层维度，默认为None
        use_bn (bool, optional): 是否使用批归一化(Batch Normalization)，默认为False
        bias (bool, optional): 是否在线性层中使用偏置项，默认为True
    返回:
        nn.Sequential: 构建好的MLP模型
    """
    if nlayers == 1:
        # 如果只有一层，直接返回一个线性层，将输入维度映射到瓶颈维度
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        # 初始化层列表
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        # 如果需要使用批归一化，则添加BatchNorm1d层
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        # 添加GELU激活函数
        layers.append(nn.GELU())
        # 循环添加中间层（除了第一层和最后一层）
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        # 添加输出层，将隐藏层维度映射到瓶颈维度
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        # 将所有层组合成一个顺序模型并返回
        return nn.Sequential(*layers)
