import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet

import BaseASD.DifferNet.config as c
from BaseASD.DifferNet.freia_funcs import permute_layer, glow_coupling_layer, F_fully_connected, \
    ReversibleGraphNet, OutputNode, InputNode, Node

# 获取当前文件路径
current_path = os.path.dirname(os.path.abspath(__file__))

WEIGHT_DIR = os.path.join(current_path, 'weights')
MODEL_DIR = os.path.join(current_path, 'models')


def nf_head(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp_alpha, 'F_class': F_fully_connected,
                           'F_args': {'internal_size': c.fc_internal, 'dropout': c.dropout}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


class DifferNet(nn.Module):
    def __init__(self):
        super(DifferNet, self).__init__()
        self.feature_extractor = alexnet(pretrained=True)
        self.nf = nf_head()

    def forward(self, x):
        y_cat = list()

        for s in range(c.n_scales):
            x_scaled = F.interpolate(x, size=c.img_size[0] // (2 ** s)) if s > 0 else x
            feat_s = self.feature_extractor.features(x_scaled)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))

        y = torch.cat(y_cat, dim=1)
        z = self.nf(y)
        return z


def save_model(model, filename):
    """
    保存完整模型
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))

def save_weights(model, filename):
    """
    保存模型权重
    """
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))


def load_model(filename):
    """
    加载完整模型
    """
    path = os.path.join(MODEL_DIR, filename)
    print("Loading model file: ", path)
    # 判断模型文件是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found.")
    model_ins = torch.load(path, weights_only=False)
    return model_ins


def load_weights(model, filename):
    """
    加载模型权重
    """
    path = os.path.join(WEIGHT_DIR, filename)
    print("Loading weights file: ", path)
    # 判断模型文件是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights file {path} not found.")
    model.load_state_dict(torch.load(path))

