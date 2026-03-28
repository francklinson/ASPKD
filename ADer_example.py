# -*- coding: utf-8 -*-
# ADer用法示例
import sys
import os

# 添加 algorithms 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'algorithms'))

from ADer import MambaAD, ViTAD, InVad, DiAD, UniAD, CFlow,PyramidFLow,SimpleNet
m = SimpleNet()
m.train()