# -*- coding: utf-8 -*-
"""
项目路径统一解析模块

所有模块应通过此处获取 PROJECT_ROOT，避免各自重复 os.path.dirname(__file__) 链。
"""
import os

# 项目根目录: _paths.py → core/ → backend/ → 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_path(*parts: str) -> str:
    """将相对于项目根目录的路径片段拼接为绝对路径"""
    return os.path.join(PROJECT_ROOT, *parts)
