# -*- coding: utf-8 -*-
"""
Database connector module

默认使用进程内内存数据库 (InMemoryConnector)，无需外部数据库服务。
MySQLConnector 仍然可用 (backend="mysql")。
"""

from .connector import MySQLConnector, DatabaseChecker
from .in_memory import InMemoryConnector, InMemoryDatabaseChecker, _MemDB

__all__ = [
    'MySQLConnector',
    'DatabaseChecker',
    'InMemoryConnector',
    'InMemoryDatabaseChecker',
]
