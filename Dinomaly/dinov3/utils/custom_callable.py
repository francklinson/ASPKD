import contextlib
import importlib
import inspect
import os
import sys
from pathlib import Path


@contextlib.contextmanager  # 使用contextlib装饰器将函数转换为上下文管理器
def _load_modules_from_dir(dir_: str):  # 定义一个名为_load_modules_from_dir的函数，接受一个目录路径作为参数
    sys.path.insert(0, dir_)  # 将指定目录插入到Python模块搜索路径(sys.path)的开头
    yield  # 通过yield生成器，将控制权暂时交给上下文管理器的使用方
    sys.path.pop(0)  # 当退出上下文时，从模块搜索路径中移除刚才添加的目录


def load_custom_callable(module_path: str | Path, callable_name: str):
    """
    从指定模块路径加载可调用对象
    参数:
        module_path: 模块文件路径，可以是字符串或Path对象
        callable_name: 要加载的可调用对象名称
    返回:
        加载的可调用对象
    异常:
        AssertionError: 当模块文件不存在时抛出
    """
    # 获取模块的完整绝对路径
    module_full_path = os.path.realpath(module_path)
    # 断言模块文件必须存在，否则抛出异常
    assert os.path.exists(module_full_path), f"module {module_full_path} does not exist"
    # 分割路径，获取模块所在目录和文件名
    module_dir, module_filename = os.path.split(module_full_path)
    # 分割文件名，获取模块名（不含扩展名）
    module_name, _ = os.path.splitext(module_filename)

    # 在模块目录中加载模块
    with _load_modules_from_dir(module_dir):
        # 导入模块
        module = importlib.import_module(module_name)
        # 检查导入的模块文件路径是否与目标路径一致，不一致则重新加载
        if inspect.getfile(module) != module_full_path:
            importlib.reload(module)
        # 从模块中获取指定的可调用对象
        callable_ = getattr(module, callable_name)

    # 返回加载的可调用对象
    return callable_


@contextlib.contextmanager  # 使用contextlib装饰器定义一个上下文管理器
def change_working_dir_and_pythonpath(new_dir):
    """
    一个上下文管理器，用于临时更改工作目录和Python路径

    Args:
        new_dir (str): 要切换到的目标目录路径

    功能:
        1. 保存当前工作目录和Python路径
        2. 切换到新的工作目录
        3. 将新目录添加到Python路径的开头
        4. 在退出时自动恢复原始设置
    """
    old_dir = Path.cwd()  # 保存当前工作目录
    new_dir = Path(new_dir).expanduser().resolve().as_posix()  # 规范化新目录路径
    old_pythonpath = sys.path.copy()  # 保存当前的Python路径
    sys.path.insert(0, new_dir)  # 将新目录添加到Python路径开头
    os.chdir(new_dir)  # 切换工作目录
    try:
        yield  # 控制权交给with代码块
    finally:

        # 无论是否发生异常，都会执行以下恢复操作
        os.chdir(old_dir)  # 恢复原始工作目录
        sys.path = old_pythonpath  # 恢复原始Python路径
