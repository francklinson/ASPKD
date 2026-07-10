"""桥接到 configs/__base__/ 的 cfg 类"""
import sys, os
# 确保父目录在 sys.path 中（用于 glob.glob）
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# 从 configs.__base__ 导入所有 cfg_* 类
from configs.__base__.cfg_common import cfg_common
from configs.__base__.cfg_dataset_default import cfg_dataset_default

# 动态导入所有 cfg_model_* 类
import importlib, glob as _glob
_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '__base__')
for _file in _glob.glob(os.path.join(_base_dir, 'cfg_model_*.py')):
    _mod_name = os.path.basename(_file)[:-3]
    _mod = importlib.import_module(f'configs.__base__.{_mod_name}')
    for _obj_name in dir(_mod):
        if _obj_name.startswith('cfg_model'):
            globals()[_obj_name] = getattr(_mod, _obj_name)
