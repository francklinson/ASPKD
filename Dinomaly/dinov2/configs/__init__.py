import pathlib

from omegaconf import OmegaConf


def load_config(config_name: str):
    """
    加载指定名称的YAML配置文件
    参数:
        config_name (str): 配置文件的名称，不带.yaml扩展名
    返回:
        OmegaConf对象: 包含加载的配置数据的OmegaConf对象
    说明:
        该函数会将传入的config_name与.yaml扩展名拼接成完整的文件名，
        然后从当前脚本所在目录的父目录中查找并加载该配置文件
    """
    # 构建完整的配置文件名，添加.yaml扩展名
    config_filename = config_name + ".yaml"
    # 返回加载的配置文件，路径为当前脚本所在目录的父目录下的配置文件
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)


dinov2_default_config = load_config("ssl_default_config")


def load_and_merge_config(config_name: str):
    """
    加载并合并配置文件的函数
    该函数首先创建一个默认配置，然后加载指定的配置文件，最后将两者合并。
    参数:
        config_name (str): 要加载的配置文件的名称
    返回:
        OmegaConf: 合并后的配置对象
    """
    # 创建默认配置对象
    default_config = OmegaConf.create(dinov2_default_config)
    # 加载指定的配置文件
    loaded_config = load_config(config_name)
    # 合并默认配置和加载的配置，后者会覆盖前者中的相同项
    return OmegaConf.merge(default_config, loaded_config)
