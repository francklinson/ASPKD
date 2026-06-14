import yaml
import os

def _get_default_config_path():
    """动态检测项目根目录，返回配置文件的绝对路径"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "backend", "config", "config.yaml")

def yaml_load(config_path=None):
    """
    load yaml file
    """
    if config_path is None:
        config_path = _get_default_config_path()
    param_in_config_file = None
    with open(config_path, encoding='utf-8') as stream:
        param_in_config_file = yaml.safe_load(stream)
    if param_in_config_file:
        # 配置环境变量
        if "environments" in param_in_config_file:
            for env_var in param_in_config_file["environments"]:
                os.environ[str(env_var)] = str(param_in_config_file["environments"][env_var])
            print("完成环境变量配置!!!")
    else:
        raise FileNotFoundError("未正确读取到配置文件，请检查!!!")

    return param_in_config_file