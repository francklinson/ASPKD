import yaml
import os

def yaml_load(config_path="config/algorithms.yaml"):
    """
    load yaml file
    """
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