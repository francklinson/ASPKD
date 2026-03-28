import os
import yaml
 
def load_yaml(config_path):
    """
    :param config_path:
    Args:
        config_path:

    Returns:

    """
    if config_path is None:
        raise ValueError('config_path must be a valid path string')
    filepath = os.path.join(os.getcwd(), config_path)
    with open(filepath, 'r', encoding='UTF-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs
 
 