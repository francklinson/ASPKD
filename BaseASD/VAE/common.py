import yaml

def yaml_load():
    """
    load yaml file
    """
    with open("baseline.yaml", encoding='utf-8') as stream:
    # with open("BaseASD/VAE/baseline.yaml", encoding='utf-8') as stream:
    # with open("baseline.yaml", encoding='utf-8') as stream:
        param = yaml.safe_load(stream)
    return param
