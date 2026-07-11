import glob
import importlib

from util.registry import Registry
TRAINER = Registry('Trainer')

files = glob.glob('trainer/[!_]*.py')
for file in files:
    try:
        model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))
    except (ImportError, AttributeError) as e:
        import warnings
        warnings.warn(f"Skip loading trainer module {file}: {e}")


def get_trainer(cfg):
	return TRAINER.get_module(cfg.trainer.name)(cfg)
