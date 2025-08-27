# 分配ADer任务
import argparse
from ADer.configs import get_cfg
from ADer.util.net import init_training
from ADer.util.util import run_pre, init_checkpoint
from ADer.trainer import get_trainer


class ADerTaskAssigner:
    def __init__(self, method):
        assert method in ['MambaAD', 'InVad', 'ViTAD', 'DiAD', 'UniAD']
        self.method = method
        if self.method == 'MambaAD':
            self.cfg_path = 'ADer/configs/mambaad/mambaad_spk.py'
        elif self.method == 'InVad':
            self.cfg_path = 'ADer/configs/invad/invad_spk.py'
        elif self.method == 'ViTAD':
            self.cfg_path = 'ADer/configs/vitad/vitad_spk.py'
        elif self.method == 'DiAD':
            # 暂未实现
            self.cfg_path = 'ADer/configs/diad/diad_spk.py'
        elif self.method == 'UniAD':
            self.cfg_path = 'ADer/configs/uniad/uniad_spk.py'
        else:
            raise NotImplementedError

    def test(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--cfg_path', default=self.cfg_path)
        parser.add_argument('-m', '--mode', default='test', choices=['train', 'test'])
        parser.add_argument('--sleep', type=int, default=-1)
        parser.add_argument('--memory', type=int, default=-1)
        parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
        parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
        parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER, )
        cfg_terminal = parser.parse_args()
        cfg = get_cfg(cfg_terminal)

        # 添加可视化配置
        cfg.vis = True
        cfg.vis_dir = 'vis'

        run_pre(cfg)
        init_training(cfg)
        init_checkpoint(cfg)
        i_trainer = get_trainer(cfg)
        i_trainer.run()

    def train(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--cfg_path', default=self.cfg_path)
        parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
        parser.add_argument('--sleep', type=int, default=-1)
        parser.add_argument('--memory', type=int, default=-1)
        parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
        parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
        parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER, )
        cfg_terminal = parser.parse_args()
        cfg = get_cfg(cfg_terminal)
        run_pre(cfg)
        init_training(cfg)
        init_checkpoint(cfg)
        i_trainer = get_trainer(cfg)
        i_trainer.run()


class MambaAD(ADerTaskAssigner):
    def __init__(self, method='MambaAD'):
        super().__init__(method)


class InVad(ADerTaskAssigner):
    def __init__(self, method='InVad'):
        super().__init__(method)


class ViTAD(ADerTaskAssigner):
    def __init__(self, method='ViTAD'):
        super().__init__(method)


class DiAD(ADerTaskAssigner):
    def __init__(self, method='DiAD'):
        super().__init__(method)


class UniAD(ADerTaskAssigner):
    def __init__(self, method='UniAD'):
        super().__init__(method)
