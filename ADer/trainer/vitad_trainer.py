import os
import glob
import shutil
import time
import tabulate
import torch
import numpy as np
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from ._base_trainer import BaseTrainer
from . import TRAINER
from ADer.util.vis import vis_rgb_gt_amp
from ADer.util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from ADer.util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from ADer.util.net import get_loss_scaler, get_autocast, distribute_bn
from ADer.optim.scheduler import get_scheduler
from ADer.model import get_model
from ADer.optim import get_optim
from ADer.loss import get_loss_terms
from ADer.util.metric import get_evaluator

@TRAINER.register_module
class ViTADTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(ViTADTrainer, self).__init__(cfg)

    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        self.bs = self.imgs.shape[0]

    def forward(self):
        self.feats_t, self.feats_s = self.net(self.imgs)

    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        with self.amp_autocast():
            self.forward()
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
        self.backward_term(loss_cos, self.optim)
        update_log_term(self.log_terms.get('cos'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 1,
                        self.master)

    @torch.no_grad()
    def test(self):
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        self.reset(isTrain=False)
        imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            # if batch_idx == 10:
            # 	break
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
            update_log_term(self.log_terms.get('cos'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(),
                            1, self.master)
            # get anomaly maps
            anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats_t, self.feats_s,
                                                            [self.imgs.shape[2], self.imgs.shape[3]], uni_am=False,
                                                            amap_mode='add', gaussian_sigma=4)
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            if self.cfg.vis:
                if self.cfg.vis_dir is not None:
                    root_out = self.cfg.vis_dir
                else:
                    root_out = self.writer.logdir
                vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map,
                               self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
            imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(anomaly_map)
            cls_names.append(np.array(self.cls_name))
            anomalys.append(self.anomaly.cpu().numpy().astype(int))

            # # 获取真实标签，可以从test_data["img_path"]下手，如果没有good,则是异常的
            # gt_labels.append([0 if "good" in img else 1 for img in test_data["img_path"]])

            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
            # ---------- log ----------
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
                    log_msg(self.logger, msg)
        # merge results
        if self.cfg.dist:
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
            torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
            if self.master:
                results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
                valid_results = False
                while not valid_results:
                    results_files = glob.glob(f'{self.tmp_dir}/*.pth')
                    if len(results_files) != self.cfg.world_size:
                        time.sleep(1)
                    else:
                        idx_result = 0
                        while idx_result < self.cfg.world_size:
                            results_file = results_files[idx_result]
                            try:
                                result = torch.load(results_file)
                                for k, v in result.items():
                                    results[k].extend(v)
                                idx_result += 1
                            except:
                                time.sleep(1)
                        valid_results = True
        else:
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
        if self.master:
            results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
            msg = {}
            for idx, cls_name in enumerate(self.cls_names):
                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None
                # msg += f'\n{cls_name:<10}'
                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
            msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center",
                                    stralign="center", )
            log_msg(self.logger, f'\n{msg}')

    def inference(self):
        """
        对输入样本进行inference
        Returns:
        """
        self.reset(isTrain=False)
        imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            # if batch_idx == 10:
            # 	break
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            """这里做筛选，只留下cls_name字段为INF的数据"""
            img_collector = []
            img_mask_collector = []
            anomaly_collector = []
            cls_name_collector = []
            img_path_collector = []
            any_inf = False
            for idx in range(len(test_data['cls_name'])):
                if test_data['cls_name'][idx] == 'INF':
                    any_inf = True
                    img_collector.append(test_data['img'][idx])
                    img_mask_collector.append(test_data['img_mask'][idx])
                    cls_name_collector.append(test_data['cls_name'][idx])
                    anomaly_collector.append(test_data['anomaly'][idx])
                    img_path_collector.append(test_data['img_path'][idx])
                    print("Inferencing: ", img_path_collector[-1])
            if not any_inf:
                continue
            # 把new_test_data['img']、new_test_data['img_mask']、new_test_data['anomaly']都转成tensor数据
            new_test_data = {'img': torch.stack(img_collector, dim=0),
                             'img_mask': torch.stack(img_mask_collector, dim=0),
                             'cls_name': cls_name_collector,
                             'anomaly': torch.stack(anomaly_collector, dim=0),
                             'img_path': img_path_collector}

            # 设置输入数据
            self.set_input(new_test_data)
            self.forward()
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
            update_log_term(self.log_terms.get('cos'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(),
                            1, self.master)
            # get anomaly maps
            anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats_t, self.feats_s,
                                                            [self.imgs.shape[2], self.imgs.shape[3]], uni_am=False,
                                                            amap_mode='add', gaussian_sigma=4)
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            if self.cfg.vis:
                if self.cfg.vis_dir is not None:
                    root_out = self.cfg.vis_dir
                else:
                    root_out = self.writer.logdir
                vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map,
                               self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
            imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(anomaly_map)
            cls_names.append(np.array(self.cls_name))
            anomalys.append(self.anomaly.cpu().numpy().astype(int))

            # # 获取真实标签，可以从test_data["img_path"]下手，如果没有good,则是异常的
            # gt_labels.append([0 if "good" in img else 1 for img in test_data["img_path"]])

            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
