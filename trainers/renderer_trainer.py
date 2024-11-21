import os
import random

import torch
import torch.distributed as dist
import torchvision
import torch_fidelity

import utils
from utils.geometry_liif import make_coord_cell_grid

from .trainers import register
from trainers.base_trainer import BaseTrainer

@register('renderer_trainer')
class RendererTrainer(BaseTrainer):

    def prepare_visualize(self):
        self.vis_spec = dict()

        def get_samples(dataset, s):
            n = len(dataset)
            lst = [dataset[i] for i in list(range(0, n, n // s))[:s]]
            data = dict()
            for k in lst[0].keys():
                data[k] = torch.stack([_[k] for _ in lst]).cuda()
            return data

        self.vis_spec['ds_samples'] = self.cfg.visualize.get('ds_samples', 0)
        if self.vis_spec['ds_samples'] > 0:
            self.vis_ds_samples = {'train': get_samples(self.datasets['train'], self.vis_spec['ds_samples'])}
            if self.datasets.get('val') is not None:
                self.vis_ds_samples['val'] = get_samples(self.datasets['val'], self.vis_spec['ds_samples'])

    def make_datasets(self):
        super().make_datasets()

        self.vis_resolution = self.cfg.visualize.resolution
        if isinstance(self.vis_resolution, int):
            self.vis_resolution = (self.vis_resolution, self.vis_resolution)
        if self.is_master:
            random.seed(0) # to get a fixed vis set from wrapper_cae
            # self.prepare_visualize()
            if self.cfg.random_seed is not None:
                random.seed(self.cfg.random_seed + self.rank)
            else:
                random.seed()

    def make_model(self, model_spec=None):
        super().make_model(model_spec)
        for name, m in self.model.named_children():
            self.log(f'  .{name} {utils.compute_num_params(m)}')

        self.has_opt = dict()
        if self.cfg.get('optimizers') is not None:
            for name in self.cfg.optimizers.keys():
                self.has_opt[name] = True

    def make_optimizers(self):
        self.optimizers = dict()
        for name, spec in self.cfg.optimizers.items():
            self.optimizers[name] = utils.make_optimizer(self.model.get_params(name), spec)

    def train_step(self, batch, bp=True):
        gan_iter = self.cfg.get("gan_start_after_iters")
        use_gan = ((gan_iter is not None) and self.iter > gan_iter)

        ret = self.model_ddp(batch, mode="loss", use_gan=use_gan)
        loss = ret.pop("loss")
        ret["loss"] = loss.item()

        if bp:
            self.model_ddp.zero_grad()
            loss.backward()
            for name, o in self.optimizers.items():
                if name != "disc":
                    o.step()
        
        if use_gan:
            disc_ret = self.model_ddp(batch, mode="disc_loss", use_gan=use_gan)
            loss = disc_ret.pop("loss")
            ret["disc_loss"] = loss.item()
            ret.update(disc_ret)

            if bp:
                self.optimizers["disc"].zero_grad()
                loss.backward()
                self.optimizers["disc"].step()

        return ret
    
    def train_iter_start(self):
        pass

    def run_training(self):
        super().run_training()

    def visualize(self):
        self.model_ddp.eval()
        if self.is_master:
            with torch.no_grad():
                if self.vis_spec['ds_samples'] > 0:
                    self.visualize_ae()
        if self.distributed:
            dist.barrier()

    def visualize_ae_(self, name, data, bs=1):
        gt = data['gt']
        n = data['inp'].shape[0]
        pred = []
        center_zoom = []

        for i in range(0, n, bs):
            d = {k: v[i: min(i + bs, n)] for k, v in data.items()}
            pred.append(self.model(d, mode='pred'))

            if (self.vis_ae_center_zoom_res is not None) and (not name.endswith('_whole')):
                r0 = self.vis_resolution[0] / self.vis_ae_center_zoom_res
                r1 = self.vis_resolution[1] / self.vis_ae_center_zoom_res
                d['gt_coord'], d['gt_cell'] = make_coord_cell_grid(
                    self.vis_resolution, [[-r0, r0], [-r1, r1]], device=d['gt_coord'].device, bs=d['gt_coord'].shape[0])
                center_zoom.append(self.model(d, mode='pred'))

        pred = torch.cat(pred, dim=0)
        if self.is_master:
            vimg = []
            for i in range(len(gt)):
                vimg.extend([pred[i], gt[i]])
            vimg = torch.stack(vimg)
            vimg = torchvision.utils.make_grid(vimg, nrow=4, normalize=True, value_range=(-1, 1))
            self.log_image(name, vimg)

        if (self.vis_ae_center_zoom_res is not None) and (not name.endswith('_whole')):
            center_zoom = torch.cat(center_zoom, dim=0)
            if self.is_master:
                vimg = []
                for i in range(len(gt)):
                    vimg.extend([center_zoom[i], center_zoom[i]])
                vimg = torch.stack(vimg)
                vimg = torchvision.utils.make_grid(vimg, nrow=4, normalize=True, value_range=(-1, 1))
                self.log_image(name + '_center_zoom', vimg)

    def visualize_ae(self):
        for split in ['train', 'val']:
            if self.vis_ds_samples.get(split) is None:
                continue
            data = self.vis_ds_samples[split]
            self.visualize_ae_(split, data)

            if self.cfg.visualize.get('vis_ae_whole', False):
                x = data['inp']
                coord, cell = make_coord_cell_grid(x.shape[-2:], device=x.device, bs=x.shape[0])
                data_whole = {'inp': x, 'gt': x, 'gt_coord': coord, 'gt_cell': cell}
                self.visualize_ae_(split + '_whole', data_whole)

