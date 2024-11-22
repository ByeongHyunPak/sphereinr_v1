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

    def make_datasets(self):
        super().make_datasets()

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

    def visualize_ae(self):
        for split in ['train', 'val']:
            if self.vis_ds_samples.get(split) is None:
                continue
            data = self.vis_ds_samples[split]
            self.visualize_ae_(split, data)

    def visualize_ae_(self, name, data, bs=1):
        gt = data['gt']
        n = data['inp'].shape[0]
        pred = []

        for i in range(0, n, bs):
            d = {k: v[i: min(i + bs, n)] for k, v in data.items()}
            pred.append(self.model(d, mode='pred'))
            
        pred = torch.cat(pred, dim=0)
        if self.is_master:
            vimg = []
            for i in range(len(gt)):
                vimg.extend([pred[i], gt[i]])
            vimg = torch.stack(vimg)
            vimg = torchvision.utils.make_grid(vimg, nrow=4, normalize=True, value_range=(-1, 1))
            self.log_image(name, vimg)