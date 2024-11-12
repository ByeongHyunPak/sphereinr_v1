import os
import random

import torch
import torch.distributed as dist

import torchvision
import torch_fidelity

import utils
from utils.geometry import make_coord_cell_grid

from trainers import register, BaseTrainer

@register('custom_trainer')
class CustomTrainer(BaseTrainer):

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

    def train_step(self, data, bp=True):
        g_iter = self.cfg.get('gan_start_after_iters')
        use_gan = ((g_iter is not None) and self.iter > g_iter)

        ret = self.model_ddp.forward_train(data, has_opt=self.has_opt, use_gan=use_gan)