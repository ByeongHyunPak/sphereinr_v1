import os
import random

import torch
import torch.distributed as dist

import torchvision
import torch_fidelity

import utils

from trainers import register, BaseTrainer

@register('custom_trainer')
class CustomTrainer(BaseTrainer):

    def prepare_visualize(self):
        pass

    def make_datasets(self):
        super().make_datasets()
        pass

    def make_model(self, model_spec=None):
        super().make_model(model_spec)()
        pass

    def make_optimizers(self):
        super().make_optimizers()
        pass

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

            if bp:
                self.optimizers["disc"].zero_grad()
                loss.backward()
                self.optimizers["disc"].step()

        return ret
