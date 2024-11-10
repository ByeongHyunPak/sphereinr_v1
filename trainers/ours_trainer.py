import os
import random

import torch
import torch.distributed as dist

import torchvision
import torch_fidelity

import utils
from utils.geometry import make_coord_cell_grid

from trainers import register, BaseTrainer

@register('ours_trainer')
class OURSTrainer(BaseTrainer):

    def train_step(self, data, bp=True):
        pass