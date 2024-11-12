import math
import random

import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import datasets
from datasets import register

from utils import make_coord

def resize_fn(img, size):
    return transforms.Resize(size, Image.BICUBIC)(img)

@register('liif-erp-text-paired')
class liifERPTextPaired(Dataset):

    def __init__(self, 
                 img_folder, 
                 text_folder, 
                 lr_inp_size=(512, 1024), 
                 gt_size_min=(1024, 2048), 
                 gt_size_max=None,
                 gt_crop_size=(256, 256)):

        self.img_folder = datasets.make(img_folder)
        self.text_folder = datasets.make(text_folder)

        self.lr_inp_size = lr_inp_size
        self.gt_h_min = gt_size_min[0]
        self.gt_h_max = gt_size_max[0] \
            if gt_size_max is not None else None
        self.gt_crop_size = gt_crop_size

    def __len__(self):
        return len(self.img_folder)
    
    def __getitem__(self, idx): 

        img = self.img_folder[idx]
        text = self.text_folder[idx]
        
        # get lr input image
        inp = resize_fn(img, self.lr_inp_size)

        # get hr gt patch
        if self.gt_h_max is None:
            self.gt_h_max = img.size(-1)
        gt_h = random.randint(self.gt_h_min, self.gt_h_max)
        gt_img = resize_fn(img, (gt_h, 2 * gt_h))

        # ERP grid (naive regular grid)
        grid_ij = make_coord(gt_img.shape[-2:], flatten=False) # (h, w, 2)

        # calculate cell decode (naive regular cell)
        cell_ij = torch.ones_like(grid_ij) # (h, w, 2)
        cell_ij[:, :, 0] *= 2 / gt_img.shape[-2]
        cell_ij[:, :, 1] *= 2 / gt_img.shape[-1]

        # random crop gt_img and corresponding coord and cell
        cs = self.gt_crop_size
        i0 = random.randint(0, gt_img.shape[-2] - cs)
        j0 = random.randint(0, gt_img.shape[-1] - cs)
        crop_gt_img = gt_img[:, i0:i0+cs, j0:j0+cs].view(3, -1).permute(1, 0) # (cs*cs, 3)
        crop_grid_ij = grid_ij[i0:i0+cs, j0:j0+cs, :].view(-1, 2) # (cs*cs, 2)
        crop_cell_ij = cell_ij[i0:i0+cs, j0:j0+cs, :].view(-1, 2) # (cs*cs, 2)

        return {
            'inp': inp,
            'prompt': text,
            'gt': crop_gt_img,
            'gt_coord': crop_grid_ij,
            'gt_cell': crop_cell_ij,
        }
    
    
