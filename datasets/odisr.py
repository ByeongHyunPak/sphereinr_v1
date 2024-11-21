import math
import random

import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import datasets
from datasets import register

from utils.geometry_liif import make_coord_grid

def resize_fn(img, size):
    return transforms.Resize(size, Image.BICUBIC)(img)

@register('odisr-liif')
class ODISR_LIIF(Dataset):

    def __init__(self,
                 dataset,
                 lr_inp_size=(512, 1024),
                 hr_tgt_size=(4096, 8192),
                 gt_crop_size=(256, 256),
                 gt_size_min=(1024, 2048),
                 gt_size_max=None):
        
        self.img_folder = datasets.make(dataset)

        self.lr_inp_size = lr_inp_size
        self.hr_h_tgt = hr_tgt_size[0]
        
        self.gt_crop_size = gt_crop_size
        self.gt_h_min = gt_size_min[0]
        self.gt_h_max = gt_size_max[0] \
            if gt_size_max is not None else None
    
    def __len__(self):
        return len(self.img_folder)
    
    def __getitem__(self, idx): 

        img = self.img_folder[idx]

        # pre-upsampling hr image
        if img.size[-1] < self.hr_h_tgt:
            size = (self.hr_h_tgt, 2* self.hr_h_tgt)
            img = resize_fn(img, size)

        # lr input image
        inp = resize_fn(img, self.lr_inp_size)
        inp = transforms.ToTensor()(inp)

        # hr gt patch
        if self.gt_h_max is None:
            self.gt_h_max = img.size[-1]
        gt_h = random.randint(self.gt_h_min, self.gt_h_max)
        gt_img = resize_fn(img, (gt_h, 2 * gt_h))
        gt_img = transforms.ToTensor()(gt_img)

        # ERP grid (naive regular grid - liif)
        grid = make_coord_grid(gt_img.shape[-2:]) # (h, w, 2)

        # calculate cell decode (naive regular cell - liif)
        cell = torch.ones_like(grid) # (h, w, 2)
        cell[:, :, 0] *= 2 / gt_img.shape[-2]
        cell[:, :, 1] *= 2 / gt_img.shape[-1]

        # random crop gt_img and corresponding coord and cell
        cs = self.gt_crop_size[0]
        i0 = random.randint(0, gt_img.shape[-2] - cs)
        j0 = random.randint(0, gt_img.shape[-1] - cs)
        crop_gt_img = gt_img[:, i0:i0+cs, j0:j0+cs] # (3, cs, cs)
        crop_grid = grid[i0:i0+cs, j0:j0+cs, :] # (cs, cs, 2)
        crop_cell = cell[i0:i0+cs, j0:j0+cs, :] # (cs, cs, 2)

        return {
            'inp': inp,
            'gt': crop_gt_img,
            'gt_coord': crop_grid,
            'gt_cell': crop_cell,
        }