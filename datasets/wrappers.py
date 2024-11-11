import math
import random

import torch
import numpy as np

from io import BytesIO
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import datasets
from datasets import register
from utils import to_pixel_samples

def resize_fn(img, size):
    return transforms.Resize(size, Image.BICUBIC)(img)

@register('custom_dataset')
class CustomDataset(Dataset):

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
            w, h = img.size
            self.gt_h_max = h
        
        gt_h = random.randint(self.gt_h_min, self.gt_h_max)
        gt_img = resize_fn(img, (gt_h, 2 * gt_h))

        # TODO calculate spherical coord. on ERP grid
        # TODO calculate cell decode
        # TODO random crop gt_img and corresponding coord and cell      
        
        

        return {
            'inp': inp,
            'prompt': text,
        }

        



