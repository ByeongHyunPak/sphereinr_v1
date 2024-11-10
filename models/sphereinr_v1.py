import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers

import models
from models import register

from stitchdiffusion import STITCHDIFFUSION_ARGS

@register('sphereinr_v1')
class Ours(nn.Module):

    def __init__(self, diffusion_model):
        
        if diffusion_model.name == "StitchDiffusion":
            diffusion_model['args'] = STITCHDIFFUSION_ARGS
        else:
            raise NotImplementedError(f"{diffusion_model.name} is not defined.")
        
        diffusion_model = models.make(diffusion_model)


    def get_params(self, name):
        pass

    def encode(self, x):
        pass

    def decode(self, z):
        pass

    def forward(self, data, **kwargs):
        pass




