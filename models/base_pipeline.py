import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers

import models
from models import register
from models.vqgan.lpips import lpips
from models.vqgan.quantizer import VectorQuantizer
from models.vqgan.discriminator import make_discriminator


@register('base_pipeline')
class basePipeline(nn.Module):

    def __init__(
            self, 
            disc=None,
            denoiser=None,
            renderer=None,
            loss_cfg = dict(),
            **kwargs
        ):

        self.vae = ... # TODO build custom AutoencoderKL's Decoder
        self.renderer = models.make(renderer) if renderer else None

        input_nc = 3 if not disc.disc_cond_scale else 4
        self.disc_cond_scale = disc.disc_cond_scale
        self.disc = make_discriminator(**disc, input_nc=input_nc) if disc else None

        if denoiser is not None:
            self.unet = ...
            self.tokenizer = ...
            self.text_encoder = ...
            self.scheduler = ...

        self.loss_cfg = loss_cfg

    def get_gd_from_opt(self, opt):
        if opt is None:
            opt = dict()
        gd = dict()
        # TODO Fill with components
        return gd

    def get_params(self, name):
        if name == 'vae':
            return self.vae.parameters()
        elif name == 'renderer':
            return self.renderer.parameters()

    def encode_latents(self, x):
        x = x.to(self.vae.dtype)
        z = self.vae.encode(x).latent_dist.sample()
        z = z * self.vae.config.scaling_factor
        return z.to(self.dtype)

    def decode_latents(self, z):
        z = 1 / self.vae.config.scaling_factor * z
        z = z.to(self.vae.dtype)
        feat = self.vae.decode(z).sample
        return feat.to(self.dtype)

    def forward_train(self, batch, **kwargs):

        # encode training images
        latents = self.encode_latents(batch['inp'])

        # decode latents
        feats = self.decode_latents(latents)

        # render query pixels
        if self.renderer:
            preds = self.renderer(feats, batch['gt_coord'], batch['gt_cell'])
            batch['pred'] = preds
        else:
            batch['pred'] = feats

        # compute losses
        ret = self.compute_loss(batch, kwargs)

        # compute training psnr
        mse = ((batch['gt'] - batch['pred']) / 2).pow(2).mean(dim=[-2, -1])
        ret['psnr'] = (-10 * torch.log10(mse)).mean().item()

        return ret

    def compute_loss(self, batch, **kwargs):

        ret = {'loss': torch.tensor(0, dtype=torch.float32, device=batch['inp'].device)}

        pred = batch['pred']
        target = batch['target']
        use_gan_loss = kwargs.get('use_gan', False)
        
        # L1 Loss
        l1_loss = torch.abs(pred - target).mean()
        l1_loss_w = self.loss_cfg.get('l1_loss', 1)
        ret['l1_loss'] = l1_loss.item()
        ret['loss'] = ret['loss'] + l1_loss_w * l1_loss
        
        # Perception Loss
        perc_loss = lpips(pred, target).mean()
        perc_loss_w = self.loss_cfg.get('perc_loss', 1)
        ret['perc_loss'] = perc_loss.item()
        ret['loss'] = ret['loss'] + perc_loss_w * perc_loss

        # GAN Loss
        if use_gan_loss:

            if not self.disc_cond_scale:
                logits_fake = self.disc(pred)
            else:
                smap = (batch['gt_cell'][..., 0] / 2 * batch['inp'].shape[-1]).unsqueeze(1)
                logits_fake = self.disc(torch.cat([pred, smap], dim=1))

            gan_g_loss = -torch.mean(logits_fake)
            gan_g_loss_w = self.loss_cfg.get('gan_g_loss', 1)
            ret['gan_g_loss'] = gan_g_loss.item()
            
            if self.training and self.loss_cfg.adaptive_gan_weight:
                nll_loss = l1_loss * l1_loss_w + perc_loss * perc_loss_w
                adaptive_g_w = self.calculate_adaptive_g_w(nll_loss, gan_g_loss, self.renderer.get_last_layer_weight())
                ret['adaptive_g_w'] = adaptive_g_w.item()
                gan_g_loss_w = gan_g_loss_w * adaptive_g_w

            ret['loss'] = ret['loss'] + gan_g_loss * gan_g_loss_w

        return ret