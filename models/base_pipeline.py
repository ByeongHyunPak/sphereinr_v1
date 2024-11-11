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
from models.vqgan.discriminator import make_discriminator


@register('base_pipeline')
class basePipeline(nn.Module):

    def __init__(
            self, 
            loss_cfg = None,
            disc=True, 
            disc_cond_scale=False, 
            disc_use_custom=False, 
            adaptive_gan_weight=False, 
            **kwargs
        ):

        self.loss_cfg = loss_cfg

        self.vae = ...
        self.unet = ...
        self.tokenizer = ...
        self.text_encoder = ...
        self.scheduler = ...
        self.renderer = models.make(...)


        input_nc = 3 if not disc_cond_scale else 4
        self.disc = make_discriminator(use_custom=disc_use_custom, input_nc=input_nc) if disc else None
        self.disc_cond_scale = disc_cond_scale
        self.adaptive_gan_weight = adaptive_gan_weight
        

    def add_lora(self):
        pass

    def get_params(self, name):
        pass

    def encode_text(self, text):

        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device), attention_mask=attention_mask)
        
        return prompt_embeds[0].to(self.dtype)

    def encode_latent(self, x):
        x = x.to(self.vae.dtype)
        z = self.vae.encode(x).latent_dist.sample()
        z = z * self.vae.config.scaling_factor
        return z.to(self.dtype)
    
    def rotate_latent(self, z, degree=None):
        if degree is None:
            degree = self.rot_diff
        if degree % 360 == 0:
            return z
        return torch.roll(z, degree // 360 * z.shape[-1], dims=-1)

    def decode_latent(self, z):
        z = z / self.vae.config.scaling_factor
        image = self.vae.decode(z.to(self.vae.dtype)).sample
        return image.to(self.dtype)
    
    def forward_train(self, batch, **kwargs):

        # TODO encode traing images
        latents = self.encode_latent(batch['inp'])
        b, c, h, w = latents.shape

        # TODO add_noise

        # TODO unet denoise
        # -- TODO if compute_snr_loss is True

        ret = ...

        # TODO decode latents

        # render w/ sphere-inr
        pred = self.renderer(latents, batch['gt_coord'], batch['gt_cell'])
        batch['pred'] = pred

        # compute losses
        ret = self.compute_loss(batch, ret, kwargs)

        mse = ((batch['gt'] - pred) / 2).pow(2).mean(dim=[1, 2, 3])
        ret['psnr'] = (-10 * torch.log10(mse)).mean().item()

        return ret

    def compute_loss(self, batch, ret, **kwargs):

        pred = batch['pred']
        target = batch['target']
        use_gan_loss = kwargs.get('use_gan', False)
        loss_cfg = self.loss_cfg
        
        # L1 Loss
        l1_loss = torch.abs(pred - target).mean()
        l1_loss_w = loss_cfg.get('l1_loss', 1)
        ret['l1_loss'] = l1_loss.item()
        ret['loss'] = ret['loss'] + l1_loss_w * l1_loss
        
        # Perception Loss
        perc_loss = lpips(pred, target).mean()
        perc_loss_w = loss_cfg.get('perc_loss', 1)
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
            
            if self.training and self.adaptive_gan_weight:
                nll_loss = l1_loss * l1_loss_w + perc_loss * perc_loss_w
                adaptive_g_w = self.calculate_adaptive_g_w(nll_loss, gan_g_loss, self.renderer.get_last_layer_weight())
                ret['adaptive_g_w'] = adaptive_g_w.item()
                gan_g_loss_w = gan_g_loss_w * adaptive_g_w

            ret['loss'] = ret['loss'] + gan_g_loss * gan_g_loss_w

        return ret