import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import CLIPTextModel, CLIPTokenizer
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
            vae,
            disc=None,
            renderer=None,
            diffuser=None,
            loss_cfg = dict(),
            **kwargs
        ):

        self.vae = CustomAutoencoderKL.from_pretrained(vae)

        input_nc = 3 if not disc.disc_cond_scale else 4
        self.disc_cond_scale = disc.disc_cond_scale
        self.disc = make_discriminator(**disc, input_nc=input_nc) if disc else None

        self.renderer = models.make(renderer) if renderer else None

        if diffuser is not None: # TODO
            self.text_encoder = CLIPTextModel.from_pretrained(diffuser.text_encoder)
            self.tokenizer = CLIPTokenizer.from_pretrained(diffuser.tokenizer)
            self.scheduler = DDIMScheduler.from_pretrained(diffuser.scheduler)
            self.unet = UNet2DConditionModel.from_pretrained(diffuser.unet)
            if diffuser.get('lora', False):
                self.add_lora(self.unet)

        self.loss_cfg = loss_cfg
    
    def add_lora(self, unet):
        pass # TODO

    def get_gd_from_opt(self, opt):
        if opt is None:
            opt = dict()
        gd = dict()
        gd['vae.encoder'] = opt.get('vae.encoder', False)
        gd['vae.decoder'] = opt.get('vae.decoder', False)
        gd['renderer'] = opt.get('renderer', False)
        gd['text_encoder'] = opt.get('text_encoder', False)
        gd['unet'] = opt.get('unet', False)
        return gd

    def get_params(self, name):
        if name == 'vae':
            return self.vae.parameters()
        elif name == 'renderer':
            return self.renderer.parameters()
        elif name == "text_encoder":
            return self.text_encoder.parameters()
        elif name == "unet":
            return self.unet.parameters()
        elif name == "disc":
            return self.disc.parameters()
        else:
            raise NotImplementedError()

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

    def rotate_latents(self, z, degree=None):
        if degree is None:
            degree = self.rot_diff
        if degree % 360 == 0:
            return z
        return torch.roll(z, degree // 360 * z.shape[-1], dims=-1)
    
    def forward_autoencoder(self, batch):
        latents = self.encode_latents(batch['inp'])
        feats = self.decode_latents(latents)
        return feats

    def forward(self, batch, mode, **kwargs):

        if mode == "pred":
            with torch.no_grad():
                feats = self.forward_autoencoder(batch)
                preds = self.renderer(feats, batch['gt_coord'], batch['gt_cell'])
                return preds

        elif mode == "loss":
            feats = self.forward_autoencoder(batch)
            batch['pred'] = self.renderer(feats, batch['gt_coord'], batch['gt_cell'])
            ret = self.compute_loss(batch, mode="loss", **kwargs)
            # compute training psnr
            mse = ((batch['gt'] - batch['pred']) / 2).pow(2).mean(dim=[-2, -1])
            ret['psnr'] = (-10 * torch.log10(mse)).mean().item()

        elif mode == "disc_loss":
            with torch.no_grad():
                feats = self.forward_autoencoder(batch)
                batch['pred'] = self.renderer(feats, batch['gt_coord'], batch['gt_cell'])
            ret = self.compute_loss(batch, mode="disc_loss", **kwargs)

        return ret

    def compute_loss(self, batch, mode, **kwargs):

        if mode == "loss":
            ret = {'loss': torch.tensor(0, dtype=torch.float32, device=batch['inp'].device)}
            
            # L1 Loss
            l1_loss = torch.abs(batch['pred'] - batch['target']).mean()
            l1_loss_w = self.loss_cfg.get('l1_loss', 1)
            ret['l1_loss'] = l1_loss.item()
            ret['loss'] = ret['loss'] + l1_loss_w * l1_loss
            
            # Perception Loss
            perc_loss = lpips(batch['pred'], batch['target']).mean()
            perc_loss_w = self.loss_cfg.get('perc_loss', 1)
            ret['perc_loss'] = perc_loss.item()
            ret['loss'] = ret['loss'] + perc_loss_w * perc_loss

            # GAN Loss
            if kwargs.get('use_gan', False):
                if not self.disc_cond_scale:
                    logits_fake = self.disc(batch['pred'])
                else:
                    smap = (batch['gt_cell'][..., 0] / 2 * batch['inp'].shape[-1]).unsqueeze(1)
                    logits_fake = self.disc(torch.cat([batch['pred'], smap], dim=1))

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
        
        elif mode == "disc_loss":
            if not self.disc_cond_scale:
                logits_real = self.disc(batch['gt'])
                logits_fake = self.disc(batch['pred'])
            else:
                smap = (batch['gt_cell'][..., 0] / 2 * batch['inp'].shape[-1]).unsqueeze(1)
                logits_real = self.disc(torch.cat([batch['gt'], smap], dim=1))
                logits_fake = self.disc(torch.cat([batch['pred'], smap], dim=1))

            disc_loss_type = self.loss_cfg.get('disc_loss_type', 'hinge')

            if disc_loss_type == 'hinge':
                loss_real = torch.mean(F.relu(1. - logits_real))
                loss_fake = torch.mean(F.relu(1. + logits_fake))
                loss = (loss_real + loss_fake) / 2
            elif disc_loss_type == 'vanilla':
                loss_real = torch.mean(F.softplus(-logits_real))
                loss_fake = torch.mean(F.softplus(logits_fake))
                loss = (loss_real + loss_fake) / 2

            return {
                'loss': loss,
                'disc_logits_real': logits_real.mean().item(),
                'disc_logits_fake': logits_fake.mean().item(),
            }
    
    def calculate_adaptive_g_w(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size > 1:
            dist.all_reduce(nll_grads, op=dist.ReduceOp.SUM)
            nll_grads.div_(world_size)
            dist.all_reduce(g_grads, op=dist.ReduceOp.SUM)
            g_grads.div_(world_size)
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
    

from typing import Dict, Optional, Tuple, Union
from diffusers.utils import is_torch_version
from diffusers.models.autoencoders.vae import DecoderOutput

class CustomAutoencoderKL(AutoencoderKL):

    def from_pretrained(self, key):
        vae = super().from_pretrained(key)
        self.decoder = vae.decoder
        return self
    
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        # custom decode logic, bypassing post-process in the original `_decode` method
        if self.use_slicing and z.shape[0] > 1:
            # Using custom decoder logic on slices
            decoded_slices = [self.custom_decoder_forward(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            # Direct call to the custom decoder forward function
            decoded = self.custom_decoder_forward(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
    
    def custom_decoder_forward(self, sample: torch.Tensor, latent_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        sample = self.decoder.conv_in(sample)
        upscale_dtype = next(iter(self.decoder.up_blocks.parameters())).dtype

        if torch.is_grad_enabled() and self.decoder.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.decoder.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                for up_block in self.decoder.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.decoder.mid_block), sample, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                for up_block in self.decoder.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            sample = self.decoder.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            for up_block in self.decoder.up_blocks:
                sample = up_block(sample, latent_embeds)

        return sample  # post-process removed here
    
    
    

