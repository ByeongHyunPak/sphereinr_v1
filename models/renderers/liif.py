import copy

import torch
import torch.nn as nn

import models
from models import register
from utils.geometry_liif import convert_liif_feat_coord_cell, convert_posenc


@register('liif')
class LIIF(nn.Module):

    def __init__(self, net, z_dec_channels, co_pe_dim=None, co_pe_w_max=None, ce_pe_dim=None, ce_pe_w_max=None, x_channels=None):
        super().__init__()
        self.x_channels = x_channels
        coord_dim = 0

        self.co_pe_dim = co_pe_dim
        self.co_pe_w_max = co_pe_w_max
        coord_dim += 2 if co_pe_dim is None else 2 * co_pe_dim

        self.ce_pe_dim = ce_pe_dim
        self.ce_pe_w_max = ce_pe_w_max
        coord_dim += 2 if ce_pe_dim is None else 2 * ce_pe_dim

        net_spec = copy.copy(net)
        net_spec['args']['in_channels'] = (x_channels if x_channels is not None else 0) + z_dec_channels + coord_dim
        self.net = models.make(net_spec)

    def get_last_layer_weight(self):
        return self.net.get_last_layer_weight()

    def forward(self, z_dec, coord, cell):
        q_feat, rel_coord, rel_cell = convert_liif_feat_coord_cell(z_dec, coord, cell)
        if self.co_pe_dim is not None:
            rel_coord = convert_posenc(rel_coord, self.co_pe_dim, self.co_pe_w_max)
        if self.ce_pe_dim is not None:
            rel_cell = convert_posenc(rel_cell, self.ce_pe_dim, self.ce_pe_w_max)
        # print(q_feat.shape, rel_coord.shape, rel_cell.shape) 
        # torch.Size([1, 64, 64, 128]) torch.Size([1, 64, 64, 2]) torch.Size([1, 64, 64, 2])
        layout = torch.cat([q_feat, rel_coord, rel_cell], dim=-1).permute(0, 3, 1, 2)
        return self.net(layout)


@register('simple_renderer_net')
class SimpleRendererNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=3, kernel_size=3, normalize=True, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=(kernel_size - 1) // 2),
                                    ResnetBlock(in_channels=hidden_channels,
                                                out_channels=hidden_channels,
                                                kernel_size=kernel_size,
                                                temb_channels=0, dropout=0.0, normalize=normalize),
                                    ResnetBlock(in_channels=hidden_channels,
                                                out_channels=hidden_channels,
                                                kernel_size=kernel_size,
                                                temb_channels=0, dropout=0.0, normalize=normalize)])
        self.norm_out = Normalize(hidden_channels) if normalize else torch.nn.Identity()
        self.conv_out = torch.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def get_last_layer_weight(self):
        return self.conv_out.weight

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1, 2]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x
    
def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, kernel_size=3, conv_shortcut=False,
                 dropout, temb_channels=512, normalize=True):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels) if normalize else torch.nn.Identity()
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=(kernel_size - 1) // 2)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels) if normalize else torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=(kernel_size - 1) // 2)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=kernel_size,
                                                     stride=1,
                                                     padding=(kernel_size - 1) // 2)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h