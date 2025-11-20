"""
Enhanced decoder blocks for dual-stream architecture.

This module contains:
- DualStreamDecoderBlock: Decoder that fuses both cognitive and inspective features
"""

import torch
import torch.nn as nn

from .effnetv2 import MBConv
from .modules import VRSA, CoSA
class DecoderBlock(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, use_cosa=True):
        super(DecoderBlock, self).__init__()

        self.dconv = nn.ConvTranspose2d(in_channels_2, in_channels_1, 4, padding=1, stride=2)
        self.irb = MBConv(in_channels_1 * 2, in_channels_1, stride=1, expand_ratio=6, use_se=False, use_ela=True)

        if use_cosa:
            self.attention = CoSA(in_channels_1, in_channels_1, in_channels_1)
        else:
            self.attention = nn.Sequential(
                VRSA(in_channels_1, in_channels_1),
                nn.ConvTranspose2d(in_channels_1, in_channels_1, 4, padding=2, stride=1),
            )

    def forward(self, x1, x2):
        # Process attention and upsampling in parallel (can be optimized by compiler)
        x1_att = self.attention(x1)
        x2_up = self.dconv(x2)
        
        # Fused concatenation - torch.cat is already optimized
        x = torch.cat([x1_att, x2_up], dim=1)
        
        x = self.irb(x)
        return x
