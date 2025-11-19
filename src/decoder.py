"""
Enhanced decoder blocks for dual-stream architecture.

This module contains:
- DualStreamDecoderBlock: Decoder that fuses both cognitive and inspective features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .effnetv2 import MBConv
from .modules import VRSA, CoSA


class DualStreamDecoderBlock(nn.Module):
    """Decoder that fuses both cognitive and inspective features"""
    def __init__(self, in_channels_1, in_channels_2, inspective_channels, use_cosa=True):
        super().__init__()
        self.dconv = nn.ConvTranspose2d(in_channels_2, in_channels_1, 4, padding=1, stride=2)
        
        # Fusion with inspective features
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels_1 + inspective_channels, in_channels_1, 3, padding=1),
            nn.BatchNorm2d(in_channels_1),
            nn.ReLU(inplace=True)
        )
        
        # Use original components
        self.irb = MBConv(in_channels_1 * 2, in_channels_1, stride=1, expand_ratio=6, 
                         use_se=False, use_ela=True)
        
        if use_cosa:
            self.attention = CoSA(in_channels_1, in_channels_1, in_channels_1)
        else:
            self.attention = nn.Sequential(
                VRSA(in_channels_1, in_channels_1),
                nn.ConvTranspose2d(in_channels_1, in_channels_1, 4, padding=2, stride=1),
            )
    
    def forward(self, x1, x2, inspective_feat=None):
        x1_att = self.attention(x1)
        x2_up = self.dconv(x2)
        
        # Match spatial dimensions if needed
        if x1_att.shape[-2:] != x2_up.shape[-2:]:
            x2_up = F.interpolate(
                x2_up,
                size=x1_att.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Fuse with inspective features if available
        if inspective_feat is not None:
            if x1_att.shape[-2:] != inspective_feat.shape[-2:]:
                inspective_feat = F.interpolate(
                    inspective_feat, 
                    size=x1_att.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            x1_att = torch.cat([x1_att, inspective_feat], dim=1)
            x1_att = self.fusion(x1_att)
        
        x = torch.cat([x1_att, x2_up], dim=1)
        x = self.irb(x)
        return x
