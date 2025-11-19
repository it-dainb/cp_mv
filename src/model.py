import torch
import torch.nn as nn

from .effnetv2 import effnetv2_s, MBConv
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

class CMSegNet(nn.Module):
    def __init__(self):
        super(CMSegNet, self).__init__()
        
        self.backbone = effnetv2_s()
        '''
        Checkpoints feature map shapes:
        {6: (24, 48), 10: (48, 64), 25: (64, 160), 41: (160, 1792)}
        '''

        decoders = []
        for idx, (n_layer, (in_c, out_c)) in enumerate(self.backbone.ckpt_layers.items()):
            decoders.append(DecoderBlock(in_c, out_c, use_cosa=idx != 0))

            if idx == 0:
                self.dconv = nn.ConvTranspose2d(out_c, in_c, 4, stride=4, padding=0)
                self.conv_last = nn.Conv2d(in_c, 3, 1)
            
        self.decoders = nn.ModuleList(
            decoders[::-1]  # Reverse the list to match the order of feature maps
        )

        self.conv_score = nn.Conv2d(3, 1, 1)
        
        # Fuse final convolutions for better performance
        self.final_conv = nn.Sequential(
            self.dconv,
            self.conv_last,
            self.conv_score
        )

    def forward(self, x):
        # Extract features with checkpoints
        features_ckpts = self.backbone(x, return_ckpt=True)

        # Initial decoder pass
        x = self.decoders[0](features_ckpts[-2], features_ckpts[-1])

        # Sequential decoder passes - iterate efficiently
        for idx in range(1, len(self.decoders)):
            feat_idx = -3 - (idx - 1)
            x = self.decoders[idx](features_ckpts[feat_idx], x)

        # Apply fused final convolutions
        x = self.final_conv(x)
        
        return x