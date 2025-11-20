import torch
import torch.nn as nn

from .decoder import DecoderBlock
from.effnetv2 import effnetv2_s

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
        # Only iterate while we have features remaining (features_ckpts length - 2 already used)
        for idx in range(1, min(len(self.decoders), len(features_ckpts) - 1)):
            feat_idx = -3 - (idx - 1)
            x = self.decoders[idx](features_ckpts[feat_idx], x)

        # Apply fused final convolutions
        x = self.final_conv(x)
        
        return x