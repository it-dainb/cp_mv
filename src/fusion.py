"""
Bidirectional fusion module for combining cognitive and inspective branches.

This module contains:
- BidirectionalFusion: Cross-branch fusion between cognitive (RGB) and inspective (Noise) streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalFusion(nn.Module):
    """Bidirectional fusion between cognitive and inspective branches"""
    def __init__(self, cognitive_dim, inspective_dim):
        super().__init__()
        self.cognitive_dim = cognitive_dim
        
        # Inspective -> Cognitive (provide detail guidance)
        self.insp_to_cog = nn.Sequential(
            nn.Conv2d(inspective_dim, cognitive_dim, 1),
            nn.BatchNorm2d(cognitive_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fusion (outputs cognitive_dim to maintain compatibility with decoder)
        self.fusion = nn.Sequential(
            nn.Conv2d(cognitive_dim * 2, cognitive_dim, 3, padding=1),
            nn.BatchNorm2d(cognitive_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, cognitive_feat, inspective_feat):
        # Upsample inspective features to match cognitive resolution
        if cognitive_feat.shape[-2:] != inspective_feat.shape[-2:]:
            inspective_feat = F.interpolate(
                inspective_feat, 
                size=cognitive_feat.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Transform and fuse
        insp_adapted = self.insp_to_cog(inspective_feat)
        fused = torch.cat([cognitive_feat, insp_adapted], dim=1)
        fused = self.fusion(fused)
        
        return fused
