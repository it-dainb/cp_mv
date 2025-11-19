"""
Inspective Branch (Noise Stream) for fine-grained manipulation trace extraction.

This module contains:
- InspectiveBlock: Lightweight convolution block for trace extraction
- InspectiveBranch: Full inspective branch for local manipulation detection
"""

import torch
import torch.nn as nn

from .frequency import BayarConstrainedConv


class InspectiveBlock(nn.Module):
    """Lightweight convolution block for fine-grained trace extraction"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class InspectiveBranch(nn.Module):
    """Inspective branch for local fine-grained manipulation traces"""
    def __init__(self, channels=[64, 128, 128]):
        super().__init__()
        
        # High-frequency feature extraction
        self.noise_extract = BayarConstrainedConv(3, channels[0])
        
        # Multi-scale inspective blocks
        self.blocks = nn.ModuleList([
            InspectiveBlock(channels[0], channels[0]),
            InspectiveBlock(channels[0], channels[1], stride=2),
            InspectiveBlock(channels[1], channels[2]),
        ])
        
        # Generate support features for cognitive branch
        self.support_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 1),
                nn.AdaptiveAvgPool2d(1)
            ) for c in channels
        ])
        
    def forward(self, x):
        # Extract high-frequency noise patterns
        x = self.noise_extract(x)
        
        features = []
        support_features = []
        
        for idx, (block, support_conv) in enumerate(zip(self.blocks, self.support_convs)):
            x = block(x)
            features.append(x)
            # Generate global support features
            support = support_conv(x)
            support_features.append(support)
            
        return features, support_features
