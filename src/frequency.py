"""
High-frequency feature extraction components for manipulation detection.

This module contains:
- BayarConstrainedConv: Constrained convolution for extracting manipulation artifacts
- DWTDecomposition: Discrete Wavelet Transform for frequency decomposition
- IDWTReconstruction: Inverse Discrete Wavelet Transform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayarConstrainedConv(nn.Module):
    """Constrained convolution for extracting manipulation artifacts"""
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        # Initialize with Bayar constraint
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Central pixel should be -1, surrounding pixels should sum to 1
        weight = torch.zeros(self.conv.weight.shape)
        weight[:, :, 2, 2] = -1.0
        # Distribute +1 among surrounding pixels
        mask = torch.ones_like(weight)
        mask[:, :, 2, 2] = 0
        weight = weight + mask / 24.0  # 25 pixels - 1 center = 24
        self.conv.weight.data = weight
        
    def forward(self, x):
        # Apply constraint during forward pass
        weight = self.conv.weight
        weight_c = weight - weight.mean(dim=[2, 3], keepdim=True)
        weight_c[:, :, 2, 2] = -weight_c.sum(dim=[2, 3])
        return F.conv2d(x, weight_c, padding=2)


class DWTDecomposition(nn.Module):
    """Discrete Wavelet Transform for frequency decomposition"""
    def __init__(self):
        super().__init__()
        # Haar wavelet filters
        self.register_buffer('ll', torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2))
        self.register_buffer('lh', torch.tensor([[0.5, 0.5], [-0.5, -0.5]]).view(1, 1, 2, 2))
        self.register_buffer('hl', torch.tensor([[0.5, -0.5], [0.5, -0.5]]).view(1, 1, 2, 2))
        self.register_buffer('hh', torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2))
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Apply wavelet filters per channel (use reshape instead of view for non-contiguous tensors)
        ll = F.conv2d(x.reshape(B*C, 1, H, W), self.ll, stride=2, padding=0)
        lh = F.conv2d(x.reshape(B*C, 1, H, W), self.lh, stride=2, padding=0)
        hl = F.conv2d(x.reshape(B*C, 1, H, W), self.hl, stride=2, padding=0)
        hh = F.conv2d(x.reshape(B*C, 1, H, W), self.hh, stride=2, padding=0)
        
        # Reshape back
        _, _, h, w = ll.shape
        ll = ll.view(B, C, h, w)
        lh = lh.view(B, C, h, w)
        hl = hl.view(B, C, h, w)
        hh = hh.view(B, C, h, w)
        
        return ll, lh, hl, hh


class IDWTReconstruction(nn.Module):
    """Inverse Discrete Wavelet Transform"""
    def __init__(self):
        super().__init__()
        # Inverse Haar wavelet filters
        self.register_buffer('ll', torch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2) * 0.5)
        self.register_buffer('lh', torch.tensor([[1, 1], [-1, -1]]).view(1, 1, 2, 2) * 0.5)
        self.register_buffer('hl', torch.tensor([[1, -1], [1, -1]]).view(1, 1, 2, 2) * 0.5)
        self.register_buffer('hh', torch.tensor([[1, -1], [-1, 1]]).view(1, 1, 2, 2) * 0.5)
        
    def forward(self, ll, lh, hl, hh):
        B, C, h, w = ll.shape
        H, W = h * 2, w * 2
        
        # Upsample each component (use reshape instead of view for non-contiguous tensors)
        ll_up = F.conv_transpose2d(ll.reshape(B*C, 1, h, w), self.ll, stride=2, padding=0)
        lh_up = F.conv_transpose2d(lh.reshape(B*C, 1, h, w), self.lh, stride=2, padding=0)
        hl_up = F.conv_transpose2d(hl.reshape(B*C, 1, h, w), self.hl, stride=2, padding=0)
        hh_up = F.conv_transpose2d(hh.reshape(B*C, 1, h, w), self.hh, stride=2, padding=0)
        
        # Combine and reshape
        out = (ll_up + lh_up + hl_up + hh_up).view(B, C, H, W)
        return out
