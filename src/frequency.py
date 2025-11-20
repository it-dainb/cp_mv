"""
Frequency domain operations for image manipulation detection.

This module contains frequency-based filtering methods extracted from FreqNet:
- High-frequency filtering in spatial dimensions (height/width)
- High-frequency filtering in channel dimension
- Complex-valued frequency domain convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyFilter(nn.Module):
    """
    Frequency domain filtering operations.
    
    Contains methods for:
    1. High-frequency filtering in spatial (H, W) dimensions
    2. High-frequency filtering in channel (C) dimension
    3. Complex frequency domain convolution layers
    """
    
    def __init__(self):
        super(FrequencyFilter, self).__init__()
    
    @staticmethod
    def hfreq_spatial(x, scale=4):
        """
        High-pass frequency filter in spatial dimensions (Height, Width).
        
        Removes low-frequency components by zeroing out the center of the frequency spectrum.
        This emphasizes edges and high-frequency artifacts which are useful for forgery detection.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            scale: Scale factor for frequency cutoff. Higher values = more aggressive filtering.
                   Center region of size (H/scale, W/scale) will be zeroed.
        
        Returns:
            Filtered tensor of shape (B, C, H, W) with high-frequency components
        
        Example:
            >>> x = torch.randn(2, 3, 256, 256)
            >>> x_filtered = FrequencyFilter.hfreq_spatial(x, scale=4)
            >>> x_filtered.shape
            torch.Size([2, 3, 256, 256])
        """
        assert scale > 2, "Scale must be greater than 2"
        
        # Save original dtype and convert to float32 for FFT
        # cuFFT requires float32 for non-power-of-2 dimensions
        orig_dtype = x.dtype
        x = x.float()
        
        # Apply 2D FFT and shift zero frequency to center
        x_freq = torch.fft.fft2(x, norm="ortho")
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        
        # Zero out low-frequency center region
        b, c, h, w = x_freq.shape
        h_start, h_end = h // 2 - h // scale, h // 2 + h // scale
        w_start, w_end = w // 2 - w // scale, w // 2 + w // scale
        x_freq[:, :, h_start:h_end, w_start:w_end] = 0.0
        
        # Inverse FFT to get back to spatial domain
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_out = torch.fft.ifft2(x_freq, norm="ortho")
        x_out = torch.real(x_out)
        x_out = F.relu(x_out, inplace=True)
        
        # Convert back to original dtype if needed
        if orig_dtype != torch.float32:
            x_out = x_out.to(orig_dtype)
        
        return x_out
    
    @staticmethod
    def hfreq_channel(x, scale=4):
        """
        High-pass frequency filter in channel dimension.
        
        Applies FFT along the channel dimension and removes low-frequency components.
        This can help identify channel-wise inconsistencies in forged images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            scale: Scale factor for frequency cutoff. Higher values = more aggressive filtering.
                   Center region of size (C/scale) will be zeroed.
        
        Returns:
            Filtered tensor of shape (B, C, H, W) with high-frequency channel components
        
        Example:
            >>> x = torch.randn(2, 64, 128, 128)
            >>> x_filtered = FrequencyFilter.hfreq_channel(x, scale=4)
            >>> x_filtered.shape
            torch.Size([2, 64, 128, 128])
        """
        assert scale > 2, "Scale must be greater than 2"
        
        # Save original dtype and convert to float32 for FFT
        orig_dtype = x.dtype
        x = x.float()
        
        # Apply 1D FFT along channel dimension and shift
        x_freq = torch.fft.fft(x, dim=1, norm="ortho")
        x_freq = torch.fft.fftshift(x_freq, dim=1)
        
        # Zero out low-frequency center channels
        b, c, h, w = x_freq.shape
        c_start, c_end = c // 2 - c // scale, c // 2 + c // scale
        x_freq[:, c_start:c_end, :, :] = 0.0
        
        # Inverse FFT to get back to spatial domain
        x_freq = torch.fft.ifftshift(x_freq, dim=1)
        x_out = torch.fft.ifft(x_freq, dim=1, norm="ortho")
        x_out = torch.real(x_out)
        x_out = F.relu(x_out, inplace=True)
        
        # Convert back to original dtype if needed
        if orig_dtype != torch.float32:
            x_out = x_out.to(orig_dtype)
        
        return x_out


class FrequencyConvLayer(nn.Module):
    """
    Complex-valued convolution in frequency domain (FCL - Frequency Convolution Layer).
    
    Performs convolution in the frequency domain by:
    1. Applying 2D FFT to input
    2. Applying separate convolutions to real and imaginary parts
    3. Combining results and applying inverse FFT
    
    This allows the network to learn frequency-domain representations directly.
    """
    
    def __init__(self, channels):
        """
        Args:
            channels: Number of input/output channels
        """
        super(FrequencyConvLayer, self).__init__()
        self.channels = channels
        self.real_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.imag_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
    
    def forward(self, x):
        """
        Apply frequency domain convolution.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W) after frequency domain convolution
        """
        # Save original dtype and convert to float32 for FFT
        orig_dtype = x.dtype
        x = x.float()
        
        # Transform to frequency domain
        x_freq = torch.fft.fft2(x, norm="ortho")
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        
        # Apply separate convolutions to real and imaginary components
        # Extract real and imaginary parts as float32
        real_part = x_freq.real
        imag_part = x_freq.imag
        
        real_out = self.real_conv(real_part)
        imag_out = self.imag_conv(imag_part)
        
        # Ensure conv outputs are float32 before FFT (AMP may convert them to float16)
        real_out = real_out.float()
        imag_out = imag_out.float()
        
        # Combine and transform back to spatial domain
        x_freq_out = torch.complex(real_out, imag_out)
        x_freq_out = torch.fft.ifftshift(x_freq_out, dim=[-2, -1])
        x_out = torch.fft.ifft2(x_freq_out, norm="ortho")
        x_out = torch.real(x_out)
        x_out = F.relu(x_out, inplace=True)
        
        # Convert back to original dtype if needed
        if orig_dtype != torch.float32:
            x_out = x_out.to(orig_dtype)
        
        return x_out


class FrequencyBlock(nn.Module):
    """
    Complete frequency processing block combining multiple operations.
    
    This block implements the HFRI + HFRFC + FCL pattern from FreqNet:
    - HFRI: High-Frequency Region Initialization (spatial filtering)
    - HFRFC: High-Frequency Region in Feature Channel (channel filtering)
    - FCL: Frequency Convolution Layer (complex convolution)
    """
    
    def __init__(self, in_channels, out_channels, stride=1, scale=4, use_channel_filter=True):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the convolution (1 or 2)
            scale: Scale factor for frequency filtering
            use_channel_filter: Whether to apply channel-wise frequency filtering
        """
        super(FrequencyBlock, self).__init__()
        self.scale = scale
        self.stride = stride
        self.use_channel_filter = use_channel_filter
        
        # Learnable spatial convolution in frequency domain
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Frequency convolution layer
        self.freq_conv = FrequencyConvLayer(out_channels)
    
    def forward(self, x):
        """
        Apply frequency block operations.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
        
        Returns:
            Output tensor of shape (B, C_out, H/stride, W/stride)
        """
        # HFRI: High-frequency filtering in spatial domain
        x = FrequencyFilter.hfreq_spatial(x, scale=self.scale)
        
        # Standard convolution
        x = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=0)
        x = F.relu(x, inplace=True)
        
        # HFRFC: Optional channel-wise frequency filtering
        if self.use_channel_filter:
            x = FrequencyFilter.hfreq_channel(x, scale=self.scale)
        
        # FCL: Frequency domain convolution
        x = self.freq_conv(x)
        
        return x


# Helper functions for easy usage
def apply_hfreq_spatial(x, scale=4):
    """
    Apply high-frequency spatial filtering.
    
    Args:
        x: Input tensor (B, C, H, W)
        scale: Frequency cutoff scale
    
    Returns:
        Filtered tensor
    """
    return FrequencyFilter.hfreq_spatial(x, scale=scale)


def apply_hfreq_channel(x, scale=4):
    """
    Apply high-frequency channel filtering.
    
    Args:
        x: Input tensor (B, C, H, W)
        scale: Frequency cutoff scale
    
    Returns:
        Filtered tensor
    """
    return FrequencyFilter.hfreq_channel(x, scale=scale)
