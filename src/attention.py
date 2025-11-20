import torch
import torch.nn as nn
import torch.nn.functional as F

class ELA(nn.Module):
    def __init__(self,  channel, ks, ng):
        super(ELA, self).__init__()

        p0 = ks // 2
        p1 = (ks+2) // 2
        self.conv1 = nn.Conv1d(channel, channel, kernel_size=ks, padding=p0, groups=channel, bias=False)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=ks+2, padding=p1, groups=channel, bias=False)
        
        self.gn = nn.GroupNorm(ng, channel)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Use .mean() method instead of torch.mean for better optimization
        # Use reshape instead of view for more flexibility (view is fine here but reshape is more modern)
        x_h = x.mean(dim=3, keepdim=True).reshape(b, c, h)
        x_w = x.mean(dim=2, keepdim=True).reshape(b, c, w)

        x_h = self.relu(self.gn(self.conv1(x_h)))
        x_w = self.relu(self.gn(self.conv1(x_w)))

        x_h = self.sig(self.conv2(x_h)).reshape(b, c, h, 1)
        x_w = self.sig(self.conv2(x_w)).reshape(b, c, 1, w)

        # Fused multiplication: x * x_h * x_w
        # First multiply x with x_h, then multiply result with x_w
        return x.mul(x_h).mul_(x_w)

class ELA_Tiny(ELA):
    def __init__(self, channel, ks=3):
        super(ELA_Tiny, self).__init__(channel, ks=5, ng=32)

class ELA_Base(ELA):
    def __init__(self, channel, ks=3):
        super(ELA_Base, self).__init__(channel, ks=7, ng=16)

class ELA_Small(ELA):
    def __init__(self, channel, ks=3):
        super(ELA_Small, self).__init__(channel//8, ks=5, ng=16)

class ELA_Large(ELA):
    def __init__(self, channel, ks=3):
        super(ELA_Large, self).__init__(channel//8, ks=7, ng=16)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_att = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # Compute mean and max in parallel - more cache efficient
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)

        # Use torch.cat with out parameter would be ideal but not supported
        combined = torch.cat([avg_map, max_map], dim=1)
        att = torch.sigmoid(self.conv_att(combined))

        # Fused multiply-add: x + x * att = x * (1 + att)
        return x.addcmul(x, att)


class CARA(nn.Module):
    """
    Coordinate Attention-Resource Allocation (CARA) module.
    
    This module refines coarse copy-move areas by weighing the importance of 
    different positions in the matching map using coordinate attention mechanism.
    
    Reference: Hou et al., 2021
    
    Args:
        channels: Number of input channels (C')
        reduction_ratio: Channel compression factor (r). Default: 8
    """
    def __init__(self, channels, reduction_ratio=8):
        super(CARA, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Transform layer components
        # Conv2D compresses channels from C' to C'/r
        self.transform_conv = nn.Conv2d(
            channels, 
            channels // reduction_ratio, 
            kernel_size=1, 
            bias=False
        )
        self.transform_bn = nn.BatchNorm2d(channels // reduction_ratio)
        self.transform_act = nn.ReLU(inplace=True)
        
        # X-direction (height) weight generation
        self.conv_x = nn.Conv2d(
            channels // reduction_ratio, 
            channels, 
            kernel_size=1, 
            bias=False
        )
        
        # Y-direction (width) weight generation
        self.conv_y = nn.Conv2d(
            channels // reduction_ratio, 
            channels, 
            kernel_size=1, 
            bias=False
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass of CARA module.
        
        Args:
            x: Input feature map K with shape [B, C', H, W]
            
        Returns:
            Reweighted feature map W with shape [B, C', H, W]
        """
        b, c, h, w = x.size()
        
        # Step 1: Global Average Pooling along vertical (H) and horizontal (W) axes
        # XI: GAP along width dimension -> [B, C', H, 1]
        # Equation (8): XI_c^i = sum(K_c(i, u')) / W
        XI = torch.mean(x, dim=3, keepdim=True)  # [B, C', H, 1]
        
        # YI: GAP along height dimension -> [B, C', 1, W]
        # Equation (9): YI_c^i' = sum(K_c(i', u)) / H
        YI = torch.mean(x, dim=2, keepdim=True)  # [B, C', 1, W]
        
        # Step 2: Transform layer
        # Concatenate XI and YI along spatial dimension
        # Transpose XI from [B, C', H, 1] to [B, C', 1, H]
        XI_transposed = XI.permute(0, 1, 3, 2)  # [B, C', 1, H]
        
        # Concatenate along width dimension: [B, C', 1, H+W]
        # Equation (10): SI = φ(τ(ω(XI, YI)))
        concat_features = torch.cat([XI_transposed, YI], dim=3)  # [B, C', 1, H+W]
        
        # Apply Conv2D, BN, and activation (compress channels C' -> C'/r)
        SI = self.transform_conv(concat_features)  # [B, C'/r, 1, H+W]
        SI = self.transform_bn(SI)
        SI = self.transform_act(SI)  # [B, C'/r, 1, H+W]
        
        # Step 3: Split layer
        # Split SI into horizontal (H) and vertical (W) components
        SI_x = SI[:, :, :, :h]  # [B, C'/r, 1, H]
        SI_y = SI[:, :, :, h:]  # [B, C'/r, 1, W]
        
        # Reshape SI_x back to [B, C'/r, H, 1] for height-wise attention
        SI_x = SI_x.permute(0, 1, 3, 2)  # [B, C'/r, H, 1]
        
        # Step 4: Generate X and Y weighted coefficients
        # Apply Conv2D to increase channels back to C' and sigmoid activation
        WXI = self.sigmoid(self.conv_x(SI_x))  # [B, C', H, 1]
        WYI = self.sigmoid(self.conv_y(SI_y))  # [B, C', 1, W]
        
        # Step 5: Calculate total weights and reweight feature map
        # Equation (11): W_c(i,j) = K_c(i,j) × WXI_c(i) × WYI_c(j)
        # Total weight: [B, C', H, W]
        output = x * WXI * WYI
        
        return output

