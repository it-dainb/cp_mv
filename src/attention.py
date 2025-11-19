import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency import DWTDecomposition, IDWTReconstruction


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


# ==================== Wavelet-Guided Attention ====================
class WaveletGuidedAttention(nn.Module):
    """Interactive Wavelet-guided Self-Attention (IWSA)"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        # Shared global query
        self.to_q = nn.Conv2d(dim, dim, 1)
        
        # Split keys and values across heads
        self.to_k = nn.Conv2d(dim, dim // 2, 1)  # Reduced dimension
        self.to_v = nn.Conv2d(dim, dim, 1)
        
        # Wavelet transforms
        self.dwt = DWTDecomposition()
        self.idwt = IDWTReconstruction()
        
        # Refine frequency features
        self.freq_refine = nn.Conv2d(dim * 4, dim, 1)  # 4 subbands
        
        # Output projection
        self.proj = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x, support_v=None):
        B, C, H, W = x.shape
        
        # Global shared query
        Q = self.to_q(x)  # [B, C, H, W]
        
        # Apply DWT to get frequency components
        ll, lh, hl, hh = self.dwt(x)
        freq_features = torch.cat([ll, lh, hl, hh], dim=1)
        freq_features = self.freq_refine(freq_features)
        
        # Generate K, V from frequency features
        K = self.to_k(freq_features)  # [B, C//2, H//2, W//2]
        V = self.to_v(freq_features)  # [B, C, H//2, W//2]
        
        # Add support from inspective branch
        if support_v is not None:
            V = V + support_v
        
        # Reshape for attention
        Q = Q.view(B, self.num_heads, self.head_dim, H * W)
        K = K.view(B, self.num_heads, self.head_dim // 2, -1)
        V = V.view(B, self.num_heads, self.head_dim, -1)
        
        # Compute attention
        Q = Q.transpose(-2, -1)  # [B, heads, HW, head_dim]
        K = K.transpose(-2, -1)  # [B, heads, h*w, head_dim//2]
        
        # Scaled dot-product attention (Q and K have different dims, pad K)
        K_padded = F.pad(K, (0, self.head_dim - self.head_dim // 2))
        attn = torch.matmul(Q, K_padded.transpose(-2, -1))
        attn = attn / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, V.transpose(-2, -1))  # [B, heads, HW, head_dim]
        out = out.transpose(-2, -1).contiguous()
        out = out.view(B, C, H, W)
        
        # Skip connection with IDWT
        freq_skip = self.idwt(ll, lh, hl, hh)
        freq_skip = F.interpolate(freq_skip, size=(H, W), mode='bilinear', align_corners=False)
        
        out = out + freq_skip
        out = self.proj(out)
        
        return out


class EfficientWaveletTransformerBlock(nn.Module):
    """Efficient Wavelet-guided Transformer Block (EWTB)"""
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WaveletGuidedAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
        )
        
    def forward(self, x, support_v=None):
        B, C, H, W = x.shape
        
        # Self-attention with layer norm
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x_norm, support_v)
        
        # FFN with layer norm
        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.mlp(x_norm)
        
        return x