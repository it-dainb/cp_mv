from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SpatialAttention, CARA

class ZeroWindow:
    def __init__(self):
        self.store = {}

    def __call__(self, inp_tensor, height, width, scale_ratio=0.1):
        sigma_h = height * scale_ratio
        sigma_w = width * scale_ratio

        batch, channels, h_grid, w_grid = inp_tensor.shape
        cache_key = str(inp_tensor.shape) + str(scale_ratio)

        if cache_key not in self.store:
            device = inp_tensor.device
            dtype = inp_tensor.dtype
            
            # Create indices directly on device with proper dtype
            row_index = torch.arange(h_grid, device=device, dtype=dtype).view(1, 1, h_grid, 1)
            col_index = torch.arange(w_grid, device=device, dtype=dtype).view(1, 1, 1, w_grid)

            # Replace np.indices with torch.meshgrid - more efficient
            center_row_idx = torch.arange(height, device=device, dtype=dtype)
            center_col_idx = torch.arange(width, device=device, dtype=dtype)
            center_row, center_col = torch.meshgrid(center_row_idx, center_col_idx, indexing='ij')
            
            # Reshape to match channel dimension: (1, height*width, 1, 1)
            center_row = center_row.reshape(1, -1, 1, 1)
            center_col = center_col.reshape(1, -1, 1, 1)

            # Fused gaussian computation using torch operations
            # Precompute constants
            inv_2sigma_h_sq = 1.0 / (2 * sigma_h ** 2)
            inv_2sigma_w_sq = 1.0 / (2 * sigma_w ** 2)
            
            # Compute gaussian in fused operations
            g_row = torch.exp(-((row_index - center_row) ** 2) * inv_2sigma_h_sq)
            g_col = torch.exp(-((col_index - center_col) ** 2) * inv_2sigma_w_sq)

            # Fused multiply and subtract
            gauss_mask = torch.ones_like(inp_tensor).sub_(g_row.mul(g_col))
            self.store[cache_key] = gauss_mask
        else:
            gauss_mask = self.store[cache_key]

        # In-place multiplication for memory efficiency
        return inp_tensor.mul(gauss_mask)


def get_topk(tensor_in, k=10, dim=-3):
    values, _ = torch.topk(tensor_in, k=k, dim=dim)
    return values


class Cor(nn.Module):
    def __init__(self, topk=3):
        super().__init__()
        self.topk = topk
        self.zero_window = ZeroWindow()
        self.alpha = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))

    def forward(self, x):
        batch, channels, h_spatial, w_spatial = x.shape

        # Use F.normalize which is optimized
        x_norm = F.normalize(x, p=2, dim=1)

        # Fused reshape and matmul operations
        corr_matrix = torch.bmm(
            x_norm.permute(0, 2, 3, 1).reshape(batch, h_spatial * w_spatial, channels),
            x_norm.reshape(batch, channels, h_spatial * w_spatial)
        )

        corr_map = corr_matrix.view(batch, h_spatial * w_spatial, h_spatial, w_spatial)
        corr_map = self.zero_window(corr_map, h_spatial, w_spatial, scale_ratio=0.05)
        corr_map = corr_map.reshape(batch, h_spatial * w_spatial, h_spatial * w_spatial)

        # Compute scaled map once and reuse
        scaled_map = corr_map.mul(self.alpha)
        
        # Use mul instead of * for potentially better fusion
        att = F.softmax(scaled_map, dim=-1).mul(F.softmax(scaled_map, dim=-2))
        att = att.reshape(batch, h_spatial, w_spatial, h_spatial, w_spatial)

        att_flat = att.reshape(batch, h_spatial * w_spatial, h_spatial, w_spatial)
        top_vals = get_topk(att_flat, k=self.topk, dim=-3)

        return top_vals


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation_value):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation_value, dilation=dilation_value, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        target_size = x.shape[-2:]
        out = x
        for mod in self:
            out = mod(out)
        return F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates: Sequence[int], out_channels=256):
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        )

        for rate in tuple(atrous_rates):
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        concat = torch.cat(outputs, dim=1)
        return self.project(concat)
    
class VRSA(nn.Module):
    def __init__(self, in_channels=16, out_channels=16, atrous_rates=[4, 8, 12, 16], attention_type='sa'):
        """
        VRSA module with configurable attention mechanism.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            atrous_rates: Dilation rates for ASPP
            attention_type: Type of attention ('sa' for SpatialAttention, 'cara' for CARA)
        """
        super().__init__()
        self.aspp = ASPP(in_channels, atrous_rates=atrous_rates, out_channels=out_channels)
        self.attention_type = attention_type
        
        # Select attention mechanism based on attention_type
        if attention_type == 'cara':
            self.attention = CARA(channels=out_channels, reduction_ratio=8)
        elif attention_type == 'sa':
            self.attention = SpatialAttention()
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}. Choose 'sa' or 'cara'.")

    def forward(self, x):
        x = self.aspp(x)
        x = self.attention(x)
        return x

class CoSA(nn.Module):
    def __init__(self, topk=16, in_channels=16, out_channels=16, atrous_rates=[4, 8, 12, 16], attention_type='sa'):
        """
        CoSA module combining correlation and VRSA with configurable attention.
        
        Args:
            topk: Number of top-k values to keep in correlation
            in_channels: Number of input channels
            out_channels: Number of output channels
            atrous_rates: Dilation rates for ASPP
            attention_type: Type of attention ('sa' or 'cara')
        """
        super().__init__()
        self.corr = Cor(topk=topk)
        self.vrsa = VRSA(in_channels=in_channels, out_channels=out_channels, 
                        atrous_rates=atrous_rates, attention_type=attention_type)

    def forward(self, x):
        x = self.corr(x)        
        x = self.vrsa(x)
        
        return x