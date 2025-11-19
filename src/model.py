import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .effnetv2 import effnetv2_s, MBConv
from .modules import VRSA, CoSA
from .frequency import BayarConstrainedConv, DWTDecomposition, IDWTReconstruction
from .inspective import InspectiveBranch
from .attention import WaveletGuidedAttention, EfficientWaveletTransformerBlock
from .fusion import BidirectionalFusion
from .decoder import DualStreamDecoderBlock

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
        # Only iterate while we have features remaining (features_ckpts length - 2 already used)
        for idx in range(1, min(len(self.decoders), len(features_ckpts) - 1)):
            feat_idx = -3 - (idx - 1)
            x = self.decoders[idx](features_ckpts[feat_idx], x)

        # Apply fused final convolutions
        x = self.final_conv(x)
        
        return x


# ==================== CMSegNetV2 (Dual-Stream) ====================
class CMSegNetV2(nn.Module):
    """Enhanced CMSegNet with FastForensics-inspired dual-stream architecture"""
    def __init__(self):
        super().__init__()
        
        # Cognitive Branch (RGB stream) - original backbone
        self.backbone = effnetv2_s()
        
        # Inspective Branch (Noise stream)
        self.inspective_branch = InspectiveBranch(channels=[64, 128, 128])
        
        # Wavelet-guided Transformer blocks for cognitive branch
        # Match actual backbone feature dimensions: [48, 64, 160]
        self.ewtb_blocks = nn.ModuleList([
            EfficientWaveletTransformerBlock(dim=48, num_heads=4),
            EfficientWaveletTransformerBlock(dim=64, num_heads=4),
            EfficientWaveletTransformerBlock(dim=160, num_heads=4),
        ])
        
        # Bidirectional fusion modules
        self.fusions = nn.ModuleList([
            BidirectionalFusion(48, 64),
            BidirectionalFusion(64, 128),
            BidirectionalFusion(160, 128),
        ])
        
        # Enhanced decoders with dual-stream fusion
        decoders = []
        # Inspective branch outputs: [64, 128, 128] channels (indices 0, 1, 2)
        # After decoder reversal: decoders[0] gets no insp, decoders[1] gets insp[-1]=128, decoders[2] gets insp[-2]=128, decoders[3] gets insp[-3]=64
        inspective_channels_forward = [64, 128, 128, 128]  # Channels for each decoder in forward order (before reversal)
        
        for idx, (n_layer, (in_c, out_c)) in enumerate(self.backbone.ckpt_layers.items()):
            insp_c = inspective_channels_forward[min(idx, len(inspective_channels_forward)-1)]
            decoders.append(
                DualStreamDecoderBlock(in_c, out_c, insp_c, use_cosa=idx != 0)
            )
            
            if idx == 0:
                self.dconv = nn.ConvTranspose2d(out_c, in_c, 4, stride=4, padding=0)
                self.conv_last = nn.Conv2d(in_c, 3, 1)
        
        self.decoders = nn.ModuleList(decoders[::-1])
        self.conv_score = nn.Conv2d(3, 1, 1)
        
        # Final convolution sequence
        self.final_conv = nn.Sequential(
            self.dconv,
            self.conv_last,
            self.conv_score
        )
    
    def forward(self, x):
        # Extract features from cognitive branch (RGB)
        cognitive_features = self.backbone(x, return_ckpt=True)
        
        # Extract features from inspective branch (Noise)
        inspective_features, support_values = self.inspective_branch(x)
        
        # Apply EWTB to cognitive features with inspective support
        enhanced_cognitive = []
        for idx, (cog_feat, ewtb, fusion) in enumerate(
            zip(cognitive_features[:-1], self.ewtb_blocks, self.fusions)
        ):
            # Match support value dimensions
            support_v = support_values[min(idx, len(support_values)-1)]
            if support_v.shape[-2:] != (1, 1):
                support_v = F.adaptive_avg_pool2d(support_v, 1)
            
            # Expand support_v to match spatial dimensions after DWT
            B, C, H, W = cog_feat.shape
            # Broadcast to correct channels and expand spatially
            if support_v.shape[1] != C:
                # Use repeat instead of expand for channel dimension
                repeat_factor = (C + support_v.shape[1] - 1) // support_v.shape[1]  # Ceiling division
                support_v = support_v.repeat(1, repeat_factor, 1, 1)[:, :C, :, :]
            support_v = support_v.expand(B, C, H // 2, W // 2)
            
            # Apply EWTB with support
            cog_enhanced = ewtb(cog_feat, support_v)
            
            # Bidirectional fusion with inspective features
            insp_feat = inspective_features[min(idx, len(inspective_features)-1)]
            cog_enhanced = fusion(cog_enhanced, insp_feat)
            
            enhanced_cognitive.append(cog_enhanced)
        
        # Add the last feature without EWTB
        enhanced_cognitive.append(cognitive_features[-1])
        
        # Decoder with dual-stream fusion
        x = self.decoders[0](enhanced_cognitive[-2], enhanced_cognitive[-1])
        
        for idx in range(1, min(len(self.decoders), len(enhanced_cognitive) - 1)):
            feat_idx = -3 - (idx - 1)
            # Map inspective features: decoder[1]->insp[-1], decoder[2]->insp[-2], decoder[3]->insp[-3]
            insp_idx = -idx
            insp_feat = inspective_features[insp_idx] if abs(insp_idx) <= len(inspective_features) else None
            x = self.decoders[idx](enhanced_cognitive[feat_idx], x, insp_feat)
        
        # Final prediction
        x = self.final_conv(x)
        
        return x


class CMSegNetV2WithAux(CMSegNetV2):
    """
    Enhanced CMSegNetV2 with auxiliary task heads for multi-task learning.
    
    Adds:
    - Boundary detection head: Predicts manipulation boundaries
    - Position offset head: Predicts offset vectors to region centers
    
    Compatible with LossV3 full multi-task mode.
    """
    def __init__(self, aux_from_decoder_idx=1):
        """
        Args:
            aux_from_decoder_idx: Which decoder stage to extract features from for aux heads
                                  (0=deepest/smallest, 3=shallowest/largest)
                                  Default: 1 (good balance between semantics and resolution)
        """
        super().__init__()
        
        # Determine feature channels based on decoder index
        # Decoder channels (after reversal): [1792->160, 160->64, 64->48, 48->24]
        # Actual output channels: [160, 64, 48, 24]
        decoder_out_channels = [160, 64, 48, 24]
        aux_channels = decoder_out_channels[aux_from_decoder_idx]
        
        self.aux_from_decoder_idx = aux_from_decoder_idx
        
        # Boundary detection head
        # Predicts boundary probability map [B, 1, H, W]
        self.boundary_head = nn.Sequential(
            nn.Conv2d(aux_channels, aux_channels // 2, 3, padding=1),
            nn.BatchNorm2d(aux_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(aux_channels // 2, aux_channels // 4, 3, padding=1),
            nn.BatchNorm2d(aux_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(aux_channels // 4, 1, 1)
        )
        
        # Position offset head
        # Predicts offset vectors (dy, dx) [B, 2, H, W]
        self.offset_head = nn.Sequential(
            nn.Conv2d(aux_channels, aux_channels // 2, 3, padding=1),
            nn.BatchNorm2d(aux_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(aux_channels // 2, aux_channels // 4, 3, padding=1),
            nn.BatchNorm2d(aux_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(aux_channels // 4, 2, 1)
        )
    
    def forward(self, x, return_aux=True):
        """
        Forward pass with optional auxiliary outputs.
        
        Args:
            x: Input image [B, 3, H, W]
            return_aux: If True, returns dict with main + auxiliary outputs
                       If False, returns only main output (same as CMSegNetV2)
        
        Returns:
            If return_aux=True:
                dict with keys:
                    'main': [B, 1, H, W] - manipulation mask
                    'boundary': [B, 1, H, W] - boundary map
                    'offset': [B, 2, H, W] - position offsets (dy, dx)
            If return_aux=False:
                [B, 1, H, W] - manipulation mask only
        """
        target_size = x.shape[-2:]  # Original input size
        
        # Extract features from cognitive branch (RGB)
        cognitive_features = self.backbone(x, return_ckpt=True)
        
        # Extract features from inspective branch (Noise)
        inspective_features, support_values = self.inspective_branch(x)
        
        # Apply EWTB to cognitive features with inspective support
        enhanced_cognitive = []
        for idx, (cog_feat, ewtb, fusion) in enumerate(
            zip(cognitive_features[:-1], self.ewtb_blocks, self.fusions)
        ):
            # Match support value dimensions
            support_v = support_values[min(idx, len(support_values)-1)]
            if support_v.shape[-2:] != (1, 1):
                support_v = F.adaptive_avg_pool2d(support_v, 1)
            
            # Expand support_v to match spatial dimensions after DWT
            B, C, H, W = cog_feat.shape
            # Broadcast to correct channels and expand spatially
            if support_v.shape[1] != C:
                # Use repeat instead of expand for channel dimension
                repeat_factor = (C + support_v.shape[1] - 1) // support_v.shape[1]  # Ceiling division
                support_v = support_v.repeat(1, repeat_factor, 1, 1)[:, :C, :, :]
            support_v = support_v.expand(B, C, H // 2, W // 2)
            
            # Apply EWTB with support
            cog_enhanced = ewtb(cog_feat, support_v)
            
            # Bidirectional fusion with inspective features
            insp_feat = inspective_features[min(idx, len(inspective_features)-1)]
            cog_enhanced = fusion(cog_enhanced, insp_feat)
            
            enhanced_cognitive.append(cog_enhanced)
        
        # Add the last feature without EWTB
        enhanced_cognitive.append(cognitive_features[-1])
        
        # Decoder with dual-stream fusion (capture intermediate features for aux heads)
        decoder_features = []
        x_dec = self.decoders[0](enhanced_cognitive[-2], enhanced_cognitive[-1])
        decoder_features.append(x_dec)
        
        for idx in range(1, min(len(self.decoders), len(enhanced_cognitive) - 1)):
            feat_idx = -3 - (idx - 1)
            # Map inspective features: decoder[1]->insp[-1], decoder[2]->insp[-2], decoder[3]->insp[-3]
            insp_idx = -idx
            insp_feat = inspective_features[insp_idx] if abs(insp_idx) <= len(inspective_features) else None
            x_dec = self.decoders[idx](enhanced_cognitive[feat_idx], x_dec, insp_feat)
            decoder_features.append(x_dec)
        
        # Final prediction
        main_output = self.final_conv(x_dec)
        
        # Return early if auxiliary outputs not needed
        if not return_aux:
            return main_output
        
        # Generate auxiliary predictions from selected decoder stage
        aux_features = decoder_features[self.aux_from_decoder_idx]
        
        # Boundary prediction
        boundary_output = self.boundary_head(aux_features)
        boundary_output = F.interpolate(
            boundary_output, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Position offset prediction
        offset_output = self.offset_head(aux_features)
        offset_output = F.interpolate(
            offset_output, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        return {
            'main': main_output,
            'boundary': boundary_output,
            'offset': offset_output
        }