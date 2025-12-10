import torch
import torch.nn as nn

from .decoder import DecoderBlock
from .effnetv2 import effnetv2_s
from .frequency import FrequencyConvLayer


class CMSegNet(nn.Module):
    def __init__(
        self, encoder_attention_type="sa", decoder_attention_type="sa", in_channels=3
    ):
        """
        Args:
            encoder_attention_type: Type of attention mechanism to use ('sa' for SpatialAttention, 'cara' for CARA, or 'ela' for ELA)
            decoder_attention_type: Type of attention mechanism for decoder ('sa' or 'cara')
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        """
        super(CMSegNet, self).__init__()

        use_sa = False
        use_cara = False
        use_ela = False

        if encoder_attention_type == "sa":
            use_sa = True
        elif encoder_attention_type == "cara":
            use_cara = True
        elif encoder_attention_type == "ela":
            use_ela = True
        else:
            raise ValueError(
                "Unsupported attention_type. Choose 'sa', 'cara', or 'ela'."
            )

        self.backbone = effnetv2_s(
            in_channels=in_channels, use_se=use_sa, use_ela=use_ela, use_cara=use_cara
        )
        """
        Checkpoints feature map shapes:
        {6: (24, 48), 10: (48, 64), 25: (64, 160), 41: (160, 1792)}
        """

        decoders = []
        for idx, (n_layer, (in_c, out_c)) in enumerate(
            self.backbone.ckpt_layers.items()
        ):
            decoders.append(
                DecoderBlock(
                    in_c,
                    out_c,
                    use_cosa=idx != 0,
                    attention_type=decoder_attention_type,
                )
            )
            if idx == 0:
                self.dconv = nn.ConvTranspose2d(out_c, in_c, 4, stride=4, padding=0)
                self.conv_last = nn.Conv2d(in_c, 3, 1)

        self.decoders = nn.ModuleList(
            decoders[::-1]  # Reverse the list to match the order of feature maps
        )

        self.conv_score = nn.Conv2d(3, 1, 1)

        # Fuse final convolutions for better performance
        self.final_conv = nn.Sequential(self.dconv, self.conv_last, self.conv_score)

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


class CMFreqSegNet(nn.Module):
    def __init__(
        self, encoder_attention_type="sa", decoder_attention_type="sa", in_channels=3
    ):
        """
        Frequency-enhanced variant of CMSegNet with FCL layers at bottleneck and skip connections.

        Args:
            encoder_attention_type: Type of attention mechanism to use ('sa' for SpatialAttention, 'cara' for CARA, or 'ela' for ELA)
            decoder_attention_type: Type of attention mechanism for decoder ('sa' or 'cara')
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        """
        super(CMFreqSegNet, self).__init__()

        use_sa = False
        use_cara = False
        use_ela = False

        if encoder_attention_type == "sa":
            use_sa = True
        elif encoder_attention_type == "cara":
            use_cara = True
        elif encoder_attention_type == "ela":
            use_ela = True
        else:
            raise ValueError(
                "Unsupported attention_type. Choose 'sa', 'cara', or 'ela'."
            )

        self.backbone = effnetv2_s(
            in_channels=in_channels, use_se=use_sa, use_ela=use_ela, use_cara=use_cara
        )
        """
        Checkpoints feature map shapes:
        {6: (24, 48), 10: (48, 64), 25: (64, 160), 41: (160, 1792)}
        """

        # Create FCL layers for each checkpoint level
        # features_ckpts returns [48, 64, 160, 1792] (the out_c values)
        # We apply FCL to each feature map at its respective scale
        checkpoint_channels = [
            out_c for _, (_, out_c) in self.backbone.ckpt_layers.items()
        ]
        self.fcl_layers = nn.ModuleList(
            [FrequencyConvLayer(channels) for channels in checkpoint_channels]
        )

        decoders = []
        for idx, (n_layer, (in_c, out_c)) in enumerate(
            self.backbone.ckpt_layers.items()
        ):
            decoders.append(
                DecoderBlock(
                    in_c,
                    out_c,
                    use_cosa=idx != 0,
                    attention_type=decoder_attention_type,
                )
            )

            if idx == 0:
                self.dconv = nn.ConvTranspose2d(out_c, in_c, 4, stride=4, padding=0)
                self.conv_last = nn.Conv2d(in_c, 3, 1)

        self.decoders = nn.ModuleList(
            decoders[::-1]  # Reverse the list to match the order of feature maps
        )

        self.conv_score = nn.Conv2d(3, 1, 1)

        # Fuse final convolutions for better performance
        self.final_conv = nn.Sequential(self.dconv, self.conv_last, self.conv_score)

    def forward(self, x):
        # Extract features with checkpoints
        features_ckpts = self.backbone(x, return_ckpt=True)

        # Apply frequency enhancement to features
        # features_ckpts has 4 elements: [48, 64, 160, 1792] channels
        freq_features = []
        for idx, feat in enumerate(features_ckpts):
            # Apply corresponding FCL layer to each feature map
            freq_feat = self.fcl_layers[idx](feat)
            freq_features.append(freq_feat)

        # Initial decoder pass with frequency-enhanced features
        x = self.decoders[0](freq_features[-2], freq_features[-1])

        # Sequential decoder passes - iterate efficiently
        for idx in range(1, min(len(self.decoders), len(freq_features) - 1)):
            feat_idx = -3 - (idx - 1)
            x = self.decoders[idx](freq_features[feat_idx], x)

        # Apply fused final convolutions
        x = self.final_conv(x)

        return x
