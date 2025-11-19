import torch
import torch.nn as nn

from .attention import ELA_Tiny, ELA_Base, ELA_Small, ELA_Large

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Use fused multiply instead of *
        return x.mul(y)


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se=False, use_ela=False):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if use_se and use_ela:
            raise ValueError("MBConv: use_se and use_ela cannot be both True.")
        
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif use_ela:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                ELA_Base(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else: # Fused MBConv
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, output_channel=1792, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        input_channel = _make_divisible(24 * width_mult, 8)

        self.input_channel = input_channel
        self.output_channel = _make_divisible(output_channel * width_mult, 8) if width_mult > 1.0 else output_channel

        # building first layer
        layers = [conv_3x3_bn(3, input_channel, 2)]
        
        # building inverted residual blocks
        block = MBConv

        idx = 0
        self.ckpt_layers = {}
        for t, c, n, s, use_se, use_ela in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)

            # Only save checkpoint if there's a stride (spatial resolution change)
            # or if it's the first block
            # if s == 2 or idx == 0:
            #     self.ckpt_layers[idx + n] = (input_channel, output_channel)
            # elif s == 1 and s_prev != 1:
            #     prev_input_channel, prev_output_channel = self.ckpt_layers[idx + n]
            #     self.ckpt_layers[idx + n] = (prev_output_channel, input_channel)

            #     del self.ckpt_layers[idx]
            # else:
            #     s_prev = s

            if s != 1:
                self.ckpt_layers[idx + n] = (input_channel, output_channel)
            elif idx != 0:
                prev_input_channel, _ = self.ckpt_layers[idx]
                del self.ckpt_layers[idx]
                
                self.ckpt_layers[idx + n] = (prev_input_channel, output_channel)

            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se, use_ela))
                input_channel = output_channel

            idx += n


        layers.append(conv_1x1_bn(input_channel, self.output_channel))

        # Add final output layer as checkpoint
        self.ckpt_layers[idx + 1] = (self.ckpt_layers[idx][0], self.output_channel)
        del self.ckpt_layers[idx]
            
        self.features = nn.Sequential(*layers)        

        self._initialize_weights()

    def forward(self, x, return_ckpt=False):

        ckpt_outputs = []
        for idx, layer in enumerate(self.features):
            
            x = layer(x)
                        
            if idx in self.ckpt_layers and return_ckpt:
                ckpt_outputs.append(x)
        
        if return_ckpt:
            return ckpt_outputs

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use torch operations instead of math.sqrt
                # Compute n using torch for potential device compatibility
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                std = torch.sqrt(torch.tensor(2.0 / n))
                m.weight.data.normal_(0, std.item())
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE, ELA
        [1,  24,  2, 1, 0, 0],
        [4,  48,  4, 2, 0, 0],
        [4,  64,  4, 2, 0, 0],
        [4, 128,  6, 2, 0, 1],
        [6, 160,  9, 1, 0, 1],
        [6, 256, 15, 2, 0, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE, ELA
        [1,  24,  3, 1, 0, 0],
        [4,  48,  5, 2, 0, 0],
        [4,  80,  5, 2, 0, 0],
        [4, 160,  7, 2, 0, 1],
        [6, 176, 14, 1, 0, 1],
        [6, 304, 18, 2, 0, 1],
        [6, 512,  5, 1, 0, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE, ELA
        [1,  32,  4, 1, 0, 0],
        [4,  64,  7, 2, 0, 0],
        [4,  96,  7, 2, 0, 0],
        [4, 192, 10, 2, 0, 1],
        [6, 224, 19, 1, 0, 1],
        [6, 384, 25, 2, 0, 1],
        [6, 640,  7, 1, 0, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE, ELA
        [1,  32,  4, 1, 0, 0],
        [4,  64,  8, 2, 0, 0],
        [4,  96,  8, 2, 0, 0],
        [4, 192, 16, 2, 0, 1],
        [6, 256, 24, 1, 0, 1],
        [6, 512, 32, 2, 0, 1],
        [6, 640,  8, 1, 0, 1],
    ]
    return EffNetV2(cfgs, **kwargs)