import torch.nn as nn

class ELA(nn.Module):
    def __init__(self,  channel, ks, ng):
        super(ELA, self).__init__()

        p0 = ks // 2
        p1 = (ks+2) // 2
        self.conv1 = nn.Conv1d(channel, channel, kernel_size=ks, padding=p0, groups=channel, bias=False)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=ks+2, padding=p1, groups=channel, bias=False)
        
        # Ensure ng divides channel evenly
        import math
        actual_ng = math.gcd(ng, channel)
        self.gn = nn.GroupNorm(actual_ng, channel)
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