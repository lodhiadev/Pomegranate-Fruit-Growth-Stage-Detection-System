import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        x = x * ca
        sa = torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]
        ], dim=1)
        sa = self.spatial(sa)
        return x * sa
