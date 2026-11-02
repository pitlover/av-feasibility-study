"""Small ResNet18 predicting 6-DoF relative camera motion from two RGB frames + K*."""
import torch
from torch import nn
from torchvision.models import resnet18
from camera import WIDTH, HEIGHT


class PoseResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()
        self.k_encoder = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(544, 256), nn.ReLU(), nn.Linear(256, 6))

    def forward(self, x, K):
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError('Expected paired RGB tensor [B,6,H,W]')
        if K.ndim != 3 or K.shape[1:] != (3, 3):
            raise ValueError('Expected K [B,3,3]')
        v = self.encoder(x)
        k = torch.stack([K[:, 0, 0] / WIDTH, K[:, 1, 1] / HEIGHT,
                         K[:, 0, 2] / WIDTH, K[:, 1, 2] / HEIGHT], dim=-1)
        k = self.k_encoder(k.to(v.dtype))
        return self.head(torch.cat([v, k], dim=-1))
