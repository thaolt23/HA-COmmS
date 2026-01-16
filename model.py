import torch
import torch.nn as nn
from hedge_algebra import HedgeAlgebra


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.net(x)


class ImportanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, f):
        return self.net(f)


class HASemCom(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.feature = FeatureExtractor()
        self.importance = ImportanceNet()
        self.ha = HedgeAlgebra(device)

    def forward(self, x):
        f = self.feature(x)
        M = self.importance(f)
        r = self.ha.assign(M)
        return M, r
