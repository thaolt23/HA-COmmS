import torch
import torch.nn as nn
import torch.nn.functional as F
from hedge_algebra import HedgeAlgebra

# -------- Feature Extractor --------
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
        return self.net(x)   # [B,64,8,8]


# -------- Importance Network --------
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


# -------- Detection Head --------
class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, f):
        return self.net(f)


# -------- Full Model --------
class HASemCom(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.feat = FeatureExtractor()
        self.imp = ImportanceNet()
        self.det = DetectionHead()
        self.ha = HedgeAlgebra(device)

    def forward(self, x):
        f = self.feat(x)
        M = self.imp(f)
        O = self.det(f)
        r = self.ha.assign(M)
        return M, O, r
