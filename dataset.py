import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import random

class UAVToyDataset(Dataset):
    def __init__(self, train=True):
        self.ds = CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, _ = self.ds[idx]

        # ----------------------------
        # Image-dependent objectness
        # ----------------------------
        H, W = 8, 8
        obj = torch.zeros(1, H, W)

        # random "object" location (simulate UAV target)
        cx = random.randint(2, 5)
        cy = random.randint(2, 5)

        for i in range(H):
            for j in range(W):
                dist = (i - cx) ** 2 + (j - cy) ** 2
                dist = torch.tensor(dist, dtype=torch.float32)
                obj[0, i, j] = torch.exp(-dist / 2.0)

        # normalize to [0,1]
        obj = (obj - obj.min()) / (obj.max() - obj.min() + 1e-6)

        return img, obj
