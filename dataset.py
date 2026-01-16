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

        H, W = 8, 8
        obj = torch.zeros(1, H, W)

        # Object 1
        cx1 = random.randint(2, 5)
        cy1 = random.randint(2, 5)

        # Object 2
        cx2 = random.randint(1, 6)
        cy2 = random.randint(1, 6)

        for i in range(H):
            for j in range(W):
                dist1 = (i - cx1) ** 2 + (j - cy1) ** 2
                dist2 = (i - cx2) ** 2 + (j - cy2) ** 2

                dist1 = torch.tensor(dist1, dtype=torch.float32)
                dist2 = torch.tensor(dist2, dtype=torch.float32)

                obj[0, i, j] = (
                    torch.exp(-dist1 / 6.0) +
                    0.5 * torch.exp(-dist2 / 6.0)
                )

        obj = (obj - obj.min()) / (obj.max() - obj.min() + 1e-6)

        return img, obj
