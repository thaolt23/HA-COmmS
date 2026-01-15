import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms

class UAVToyDataset(Dataset):
    def __init__(self, train=True):
        self.ds = CIFAR10(
            root="./data", train=train, download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, _ = self.ds[idx]

        # Fake objectness map (center-biased, UAV-like)
        H, W = 8, 8
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij"
        )
        objectness = torch.exp(-(x**2 + y**2) * 3)
        objectness = objectness.unsqueeze(0)  # [1,H,W]

        return img, objectness
