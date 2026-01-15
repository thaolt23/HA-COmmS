import torch

class HedgeAlgebra:
    def __init__(self, device="cpu"):
        self.labels = [
            "very low", "low", "little low",
            "medium",
            "little high", "high", "very high"
        ]
        self.centers = torch.tensor(
            [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
            device=device
        )
        self.bits = torch.tensor([1, 2, 4, 5, 6, 8, 10], device=device)

    def assign(self, importance):
        """
        importance: [B,1,H,W] ∈ [0,1]
        return: HA index map [B,1,H,W]
        """
        diff = torch.abs(importance.unsqueeze(-1) - self.centers)
        return torch.argmin(diff, dim=-1)

    def get_bits(self, ha_index):
        return self.bits[ha_index]

    def explain(self, ha_index):
        total = ha_index.numel()
        report = {}
        for i, label in enumerate(self.labels):
            ratio = (ha_index == i).float().sum().item() / total * 100
            report[label] = ratio
        return report
