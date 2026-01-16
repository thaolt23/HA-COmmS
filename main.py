import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import HASemCom
from dataset import UAVToyDataset
from explain import print_ha_report

device = "cpu"

epochs = 8
batch_size = 32
lr = 1e-3

lambda_ha = 0.1
lambda_ent = 0.01   # chống collapse

ha_centers = torch.tensor(
    [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
    device=device
)

# Dataset
dataset = UAVToyDataset(train=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = HASemCom(device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Training...")

# ---------------- Training ----------------
for epoch in range(epochs):
    sum_task, sum_ha, sum_ent = 0.0, 0.0, 0.0
    count = 0

    for img, obj in loader:
        img = img.to(device)
        obj = obj.to(device)

        # small noise to avoid trivial matching
        obj = obj + 0.05 * torch.randn_like(obj)
        obj = torch.clamp(obj, 0, 1)

        M, r = model(img)

        # Task loss
        L_task = F.mse_loss(M, obj)

        # Hedge Algebra regularization
        L_HA = ((M - ha_centers[r]) ** 2).mean()

        # Entropy regularization (anti-collapse)
        entropy = -(M * torch.log(M + 1e-6) + (1 - M) * torch.log(1 - M + 1e-6))
        L_ent = entropy.mean()

        loss = L_task + lambda_ha * L_HA + lambda_ent * L_ent

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_task += L_task.item()
        sum_ha += L_HA.item()
        sum_ent += L_ent.item()
        count += 1

    print(
        f"Epoch {epoch+1}, "
        f"L_task={sum_task/count:.4f}, "
        f"L_HA={sum_ha/count:.4f}, "
        f"L_ent={sum_ent/count:.4f}"
    )

# ---------------- Explainability ----------------
print("\n=== Final HA Distribution ===")
with torch.no_grad():
    img, _ = next(iter(loader))
    img = img.to(device)

    _, r = model(img)
    report = model.ha.explain(r)
    print_ha_report(report)
