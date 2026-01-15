import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import HASemCom
from dataset import UAVToyDataset
from explain import print_ha_report

device = "cpu"

# Dataset
ds = UAVToyDataset(train=True)
loader = DataLoader(ds, batch_size=32, shuffle=True)

# Model
model = HASemCom(device).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

ha_centers = torch.tensor(
    [0.125,0.25,0.375,0.5,0.625,0.75,0.875],
    device=device
)

λ_ha = 0.1

print("Training...")
for epoch in range(8):
    for img, obj in loader:
        img, obj = img.to(device), obj.to(device)

        M, O, r = model(img)

        L_task = F.mse_loss(M, O)
        L_HA = ((M - ha_centers[r]) ** 2).mean()

        loss = L_task + λ_ha * L_HA

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch+1}, L_task={L_task.item():.4f}, L_HA={L_HA.item():.4f}")

# Explainability
with torch.no_grad():
    img, _ = next(iter(loader))
    M, _, r = model(img.to(device))
    report = model.ha.explain(r)
    print_ha_report(report)
