# Exercise 9 — Semantic Segmentation with a U-Net — solution
# Generated from exercises/09-unet-segmentation.md by scripts/extract_solutions.py.
# Run:  python solutions/09-unet-segmentation.py   (writes output files to the cwd)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)
rng = np.random.default_rng(0)
SIZE = 128

# ---- 1. Synthetic segmentation dataset ----
def sample_shape(rng):
    img = rng.normal(110, 12, (SIZE, SIZE)).clip(0, 255).astype(np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        kind = int(rng.integers(0, 3))
        cx, cy = (int(v) for v in rng.integers(20, SIZE - 20, 2))
        gray = int(rng.integers(160, 256))
        if kind == 0:
            r = int(rng.integers(10, 30))
            cv2.circle(img, (cx, cy), r, gray, -1)
            cv2.circle(mask, (cx, cy), r, 255, -1)
        elif kind == 1:
            s = int(rng.integers(16, 44))
            p1, p2 = (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2)
            cv2.rectangle(img, p1, p2, gray, -1)
            cv2.rectangle(mask, p1, p2, 255, -1)
        else:
            r = int(rng.integers(14, 32))
            pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
            cv2.fillPoly(img, [pts], gray)
            cv2.fillPoly(mask, [pts], 255)
    # illumination ramp applied AFTER drawing: global thresholds can no longer work,
    # shapes are only locally brighter than their surroundings
    ramp = np.linspace(0.55, 1.45, SIZE, dtype=np.float32)
    if rng.random() < 0.5:
        ramp = ramp[::-1]
    illum = ramp[None, :] if rng.random() < 0.5 else ramp[:, None]
    img = np.clip(img.astype(np.float32) * illum + rng.normal(0, 10, img.shape), 0, 255) / 255.0
    return img[..., None].astype(np.float32), (mask > 0).astype(np.float32)[..., None]

def to_tensor(items):
    x = torch.from_numpy(np.stack([i[0] for i in items])).permute(0, 3, 1, 2)
    y = torch.from_numpy(np.stack([i[1] for i in items])).permute(0, 3, 1, 2)
    return x, y

N_TRAIN, N_TEST = 192, 64
train_x, train_y = to_tensor([sample_shape(rng) for _ in range(N_TRAIN)])
test_x, test_y = to_tensor([sample_shape(rng) for _ in range(N_TEST)])
print("train:", tuple(train_x.shape), "test:", tuple(test_x.shape))

# ---- 2. Tiny U-Net ----
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)

class TinyUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.d1, self.d2, self.d3 = (DoubleConv(1, base),
                                     DoubleConv(base, base * 2),
                                     DoubleConv(base * 2, base * 4))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.d1(x)                          # 128
        x2 = self.d2(self.pool(x1))              # 64
        x3 = self.d3(self.pool(x2))              # 32
        b = self.bottleneck(self.pool(x3))       # 16
        y3 = self.c3(torch.cat([self.u3(b), x3], 1))
        y2 = self.c2(torch.cat([self.u2(y3), x2], 1))
        y1 = self.c1(torch.cat([self.u1(y2), x1], 1))
        return self.head(y1)

# ---- 3. Train ----
model = TinyUNet(base=16)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(6):
    model.train()
    perm = torch.randperm(N_TRAIN)
    epoch_loss = 0.0
    for i in range(0, N_TRAIN, 16):
        idx = perm[i:i + 16]
        opt.zero_grad()
        loss = loss_fn(model(train_x[idx]), train_y[idx])
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * len(idx)
    print(f"epoch {epoch + 1}: loss {epoch_loss / N_TRAIN:.4f}")

# ---- 4. Evaluate ----
model.eval()
with torch.no_grad():
    pred = (torch.sigmoid(model(test_x)) > 0.5).float()

def dice(a, b):
    return (2 * (a * b).sum() / (a.sum() + b.sum() + 1e-9)).item()

def seg_iou(a, b):
    inter = (a * b).sum()
    return (inter / (a.sum() + b.sum() - inter + 1e-9)).item()

print(f"test Dice: {dice(pred, test_y):.3f}   IoU: {seg_iou(pred, test_y):.3f}")

# ---- 5. Trivial baseline for contrast: threshold the raw image ----
trivial = (test_x > 0.65).float()
baseline = dice(trivial, test_y)
print(f"threshold baseline Dice: {baseline:.3f}  (a global threshold can't cope with the ramp)")

assert dice(pred, test_y) > 0.85, "training didn't converge — rerun, or raise epochs to 10"
assert dice(pred, test_y) > baseline + 0.15, "the model must beat the trivial baseline to justify existing"

# ---- 6. Visualize ----
fig, ax = plt.subplots(3, 3, figsize=(9, 9))
for r in range(3):
    for c, (im, t) in enumerate([(test_x[r, 0], "input"),
                                 (test_y[r, 0], "ground truth"),
                                 (pred[r, 0], "prediction")]):
        ax[r, c].imshow(im, cmap="gray", vmin=0, vmax=1)
        ax[r, c].set_title(t if r == 0 else "")
        ax[r, c].axis("off")
plt.tight_layout()
plt.savefig("ex09.png", dpi=110)
print("saved ex09.png")
