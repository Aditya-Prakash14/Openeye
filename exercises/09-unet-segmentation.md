# Exercise 9 — Semantic Segmentation with a U-Net

**Goal:** train a real (tiny) U-Net to segment objects pixel-by-pixel, and measure it with Dice and IoU.
**Concepts:** fully-convolutional nets, encoder–decoder, skip connections, per-pixel loss, segmentation metrics.
**Needs:** `torch` (advanced track). **Time:** ~45 min (includes ~1–2 min CPU training).

## Tasks

1. Generate a synthetic dataset: 128×128 grayscale images with 1–3 bright shapes on noisy background, then apply a smooth **illumination ramp** (random direction/strength) — so that no global intensity threshold can separate shapes from background. The mask is the binary union of the shapes. 192 train / 64 test images.
2. Implement a **tiny U-Net**: 3 down-blocks (conv-BN-ReLU), a bottleneck, 3 up-blocks with **skip connections**, 1×1 head. (15 lines of module definitions — that's all a U-Net is.)
3. Train 6 epochs with `Adam` + `BCEWithLogitsLoss` and print loss per epoch. (~40 s on CPU)
4. Evaluate on the test set: threshold `sigmoid(logits) > 0.5` and compute **Dice** and **IoU**. Assert Dice > 0.85 **and** at least 0.15 above the threshold baseline.
5. Print that trivial baseline (threshold the input image directly) for contrast — the ramp is what makes it fail, and that gap is what the learned model buys you.
6. Visualize 3 test cases: input, ground truth mask, predicted mask.

> U-Net in one sentence: shrink the image while widening channels (encoder), then grow it back, **concatenating** the encoder features at each level so fine detail survives (skip connections).

<details>
<summary>Solution</summary>

```python
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
```

### What's happening

- **Why U-Net and not an MLP?** Convolutions share weights across space (translation invariance), and the encoder–decoder keeps spatial layout, so the output is a mask with the same resolution as the input.
- **Skip connections** concatenate high-resolution features from the encoder into the decoder. Without them, decoder output is blurry — you'd lose exactly the boundary detail segmentation is judged on.
- **BCEWithLogitsLoss** on every pixel = per-pixel binary classification. Alternatives: Dice loss (better for class imbalance), focal loss, or a BCE+Dice combo (common in medical imaging).
- **Dice vs IoU**: both measure overlap; Dice = 2·|A∩B|/(|A|+|B|), IoU = |A∩B|/|A∪B|. Dice is more forgiving of small objects.
- **The baseline comparison is the point**: the learned model must beat trivial thresholding by a *measurable* margin to justify existing. Here the illumination ramp guarantees the threshold can't, and the model does — you can see both numbers.
- **The ramp is also a realism upgrade**: real footage has spotlights, shadows and exposure differences. A model that only learned "bright = object" fails in the wild.

### Stretch

- Remove the illumination ramp and re-run: the threshold baseline jumps to ~0.95. That's *exactly* the shortcut the ramp removes.
- Swap the loss to soft Dice: `1 - (2·p·y + 1)/(p + y + 1)` with `p = sigmoid(logits)`.
- Log its to 2–3 classes (circle / square / triangle) with `CrossEntropyLoss` and `argmax` at inference.
- Scale to real data: Oxford-IIIT Pet (`torchvision.datasets.OxfordIIITPet(..., target_types="segmentation")`).
</details>
