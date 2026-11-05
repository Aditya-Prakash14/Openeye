# Exercise 1 — Image Fundamentals

**Goal:** stop treating images as files and start treating them as what they really are: numpy arrays of numbers.
**Concepts:** BGR vs RGB, channels, color spaces, resizing vs cropping, point operations, histograms.
**Needs:** core track. **Time:** ~20 min.

## Tasks

1. Build a 240×320 test image **in code** (gradient + a white circle + text). No downloads.
2. Convert it BGR→RGB, BGR→Gray, BGR→HSV. Explain in one line why rule 1 of OpenCV is "know your channel order".
3. Resize it to half size, crop a region, flip it, rotate it 30°, using **only** `cv2.*` calls.
4. Apply point operations: brightness (`+40` on every pixel), and a manual threshold via numpy boolean indexing (no `cv2.threshold`).
5. Compute the grayscale histogram twice: once with `np.histogram`, once with `cv2.calcHist`. Assert they agree.
6. Display everything in one matplotlib grid.

> Hint: pixels are `uint8` (0–255). Adding 40 without care overflows (`240 + 40` wraps to `24`). Use `cv2.convertScaleAbs` or numpy casts.

<details>
<summary>Solution</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Synthetic test image: green ramps right, blue ramps down, white circle, text
h, w = 240, 320
img = np.zeros((h, w, 3), np.uint8)
img[:, :, 1] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]   # green
img[:, :, 2] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]   # red
cv2.circle(img, (160, 120), 60, (255, 255, 255), -1)
cv2.putText(img, "CV", (118, 136), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

# 2. Color spaces
b, g, r = cv2.split(img)                          # channel order is B, G, R
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # for matplotlib, which expects RGB
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
assert rgb[0, 0, 2] == img[0, 0, 2]               # rgb is just a channel reorder

# 3. Geometry
small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
crop = gray[60:180, 100:260]                      # numpy slicing: [rows, cols] = [y, x]
flipped = cv2.flip(gray, 1)
rot = cv2.warpAffine(gray, cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0), (w, h))

# 4. Point operations
bright = cv2.convertScaleAbs(gray, alpha=1.0, beta=40)   # saturating += 40
thresh = (gray > 128).astype(np.uint8) * 255             # manual threshold

# 5. Histograms must agree
hist_np = np.histogram(gray, bins=256, range=(0, 256))[0]
hist_cv = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
assert np.allclose(hist_np, hist_cv)

# 6. Display
fig, ax = plt.subplots(2, 4, figsize=(14, 6))
for a, im, title in zip(ax.ravel(),
                        [rgb, gray, rgb[:, :, 0], small, crop, flipped, bright, thresh],
                        ["RGB", "Gray", "R channel", "Resized", "Crop", "Flip",
                         "Brightened", "Threshold"]):
    a.imshow(im, cmap="gray")
    a.set_title(title)
    a.axis("off")
plt.tight_layout()
plt.savefig("ex01.png", dpi=110)
print("saved ex01.png")

fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3))
ax2[0].plot(hist_np)
ax2[0].set_title("histogram of gray")
ax2[1].imshow(hsv[:, :, 0], cmap="hsv")
ax2[1].set_title("HSV hue channel")
ax2[1].axis("off")
plt.tight_layout()
plt.savefig("ex01_hist.png", dpi=110)
```

### What's happening

- **An image is an array**: shape `(H, W, 3)`, dtype `uint8`. Image processing *is* array processing.
- **`cv2` is BGR, matplotlib is RGB** — the classic bug where red and blue swap in every plot.
- **Resize vs crop**: `cv2.resize` interpolates to new dimensions; slicing *selects* pixels. Use `INTER_AREA` for shrinking and `INTER_LINEAR`/`INTER_CUBIC` for enlarging.
- **Saturating arithmetic**: `uint8 + 40` wraps around by default. `cv2.convertScaleAbs` (or `np.clip`) saturates at 255, which is what you want for brightness.
- **Histograms** summarize intensity distribution and are the input to most thresholding techniques.

### Stretch

- Load a real photo with `cv2.samples.findFile("lena.jpg")` (downloads once), redo step 5.
- Swap gray→HSV and threshold only "green" pixels by range (`cv2.inRange`).
</details>
