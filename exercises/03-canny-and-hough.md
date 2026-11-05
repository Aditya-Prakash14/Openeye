# Exercise 3 — Canny from Scratch & Hough Lines

**Goal:** build the classic edge pipeline yourself, then detect lines with the Hough transform.
**Concepts:** gradients, non-maximum suppression, hysteresis, voting schemes.
**Needs:** core track. **Time:** ~30 min.

## Tasks

1. Implement `canny_from_scratch(img, low, high)`:
   - Gaussian smoothing
   - Sobel gradients `gx, gy`, magnitude and direction
   - non-maximum suppression along the gradient direction (4 sectors: 0°, 45°, 90°, 135°)
   - double threshold + **hysteresis** (weak pixels survive only if connected to strong ones)
2. Compare with `cv2.Canny`: assert that ≥ 40% of your edge pixels land on or next to the reference edges (implementations differ in detail).
3. Draw 10 straight lines on a canvas at random angles. Detect them with `cv2.HoughLinesP` and overlay.
4. **Stretch:** detect circles on the same canvas with `cv2.HoughCircles`.

> Hint: hysteresis is a flood fill — BFS from every strong pixel through weak pixels. A `collections.deque` is all you need.

<details>
<summary>Solution</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def canny_from_scratch(img, low=50, high=150, ksize=3):
    # 1. smooth
    smooth = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 1.0)
    # 2. gradients + magnitude + direction
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx) * 180.0 / np.pi
    ang[ang < 0] += 180.0

    # 3. non-maximum suppression: keep a pixel only if it beats its two
    #    neighbors lying along the gradient direction (quantized to 4 sectors)
    nms = np.zeros_like(mag)
    for y in range(1, mag.shape[0] - 1):
        for x in range(1, mag.shape[1] - 1):
            a = ang[y, x]
            if a < 22.5 or a >= 157.5:
                n1, n2 = mag[y, x - 1], mag[y, x + 1]
            elif a < 67.5:
                n1, n2 = mag[y - 1, x + 1], mag[y + 1, x - 1]
            elif a < 112.5:
                n1, n2 = mag[y - 1, x], mag[y + 1, x]
            else:
                n1, n2 = mag[y - 1, x - 1], mag[y + 1, x + 1]
            nms[y, x] = mag[y, x] if mag[y, x] >= max(n1, n2) else 0.0

    # 4. double threshold + hysteresis (BFS flood fill)
    strong = nms >= high
    weak = (nms >= low) & ~strong
    edges = np.zeros(nms.shape, np.uint8)
    q = deque(map(tuple, np.argwhere(strong)))
    while q:
        y, x = q.popleft()
        if edges[y, x]:
            continue
        edges[y, x] = 255
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < edges.shape[0] and 0 <= nx < edges.shape[1] and weak[ny, nx]:
                    q.append((ny, nx))
    return edges

# ---- Build a scene with structure ----
img = np.full((300, 400), 40, np.uint8)
rng = np.random.default_rng(3)
for _ in range(10):
    p1 = tuple(rng.integers([0, 0], [400, 300]))
    p2 = tuple(rng.integers([0, 0], [400, 300]))
    cv2.line(img, p1, p2, 220, 3)
cv2.rectangle(img, (60, 180), (160, 260), 180, -1)
img = cv2.GaussianBlur(img, (3, 3), 0)
img = np.clip(img.astype(np.int16) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)

mine = canny_from_scratch(img, 40, 120)
ref = cv2.Canny(img, 60, 180)

# "on or next to": dilate the reference by 1 px and count how many of my pixels fall inside
ref_dilated = cv2.dilate(ref, np.ones((3, 3), np.uint8))
overlap = np.mean(ref_dilated[mine > 0] > 0) if (mine > 0).any() else 0.0
print(f"my edge pixels: {(mine > 0).sum()}, overlap with cv2.Canny: {overlap:.2%}")
assert overlap > 0.4, "too far from the reference implementation"

# ---- Hough lines ----
lines = cv2.HoughLinesP(mine, 1, np.pi / 180, threshold=60, minLineLength=60, maxLineGap=8)
vis = cv2.cvtColor(mine, cv2.COLOR_GRAY2BGR)
n = 0 if lines is None else len(lines)
for l in ([] if lines is None else lines[:, 0]):
    x1, y1, x2, y2 = l
    cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
print("hough segments:", n)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, im, t in zip(ax, [img, mine, vis[..., ::-1]], ["input", "my Canny", f"Hough ({n} lines)"]):
    a.imshow(im, cmap="gray")
    a.set_title(t)
    a.axis("off")
plt.tight_layout()
plt.savefig("ex03.png", dpi=110)
print("saved ex03.png")
```

### What's happening

- **Non-maximum suppression** thins the "thick" gradient ridge to 1-pixel edges: keep a pixel only if it's bigger than its neighbors *along* the gradient direction, so perpendicular edge pixels survive.
- **Hysteresis** connects edge chains: strong pixels seed the flood, weak pixels only survive when touching a chain. This removes noise speckle while keeping faint continuations of real edges.
- **Hough's trick**: every edge pixel *votes* for all lines that could pass through it. Real lines accumulate many votes; noise spreads its votes across the parameter space. `HoughLinesP` works on line segments (better for real images); the classic `HoughLines` works on infinite lines.
- Thresholds are usually set with a 2:1 to 3:1 ratio for low:high.

### Stretch

- Replace the Python NMS loop with `scipy.ndimage.maximum_filter` and compare speed.
- Detect circles: draw some filled circles, then `cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 30, param1=120, param2=25, minRadius=10, maxRadius=80)`.
</details>
