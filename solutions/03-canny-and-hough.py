# Exercise 3 — Canny from Scratch & Hough Lines — solution
# Generated from exercises/03-canny-and-hough.md by scripts/extract_solutions.py.
# Run:  python solutions/03-canny-and-hough.py   (writes output files to the cwd)

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
