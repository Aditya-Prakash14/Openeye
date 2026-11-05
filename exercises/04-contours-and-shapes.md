# Exercise 4 — Contours & Shape Analysis

**Goal:** turn a binary mask into structured objects: contours, shape classification, centroids, counts.
**Concepts:** Otsu thresholding, contour hierarchy, `approxPolyDP`, circularity, image moments.
**Needs:** core track. **Time:** ~25 min.

## Tasks

1. Generate a 600×400 scene with four black shapes on white: a circle, a square, a triangle, and a 5-point star. Add realistic Gaussian noise.
2. Threshold it: blur → `cv2.threshold(..., THRESH_BINARY_INV + THRESH_OTSU)`. Explain in one line what Otsu picks.
3. Find external contours and drop anything smaller than 300 px².
4. For each contour, compute: area, perimeter, `approxPolyDP` vertex count, circularity `4πA/P²`, bounding box, and centroid via **image moments**.
5. Classify each shape: circularity > 0.85 → circle; 3 vertices → triangle; 4 vertices → square; else polygon. Draw the label + centroid on the image.
6. Assert you found exactly 4 shapes with the expected labels.

> Hint: `approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)` — epsilon is a fraction of the perimeter, and it changes the vertex count a lot. Try 0.01 and 0.04 to see.

<details>
<summary>Solution</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def star_points(cx, cy, r_out, r_in, n=5):
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = -np.pi / 2 + i * np.pi / n
        pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.array(pts, np.int32)

# ---- 1. Scene ----
img = np.full((400, 600, 3), 255, np.uint8)
cv2.circle(img, (110, 110), 55, (0, 0, 0), -1)
cv2.rectangle(img, (240, 55), (340, 155), (0, 0, 0), -1)
cv2.fillPoly(img, [np.array([[470, 165], [400, 55], [540, 55]])], (0, 0, 0))
cv2.fillPoly(img, [star_points(300, 300, 60, 26)], (0, 0, 0))

rng = np.random.default_rng(1)
noisy = np.clip(img.astype(np.int16) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)

# ---- 2. Threshold: Otsu picks the threshold that maximizes between-class variance ----
gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
t, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print("Otsu threshold:", t)

# ---- 3. Contours ----
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) > 300]

# ---- 4+5. Features and classification ----
vis = noisy.copy()
labels = []
for c in contours:
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    v = len(approx)
    circularity = 4 * np.pi * area / (peri ** 2)

    if circularity > 0.85:
        label = "circle"
    elif v == 3:
        label = "triangle"
    elif v == 4:
        label = "square"
    else:
        label = "polygon"
    labels.append(label)

    M = cv2.moments(c)                       # centroid from image moments
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(c)
    cv2.drawContours(vis, [approx], -1, (0, 255, 0), 2)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
    cv2.putText(vis, f"{label} A={int(area)}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 0, 0), 2)

# ---- 6. Verify ----
assert len(contours) == 4, f"expected 4 shapes, got {len(contours)}"
assert set(labels) == {"circle", "triangle", "square", "polygon"}, labels
print("shapes:", labels)

fig, ax = plt.subplots(1, 3, figsize=(14, 5))
for a, im, t_ in zip(ax, [noisy, binary, vis[..., ::-1]],
                     ["noisy input", f"Otsu binary (t={t:.0f})", "analysis"]):
    a.imshow(im, cmap="gray")
    a.set_title(t_)
    a.axis("off")
plt.tight_layout()
plt.savefig("ex04.png", dpi=110)
print("saved ex04.png")
```

### What's happening

- **Otsu** finds the threshold `t` that best separates the histogram into two classes — no manual tuning needed for well-separated bimodal images like this one.
- **Circularity** `4πA/P²` is 1.0 for a perfect circle and ~0.785 for a square, ~0.6 for a triangle. Combined with vertex count from `approxPolyDP` it makes a cheap shape classifier.
- **Moments**: `M["m00"]` is the area, `(m10/m00, m01/m00)` the centroid. They also give orientation via central moments if you need it.
- **`RETR_EXTERNAL`** returns only outer contours; `RETR_TREE` gives you holes and nesting (useful to exclude shapes inside shapes).

### Stretch

- Count coins instead of shapes: filter contours by area ratio to the median and count with `len(contours)` (classic "coin counter").
- Compute orientation of each shape from the central moments `mu20, mu02, mu11` and draw the principal axis.
</details>
