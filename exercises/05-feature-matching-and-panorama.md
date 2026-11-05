# Exercise 5 — Feature Matching & Panorama Stitching

**Goal:** detect keypoints, match them between two views, and recover the camera transform with RANSAC.
**Concepts:** ORB, descriptor distance, Lowe's ratio test, homography, RANSAC, warping/blending.
**Needs:** core track. **Time:** ~30 min.

## Tasks

1. Build a textured synthetic "scene" (random noise + colored discs). Warp it with a known homography `H_true` to get a second view.
2. Detect ORB keypoints/descriptors in both views. Print keypoint counts.
3. Match with `BFMatcher(NORM_HAMMING)` + `knnMatch(k=2)` and **Lowe's ratio test** (`d1 < 0.75·d2`). Assert you keep > 30 matches.
4. Estimate the homography with `cv2.findHomography(..., cv2.RANSAC, 3.0)`. Assert that the inlier ratio > 0.5, and that `H_est` moves the image corners at most a few pixels from where `H_true` sends them. — this is the real math check.
5. Stitch: warp view 2 back into view 1's frame with `inv(H_est)` and blend the two.
6. **Stretch:** stitch a panorama from two crops of the *same* scene (real panorama flow), or add blur/noise and watch matching degrade.

> Hint: matches come as `(queryIdx, trainIdx)` indices. `src` points come from view A keypoints, `dst` from view B — keep the direction consistent or your homography will be inverted.

<details>
<summary>Solution</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
Hh, Ww = 400, 700

# ---- 1. Textured scene + a known warp ----
scene = rng.integers(0, 255, (Hh, Ww, 3), np.uint8)
scene = cv2.GaussianBlur(scene, (3, 3), 0.8)          # keep some noise texture for ORB
for _ in range(25):
    p = tuple(int(v) for v in rng.integers([30, 30], [Ww - 30, Hh - 30]))
    col = tuple(int(v) for v in rng.integers(0, 255, 3))
    cv2.circle(scene, p, int(rng.integers(8, 28)), col, -1)

H_true = np.array([[1.0, 0.05, 60.0],
                   [0.02, 1.0, 25.0],
                   [1e-4, 2e-4, 1.0]])
view = cv2.warpPerspective(scene, H_true, (Ww, Hh))

# ---- 2. ORB keypoints ----
orb = cv2.ORB_create(nfeatures=3000)
g1 = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
k1, d1 = orb.detectAndCompute(g1, None)
k2, d2 = orb.detectAndCompute(g2, None)
print(f"keypoints: scene {len(k1)}, view {len(k2)}")
assert len(k1) > 200 and len(k2) > 200

# ---- 3. Match + ratio test ----
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
print("matches after ratio test:", len(good))
assert len(good) > 30, "not enough matches — add more texture to the scene"

# ---- 4. RANSAC homography: scene -> view (same direction as H_true) ----
src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H_est, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
assert H_est is not None
inlier_ratio = mask.mean()
print(f"inliers: {int(mask.sum())}/{len(good)} ({inlier_ratio:.0%})")
assert inlier_ratio > 0.5

Hn = H_est / H_est[2, 2]
Ht = H_true / H_true[2, 2]
corners = np.float32([[0, 0], [Ww, 0], [Ww, Hh], [0, Hh]]).reshape(-1, 1, 2)
drift = np.linalg.norm(cv2.perspectiveTransform(corners, Hn)
                       - cv2.perspectiveTransform(corners, Ht), axis=2).max()
print("max corner drift vs ground truth (px):", round(float(drift), 2))
assert drift < 5.0, "recovered transform is far from ground truth"

# ---- 5. Stitch: bring view back into scene frame ----
aligned = cv2.warpPerspective(view, np.linalg.inv(H_est), (Ww, Hh))
blend = cv2.addWeighted(scene, 0.5, aligned, 0.5, 0)

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, im, t in zip(ax, [scene, view, aligned, blend],
                    ["scene", "view (scene warped by H_true)", "view re-aligned", "blend"]):
    a.imshow(im[..., ::-1])
    a.set_title(t)
    a.axis("off")
plt.tight_layout()
plt.savefig("ex05.png", dpi=110)
print("saved ex05.png")
```

### What's happening

- **ORB** = FAST keypoints + rotated BRIEF descriptors. Because BRIEF bits are compared *within* a keypoint (rotated to its dominant orientation), matching is invariant to rotation and lighting, but **not** to big out-of-plane changes (use SIFT/learned features like SuperPoint for those).
- **Ratio test**: for a query descriptor, the best match should be clearly better than the second-best. Ambiguous matches get rejected. This single line removes most false matches.
- **RANSAC**: 4-point samples propose homographies; the model with the most inliers wins. That's why feature matching tolerates 30–40% garbage matches.
- **A homography is the exact model** for two views of the same *plane* (or a pure rotation). For general 3D scenes, panoramas need rotation-only changes or per-plane models.

### Stretch

- Real panorama: take `scene[:, :450]` and `scene[:, 250:]` as two overlapping views, match, estimate homography, warp onto a wider canvas.
- Compare `cv2.SIFT_create()` (float descriptors, `NORM_L2`) on the same pair — usually more and better matches, slower.
</details>
