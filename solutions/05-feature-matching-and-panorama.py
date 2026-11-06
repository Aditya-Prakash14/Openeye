# Exercise 5 — Feature Matching & Panorama Stitching — solution
# Generated from exercises/05-feature-matching-and-panorama.md by scripts/extract_solutions.py.
# Run:  python solutions/05-feature-matching-and-panorama.py   (writes output files to the cwd)

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
