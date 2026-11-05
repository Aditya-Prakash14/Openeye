# Exercise 6 — Camera Calibration & 3D Pose

**Goal:** recover a camera's intrinsic matrix and lens distortion, then find a known object's 6-DoF pose from a single image.
**Concepts:** pinhole model, intrinsics `K`, distortion coefficients, reprojection error, `solvePnP`.
**Needs:** core track. **Time:** ~35 min.

## Tasks

1. Generate **synthetic** chessboard correspondences: 15 "views" of a 9×6 inner-corner board (square = 25 mm) projected through a known `K_true` (focal 1100 px, centered principal point) and a mild distortion, plus 0.3 px pixel noise.
2. Calibrate with `cv2.calibrateCamera`. Assert the RMS reprojection error < 1 px and the relative error on `K` < 5%.
3. Print `K_est` next to `K_true` and the estimated distortion vs. ground truth.
4. Take a **fresh** view (new random pose), estimate its pose with `cv2.solvePnP`, and assert the reprojection error < 1 px. Draw the projected 3D axes with `cv2.drawFrameAxes`.
5. Undistort the fresh view's points with `cv2.undistortPoints` and assert they match the ideal pinhole projection.
6. **Stretch:** use the helper `corners_from_photo()` with your own phone photos of a printed board.

> Why synthetic? Calibration math is the lesson, and synthetic data has ground truth you can assert against. The exact same pipeline runs on real photos — that's what the stretch is for.

<details>
<summary>Solution</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
W, H = 1280, 960
K_true = np.array([[1100.0, 0.0, W / 2],
                   [0.0, 1100.0, H / 2],
                   [0.0, 0.0, 1.0]])
dist_true = np.array([0.10, -0.05, 0.001, 0.001, 0.0])

# ---- 1. Synthetic correspondences: 9x6 inner corners, 25 mm squares ----
rows, cols = 6, 9
sq = 0.025
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq   # board lies in z=0 plane

objpoints, imgpoints = [], []
for _ in range(15):
    rvec = rng.uniform(-0.35, 0.35, 3)
    tvec = np.array([rng.uniform(-0.10, 0.10),
                     rng.uniform(-0.06, 0.06),
                     rng.uniform(0.55, 0.90)])
    pts, _ = cv2.projectPoints(objp, rvec, tvec, K_true, dist_true)
    pts = pts.reshape(-1, 2) + rng.normal(0, 0.3, (rows * cols, 2))   # pixel noise
    objpoints.append(objp)
    imgpoints.append(pts.astype(np.float32))

# ---- 2. Calibrate ----
rms, K_est, dist_est, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (W, H), None, None)
rel_err = np.linalg.norm(K_est - K_true) / np.linalg.norm(K_true)
print(f"RMS reprojection error: {rms:.3f} px")
print(f"relative error in K:   {rel_err:.4%}")
assert rms < 1.0
assert rel_err < 0.05

np.set_printoptions(precision=2, suppress=True)
print("\nK_est:\n", K_est, "\nK_true:\n", K_true)
print("\ndist_est:", np.round(dist_est.ravel(), 4), " dist_true:", dist_true)

# ---- 3+4. Fresh view -> solvePnP ----
rvec_t = np.array([0.15, -0.20, 0.10])
tvec_t = np.array([0.05, -0.02, 0.70])
proj, _ = cv2.projectPoints(objp, rvec_t, tvec_t, K_true, dist_true)
img_t = proj.reshape(-1, 2).astype(np.float32)

ok, rvec_est, tvec_est = cv2.solvePnP(objp, img_t, K_est, dist_est)
assert ok
reproj, _ = cv2.projectPoints(objp, rvec_est, tvec_est, K_est, dist_est)
pose_err = np.mean(np.linalg.norm(reproj.reshape(-1, 2) - img_t, axis=1))
print(f"\npose reprojection error: {pose_err:.4f} px")
print("tvec_est:", np.round(tvec_est.ravel(), 3), " tvec_true:", tvec_t)
assert pose_err < 1.0

canvas = np.full((H, W, 3), 255, np.uint8)
cv2.drawFrameAxes(canvas, K_est, dist_est, rvec_est, tvec_est, sq * 3)
for p in img_t.astype(int):
    cv2.circle(canvas, tuple(p), 3, (0, 0, 255), -1)

# ---- 5. Undistortion sanity check ----
und = cv2.undistortPoints(img_t.reshape(-1, 1, 2), K_true, dist_true, P=K_true).reshape(-1, 2)
clean, _ = cv2.projectPoints(objp, rvec_t, tvec_t, K_true, np.zeros(5))
clean = clean.reshape(-1, 2)
residual = np.abs(und - clean).mean()
print(f"\nundistort residual: {residual:.4f} px")
assert residual < 0.5

plt.figure(figsize=(9, 6))
plt.imshow(canvas[..., ::-1])
plt.title("solvePnP pose: projected corners + 3D axes")
plt.axis("off")
plt.tight_layout()
plt.savefig("ex06.png", dpi=110)
print("saved ex06.png")

# ---- 6. Real photos: the same pipeline, with detection instead of simulation ----
def corners_from_photo(path, rows=6, cols=9):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    if not ok:
        return None
    return cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

# with real photos, swap the synthetic loop for:
#   imgpoints.append(corners_from_photo(p)); objpoints.append(objp)
# keeping only views where corners_from_photo returns non-None.
```

### What's happening

- **Pinhole model**: `p = K [R|t] P`. `K` holds focal lengths and the principal point. Calibration solves for `K`, distortion, and one pose per view, minimizing reprojection error.
- **Reprojection error** (RMS) is *the* calibration metric: how many pixels the model is off, on average, on the training views. Sub-1 px is good; over 2 px means bad detections.
- **`solvePnP`** needs ≥ 4 correspondences with known 3D coordinates. Same math as calibration, one view, pose only.
- **Distortion**: radial (k1, k2, k3) bends straight lines; tangential (p1, p2) comes from lens/sensor misalignment. `undistortPoints` inverts it.
- The principal point usually lands near the image center; a wildly off-center estimate is a red flag for bad data.

### Stretch

- Compute stereo depth: make two cameras with different `tvec`, project the same 3D point, then triangulate with `cv2.triangulatePoints` and compare with ground truth.
- Calibrate with fewer views (5, 8, 12) and watch the estimate wobble — that's the argument for 15–25 diverse views in practice.
</details>
