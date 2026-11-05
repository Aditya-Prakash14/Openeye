# Exercise 2 — Convolution & Filtering

**Goal:** implement 2D convolution yourself, then use it to blur, sharpen, and find edges.
**Concepts:** kernels, correlation vs convolution, padding, separability, denoising trade-offs.
**Needs:** core track. **Time:** ~25 min.

## Tasks

1. Implement `convolve_naive(img, kernel)` with two Python loops and edge padding.
2. Implement the same thing vectorized with `np.lib.stride_tricks.sliding_window_view` and `np.einsum`.
3. Assert the two agree, and that both agree with `cv2.filter2D` **in the interior** of the image (borders differ because of padding rules).
4. Apply four kernels: box 5×5, Gaussian 5×5 (built from `cv2.getGaussianKernel`), sharpen, Sobel-X. Display results.
5. Add Gaussian noise to a synthetic edge image. Denoise with Gaussian blur, median blur, and bilateral filter. Compute PSNR for each and assert each filter improves on the noisy image.

> Hint: `cv2.filter2D` actually performs *correlation*, not convolution. For symmetric kernels they are identical — use symmetric kernels for the comparison and say so in a comment.

<details>
<summary>Solution</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Naive convolution (loops) ---
def convolve_naive(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img.astype(np.float32), ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros(img.shape, np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            out[y, x] = np.sum(padded[y:y + kh, x:x + kw] * kernel)
    return out

# --- 2. Vectorized convolution (sliding windows) ---
def convolve_sliding(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img.astype(np.float32), ((ph, ph), (pw, pw)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)

# --- 3. Cross-checks ---
K_BOX = np.ones((5, 5), np.float32) / 25.0          # symmetric: convolution == correlation
K_SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
K_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
g1d = cv2.getGaussianKernel(5, 1.2)
K_GAUSS = (g1d @ g1d.T).astype(np.float32)

img = np.zeros((200, 200), np.uint8)
img[:, :100] = 60
img[:, 100:] = 200                                   # a step edge
cv2.circle(img, (140, 100), 30, 240, -1)             # + a bright disc

a = convolve_naive(img, K_BOX)
b = convolve_sliding(img, K_BOX)
c = cv2.filter2D(img, cv2.CV_32F, K_BOX)             # float output: -1 would round back to uint8
assert np.allclose(a, b, atol=1e-3)                  # naive == vectorized
assert np.allclose(b[3:-3, 3:-3], c[3:-3, 3:-3], atol=1e-2)   # both == OpenCV in the interior

boxed = b
sharp = convolve_sliding(img, K_SHARPEN)
gauss = convolve_sliding(img, K_GAUSS)
sobel = convolve_sliding(img, K_SOBEL_X)

# --- 5. Denoising bake-off ---
rng = np.random.default_rng(0)
noisy = np.clip(img.astype(np.float32) + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-12))

denoised = {
    "gaussian": cv2.GaussianBlur(noisy, (5, 5), 1.5),
    "median": cv2.medianBlur(noisy, 5),
    "bilateral": cv2.bilateralFilter(noisy, 9, 60, 60),
}
scores = {k: psnr(img, v) for k, v in denoised.items()}
print("PSNR noisy:", round(psnr(img, noisy), 2))
for k, v in scores.items():
    print(f"PSNR {k}: {v:.2f}")
    assert v > psnr(img, noisy), f"{k} made it worse"

fig, ax = plt.subplots(2, 4, figsize=(14, 6))
for a_, im, t in zip(ax.ravel(),
                     [img, boxed, gauss, sharp, np.abs(sobel), noisy,
                      denoised["gaussian"], denoised["bilateral"]],
                     ["input", "box 5x5", "gaussian", "sharpen", "|sobel-x|", "noisy",
                      "gaussian blur", "bilateral"]):
    a_.imshow(im, cmap="gray")
    a_.set_title(t)
    a_.axis("off")
plt.tight_layout()
plt.savefig("ex02.png", dpi=110)
print("saved ex02.png")
```

### What's happening

- **Convolution/correlation** slide a kernel over the image: `out[y,x] = Σ kernel[i,j] · img[y+i,x+j]`. Convolution flips the kernel first; correlation doesn't. Most deep-learning "conv" layers are actually correlation — the distinction only matters for asymmetric kernels.
- **Padding** decides what happens at borders. `edge` replicates border pixels; OpenCV defaults to `BORDER_REFLECT_101`. That's why we compare only the interior.
- **`sliding_window_view`** creates a `(H, W, kh, kw)` view for free (no copy), so `einsum` can do the whole convolution in one vectorized pass.
- **Denoising**: Gaussian blur smooths edges; median removes salt-and-pepper while preserving edges; bilateral filters by *value distance* too, so it keeps edges sharpest. Compare PSNR to see the trade-off quantitatively.

### Stretch

- Build a 15×15 Gaussian as the outer product of two 1-D kernels and time it vs. the full 2-D version (separability).
- Try `cv2.Sobel` with `ksize=3` and compare signs — remember it correlates with a flipped kernel.
</details>
