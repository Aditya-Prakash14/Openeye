# Exercise 7 — Face Detection & Eigenfaces Recognition

**Goal:** the classic two-stage face pipeline — detect faces, then recognize *who* they are — including the math behind "eigenfaces".
**Concepts:** Haar cascade detection, PCA, dimensionality reduction, nearest-neighbor classification, reconstruction error.
**Needs:** core track. **Time:** ~30 min.

## Tasks

1. Detect faces in an image with OpenCV's bundled Haar cascade. Optionally load a real photo if `cv2.samples` can download one; otherwise apply the detector code to your own images.
2. Build the recognition pipeline:
   - split the **Olivetti faces** dataset into train/test — if the download is unavailable, the code falls back to sklearn's bundled digits so the pipeline still runs offline
   - fit `PCA(n_components=50, whiten=True)` on **train only**
   - classify test images with 1-nearest-neighbor in PCA space
   - assert test accuracy > 80%
3. Visualize the mean face + first 5 eigenfaces; show test predictions vs. truth.
4. Reconstruction demo: reconstruct one test image with 2, 5, 10, 25, 50 components. Assert reconstruction error decreases with more components.

> Why PCA? Every 64×64 face is a 4096-dim vector, but faces live on a tiny subspace ("face space"). The eigenvectors of that subspace are the eigenfaces. This is the 1991 algorithm, and it's still the cleanest way to understand embedding-based recognition.

<details>
<summary>Solution</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces, load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ---- 1. Detection (classic Haar cascade, bundled with OpenCV) ----
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
assert not cascade.empty(), "cascade file missing from this OpenCV build"

def detect_and_draw(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img, faces

try:                                    # downloads lena.jpg once
    photo = cv2.imread(cv2.samples.findFile("lena.jpg"))
    out, faces = detect_and_draw(photo)
    print("faces detected in lena.jpg:", len(faces))
    cv2.imwrite("ex07_detection.png", out)
except cv2.error:
    print("sample photo unavailable — call detect_and_draw(your_image) instead")

# ---- 2. Recognition data: Olivetti if reachable, digits otherwise ----
try:
    d = fetch_olivetti_faces()
    X, y, source = d.images, d.target, "olivetti faces (40 people)"
except Exception:
    d = load_digits()
    X = (d.images / 16.0).astype(np.float32)
    y, source = d.target, "digits (offline fallback)"
side = X.shape[1]                                   # images are (N, side, side)
print("dataset:", source, X.shape)

Xf = X.reshape(len(X), -1)
Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.3, random_state=0, stratify=y)

n_comp = min(50, Xtr.shape[0] - 1, Xtr.shape[1])
pca = PCA(n_components=n_comp, whiten=True).fit(Xtr)     # fit on TRAIN only!
Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
clf = KNeighborsClassifier(n_neighbors=1).fit(Ztr, ytr)
pred = clf.predict(Zte)
acc = accuracy_score(yte, pred)
print(f"recognition accuracy with {n_comp} components: {acc:.2%}")
assert acc > 0.8

# ---- 3. Eigenfaces ----
mean_face = pca.mean_.reshape(side, side)
fig, ax = plt.subplots(2, 6, figsize=(14, 5))
ax[0, 0].imshow(mean_face, cmap="gray")
ax[0, 0].set_title("mean face")
for i in range(5):
    ax[0, i + 1].imshow(pca.components_[i].reshape(side, side), cmap="gray")
    ax[0, i + 1].set_title(f"eigenface {i + 1}")
for i in range(6):
    ax[1, i].imshow(Xte[i].reshape(side, side), cmap="gray")
    ax[1, i].set_title(f"true {yte[i]} / pred {pred[i]}")
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
plt.savefig("ex07_eigenfaces.png", dpi=110)

# ---- 4. Reconstruction quality vs components ----
sample = Xte[0]
errors = []
for k in (2, 5, 10, 25, 50):
    k = min(k, n_comp)
    pk = PCA(n_components=k).fit(Xtr)
    rec = pk.inverse_transform(pk.transform(sample[None]))[0]
    errors.append(np.mean((rec - sample) ** 2))
print("reconstruction MSE for k = [2, 5, 10, 25, 50]:", [round(e, 4) for e in errors])
assert all(b < a for a, b in zip(errors, errors[1:])), "more components must never hurt"

fig2, ax2 = plt.subplots(1, 6, figsize=(14, 3))
ax2[0].imshow(sample.reshape(side, side), cmap="gray")
ax2[0].set_title("original")
for i, k in enumerate((2, 5, 10, 25, 50)):
    pk = PCA(n_components=min(k, n_comp)).fit(Xtr)
    rec = pk.inverse_transform(pk.transform(sample[None]))[0]
    ax2[i + 1].imshow(rec.reshape(side, side), cmap="gray")
    ax2[i + 1].set_title(f"k={k}")
for a in ax2:
    a.axis("off")
plt.tight_layout()
plt.savefig("ex07_reconstruction.png", dpi=110)
print("saved ex07_eigenfaces.png / ex07_reconstruction.png")
```

### What's happening

- **Haar cascade**: a cascade of boosted rectangle filters (Viola–Jones). Fast, bundled, works on frontal faces — but not on profile faces or heavy occlusion. Modern alternatives: a small CNN via `cv2.dnn` or a face detector from a learned library.
- **Eigenfaces = PCA on face vectors.** Fit on train only. The first components capture lighting/pose, later ones identity detail — which is why a handful of components already reconstructs a recognizable face.
- **Whiten=True** divides components by their singular values, which stops the first 1–2 dominant components from drowning out the others in L2 distance.
- **1-NN in embedding space** is the ancestor of modern face recognition: ArcFace/AdaFace do the same thing in a metric-learned embedding. The pipeline is identical, only the "encoder" changes.
- **Reconstruction error decreasing with `k`** is a one-line proof of what PCA is doing.

### Stretch

- Compare accuracy for k = 5, 10, 25, 50, 100 and plot it — find the elbow.
- Replace 1-NN with cosine distance (`KNeighborsClassifier(metric="cosine")`) and compare.
- Run the detection cascade at `scaleFactor=1.05, minNeighbors=3` — trade-off between missed faces and false boxes.
</details>
