# Exercise 10 — CLIP: Contrastive Training, Zero-Shot & Its Limits

**Goal:** understand how image–text models work by building a tiny CLIP from scratch — and measure exactly which kind of zero-shot generalization works at small scale, and which doesn't.
**Concepts:** contrastive learning (InfoNCE), joint embedding space, zero-shot classification, attribute transfer vs. compositional generalization, patch attention maps.
**Needs:** `torch` (advanced track); optional `transformers` for the real-CLIP appendix. **Time:** ~45 min.

## Tasks

1. **Tiny CLIP from scratch** (training takes ~10 s on CPU):
   - synthetic images: one shape (circle/square/triangle) in one color (red/green/blue), noisy 32×32, captions like `"a red circle"` — 9 combinations total
   - image encoder: small CNN → 64-d L2-normalized embedding; text encoder: word-embedding bag + MLP → same space
   - train with **symmetric InfoNCE** loss and a learned temperature, one image per training caption per step, 1500 steps
   - **hold out 3 (color, shape) combinations** — never seen in training
2. Measure three things, all with embeddings only (no classifier head, no new labels):
   - **(a) Retrieval**: fresh images of *trained* combinations, ranked against all 9 captions. Assert top-1 > 80%. (This verifies contrastive training actually aligned the modalities.)
   - **(b) Zero-shot attribute transfer**: images of *unseen* combinations — does the top-1 caption at least get the **color** right? Assert > 80%.
   - **(c) Joint zero-shot**: top-1 caption exactly correct for unseen combinations. Print it — this is the interesting part.
3. Visualize: one unseen image against all 9 caption similarities (you'll see its color is right, its shape is not).
4. **Appendix (optional, needs a ~600 MB CLIP download):** *real* CLIP for comparison — zero-shot prompts, retrieval, and a patch-similarity heatmap. Enable with `RUN_REAL_CLIP=1`. Real CLIP gets (c) right, and running both is the whole lesson.

> Contrastive training in one sentence: push each image's embedding toward its own caption and away from every other caption in the batch — the batch *is* the negatives, no labels needed.

<details>
<summary>Solution</summary>

```python
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
rng = np.random.default_rng(0)

SHAPES = ["circle", "square", "triangle"]
COLORS = {"red": (0.90, 0.15, 0.15), "green": (0.15, 0.80, 0.20), "blue": (0.20, 0.30, 0.90)}
SIZE = 32
HELD_OUT = {("red", "triangle"), ("green", "square"), ("blue", "circle")}

# ---- 1a. Synthetic image renderer ----
def shape_mask(shape):
    m = np.zeros((SIZE, SIZE), np.uint8)
    c = (SIZE // 2, SIZE // 2)
    if shape == "circle":
        cv2.circle(m, c, 9, 1, -1)
    elif shape == "square":
        cv2.rectangle(m, (c[0] - 8, c[1] - 8), (c[0] + 8, c[1] + 8), 1, -1)
    else:
        pts = np.array([[c[0], c[1] - 10], [c[0] - 10, c[1] + 9], [c[0] + 10, c[1] + 9]], np.int32)
        cv2.fillPoly(m, [pts], 1)
    return (m > 0).astype(np.float32)[..., None]

def render(shape, color, rng):
    img = np.full((SIZE, SIZE, 3), 0.15, np.float32)
    img = img + rng.normal(0, 0.05, img.shape).astype(np.float32)
    col = np.asarray(COLORS[color], np.float32) + rng.normal(0, 0.04, 3).astype(np.float32)
    m = shape_mask(shape)
    return np.clip(img * (1 - m) + col[None, None, :] * m, 0, 1)

# ---- 1b. Model ----
VOCAB = ["a", "red", "green", "blue", "circle", "square", "triangle"]
W2I = {w: i for i, w in enumerate(VOCAB)}

def tokenize(caption):
    return torch.tensor([W2I[w] for w in caption.split()])

class TinyCLIP(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 4 * 4, dim))
        self.text_embed = nn.Embedding(len(VOCAB), dim)
        self.text_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))    # learned temperature

    def encode_image(self, x):
        return F.normalize(self.visual(x), dim=-1)

    def encode_text(self, ids):
        pooled = self.text_embed(ids).mean(dim=1)             # bag of words
        return F.normalize(self.text_mlp(pooled), dim=-1)

def clip_loss(img_emb, txt_emb, scale):
    logits = scale * img_emb @ txt_emb.T                      # (B, B) similarity matrix
    labels = torch.arange(len(img_emb))
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

# ---- 1c. Train: one image per train combo per step (unique captions per batch) ----
train_combos = [(c, s) for c in COLORS for s in SHAPES if (c, s) not in HELD_OUT]
model = TinyCLIP()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
txt_ids = torch.stack([tokenize(f"a {c} {s}") for c, s in train_combos])

for step in range(1500):
    imgs = np.stack([render(s, c, rng) for c, s in train_combos])
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    loss = clip_loss(model.encode_image(x), model.encode_text(txt_ids),
                     model.logit_scale.exp())
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (step + 1) % 500 == 0:
        print(f"step {step + 1}: loss {loss.item():.4f}, scale {model.logit_scale.exp().item():.1f}")

# ---- 2. Evaluate with embeddings only ----
all_captions = [f"a {c} {s}" for c in COLORS for s in SHAPES]
model.eval()

def caption_for(img):
    x = torch.from_numpy(img).permute(2, 0, 1)[None]
    with torch.no_grad():
        sims = model.encode_image(x) @ txt_all.T
    return all_captions[int(sims[0].argmax())]

with torch.no_grad():
    txt_all = model.encode_text(torch.stack([tokenize(c) for c in all_captions]))

# (a) retrieval on fresh images of TRAINED combinations
hits = total = 0
for c, s in train_combos:
    for _ in range(10):
        hits += caption_for(render(s, c, rng)) == f"a {c} {s}"
        total += 1
retrieval = hits / total
print(f"(a) retrieval top-1, trained combos:      {retrieval:.1%}")
assert retrieval > 0.8, "contrastive training didn't align the modalities"

# (b) zero-shot COLOR transfer on UNSEEN combinations
hits = total = 0
for color, shape in HELD_OUT:
    for _ in range(10):
        hits += caption_for(render(shape, color, rng)).split()[1] == color
        total += 1
color_acc = hits / total
print(f"(b) zero-shot color acc, unseen combos:   {color_acc:.1%}")
assert color_acc > 0.8, "attribute transfer failed — train longer"

# (c) joint zero-shot on UNSEEN combinations (the hard part)
hits = total = 0
for color, shape in HELD_OUT:
    for _ in range(10):
        hits += caption_for(render(shape, color, rng)) == f"a {color} {shape}"
        total += 1
print(f"(c) joint zero-shot, unseen combos:       {hits / total:.1%}  <- read the notes below")

# ---- 3. Visualize: where does an unseen image land? ----
import matplotlib.pyplot as plt
for j, (color, shape) in enumerate(sorted(HELD_OUT)):
    img = render(shape, color, rng)
    x = torch.from_numpy(img).permute(2, 0, 1)[None]
    with torch.no_grad():
        sims = (model.encode_image(x) @ txt_all.T)[0].numpy()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1, 3]})
    ax[0].imshow(img)
    ax[0].set_title(f"unseen: a {color} {shape}")
    ax[0].axis("off")
    ax[1].bar(range(9), sims)
    ax[1].set_xticks(range(9))
    ax[1].set_xticklabels(all_captions, rotation=45, ha="right")
    ax[1].set_ylabel("cosine similarity")
    plt.tight_layout()
    plt.savefig(f"ex10_zeroshot_{j}.png", dpi=110)
print("saved ex10_zeroshot_*.png")

# ---- 4. Appendix: the real thing ----
import os
if os.environ.get("RUN_REAL_CLIP") == "1":
    try:
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        name = "openai/clip-vit-base-patch32"      # ~600 MB download, once
        clip = CLIPModel.from_pretrained(name).eval()
        proc = CLIPProcessor.from_pretrained(name)

        photo = Image.open(cv2.samples.findFile("lena.jpg")).convert("RGB")
        prompts = ["a photo of a face", "a photo of a hat", "a photo of a cat", "a photo of a car"]
        inputs = proc(text=prompts, images=photo, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs = clip(**inputs).logits_per_image[0].softmax(-1)
        for p, s in zip(prompts, probs):
            print(f"{p:28s} {s:.3f}")

        # patch attention map: cosine similarity of each patch token to the mean patch
        inputs = proc(images=photo, return_tensors="pt")
        with torch.no_grad():
            vision = clip.vision_model(pixel_values=inputs["pixel_values"],
                                       output_hidden_states=True)
        tokens = vision.last_hidden_state[0]        # (1 + 49, D): CLS + 7x7 patches
        patches = tokens[1:]
        patches = patches / patches.norm(dim=-1, keepdim=True)
        sim = (patches @ patches.mean(0)).reshape(7, 7).numpy()
        heat = (sim - sim.min()) / (sim.ptp() + 1e-9)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1); plt.imshow(photo); plt.axis("off"); plt.title("input")
        plt.subplot(1, 2, 2)
        plt.imshow(photo.resize((224, 224)))
        plt.imshow(cv2.resize(heat, (224, 224)), cmap="jet", alpha=0.5)
        plt.axis("off"); plt.title("patch similarity map")
        plt.tight_layout(); plt.savefig("ex10_real_clip.png", dpi=110)
        print("saved ex10_real_clip.png")
    except Exception as e:
        print("real-CLIP appendix skipped:", e)
else:
    print("Set RUN_REAL_CLIP=1 to run the real-CLIP appendix (downloads ~600 MB).")
```

Expected output (verified):

```text
step 500: loss 0.0052, scale 5.7
step 1000: loss 0.0012, scale 7.0
step 1500: loss 0.0005, scale 7.7
(a) retrieval top-1, trained combos:      100.0%
(b) zero-shot color acc, unseen combos:   100.0%
(c) joint zero-shot, unseen combos:       0.0%  <- read the notes below
```

### What's happening

- **Joint embedding space**: image and text encoders are trained so matching pairs point the same way. Classification = nearest caption by cosine similarity — no classifier head, no per-class labels.
- **InfoNCE**: a batch of B (image, caption) pairs yields a B×B similarity matrix. The diagonal is the positive; every off-diagonal is a negative. Cross-entropy on both rows and columns (symmetric) pushes the diagonal up. Real CLIP uses the same loss with 32k-pair batches.
- **Learned temperature** (`logit_scale`): a single parameter controlling how peaked the softmax is (it grows from 2.7 to ~5.7 here).
- **Why (a) and (b) succeed**: color is a large, low-frequency visual property, so the image encoder learns it robustly, *independently of the shape it appears on*. The unseen blue circle image is correctly recognized as blue because "blue" was learned as a concept, not as one of two memorized pictures.
- **Why (c) fails at this scale** — this is the real lesson. With 6 training combinations, the image encoder can reach near-zero loss by memorizing a **cluster per combination**; nothing forces it to factorize the image into `color ⊗ shape`. The text side is inherently compositional (word embeddings), but for an unseen combination the image side has no matching "slot" to land in. Compositional generalization in CLIP is a *data and scale* effect (400M pairs, huge batch of negatives), not an automatic property of the architecture.
- **If you want (c) to work at small scale**, the levers are: more combinations (coverage of the attribute product), multiple augmentations per caption with a multi-positive loss, larger batches (more negatives), more capacity, and longer training. Try them — watching (c) climb from 0% is the best mini-research project in this course.
- **It's already a product**: (a) is retrieval ("find the frame matching this description"), (b) is attribute-based zero-shot tagging ("this is blue") — both work with 10 seconds of training.

### Stretch

- Add a `"big"`/`"small"` attribute (18 combos, 12 trained). Does (c) improve with a third attribute increasing the pressure to factorize?
- Implement multi-positive InfoNCE (2–4 augmentations per caption per step) and measure (c) again.
- Compare with the real CLIP appendix: run the same "unseen combination" test on real photos with seen/new prompt pairs — real CLIP gets joint compositions right. That gap *is* the field of multimodal representation learning.
- Remove the L2 normalization from the encoders and watch the loss collapse — then explain why normalization matters.
</details>
