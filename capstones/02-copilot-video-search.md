# Capstone 2 — OpenEye Copilot: Natural-Language Video Search & Q&A

**What you ship:** a system where you type *"a person crossing the street"* and get the matching moments from hours of footage — plus an optional vision-LLM answer about the top hit. This is the architecture behind every "search your footage with words" product.

```
video ─► keyframes (every N s) ─► CLIP image embeddings ─► index (npy + json)
                                                                  ▲
        query text ─► CLIP text embedding ─► cosine similarity ────┘
                │
                └─► top-k frames ─► timeline UI ─► (optional) vision LLM answer
```

**Files you will create:**

```
copilot/
  embedder.py    # CLIP wrapper: images + text -> one embedding space
  indexer.py     # video -> keyframes -> embeddings (+ offline synthetic mode)
  search.py      # VideoIndex.search(query) + CLI
  app.py         # Gradio search UI
out/index/
  frames/*.jpg   embeddings.npy   frames.json
```

---

## Step 1 — Setup

```bash
pip install -r requirements-advanced.txt
```

First real run downloads CLIP ViT-B/32 (~600 MB) from Hugging Face. Everything after that is local and offline.

## Step 2 — The embedder (`copilot/embedder.py`)

One class, two methods. The entire trick of multimodal search is that image and text encoders live in the *same* vector space, so "closeness" between a sentence and a frame is just a dot product.

```python
"""OpenEye Copilot — CLIP wrapper: images and text into one embedding space.

First use downloads openai/clip-vit-base-patch32 (~600 MB) from Hugging Face.
GPU not required; CPU embedding of a keyframe takes ~30 ms.
"""
import cv2
import numpy as np
import torch
from PIL import Image


class ClipEmbedder:
    def __init__(self, name="openai/clip-vit-base-patch32"):
        # imported here so the CLIs (--help, --index-only runs) work without transformers
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(name).eval()
        self.processor = CLIPProcessor.from_pretrained(name)

    @torch.no_grad()
    def images(self, frames_bgr):
        """List of BGR uint8 frames -> (N, D) L2-normalized embeddings."""
        pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        inputs = self.processor(images=pil, return_tensors="pt")
        emb = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(emb, dim=-1).numpy()

    @torch.no_grad()
    def texts(self, queries):
        """List of strings -> (N, D) L2-normalized embeddings."""
        inputs = self.processor(text=list(queries), return_tensors="pt", padding=True)
        emb = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(emb, dim=-1).numpy()
```

**Why L2-normalize?** After normalization, cosine similarity = dot product. One matrix multiply retrieves from the whole index — no distance code, no faiss needed until you pass ~10⁶ frames.

## Step 3 — Keyframes + index (`copilot/indexer.py`)

Don't embed every frame: sample one keyframe every ~2 seconds (scene-change sampling is a stretch goal). Store three things: JPEGs, embeddings, and a metadata list that maps row *i* of the embedding matrix back to *(time, file)*. Keep that mapping in sync or your search returns the wrong timestamps forever.

```python
"""OpenEye Copilot — build a searchable embedding index from a video.

Run:
    python copilot/indexer.py --video clip.mp4 --every 2 --out out/index
    python copilot/indexer.py --synthetic --out out/index        # self-test, no video needed

Output in <out>/:
    frames/*.jpg      sampled keyframes
    embeddings.npy    (N, D) float32, L2-normalized
    frames.json       [{idx, t, path}, ...]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from embedder import ClipEmbedder


def sample_frames(video_path, every_s, max_frames):
    """One keyframe every `every_s` seconds. Scene-change sampling is a stretch goal."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps * every_s))
    i = n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i, i / fps, frame
            n += 1
        i += 1
    cap.release()


def synthetic_frames(n=40, w=320, h=240):
    """Deterministic 'footage' with known ground truth for offline testing:
    five colored scenes, each color exactly n/5 = 8 frames."""
    rng = np.random.default_rng(0)
    colors = {"red": (0, 0, 220), "green": (0, 200, 0), "blue": (220, 0, 0),
              "yellow": (0, 220, 220), "purple": (200, 0, 160)}
    names = list(colors)
    for i in range(n):
        name = names[i % len(names)]
        frame = np.full((h, w, 3), 40.0, np.float32)
        frame = cv2.addWeighted(frame, 0.75, np.full_like(frame, colors[name]), 0.25, 0)
        cv2.putText(frame, name, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        yield i, i * 2.0, np.clip(frame, 0, 255).astype(np.uint8)


def build_index(out_dir, items, embedder, chunk=16):
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    meta, frames = [], []
    for idx, t, frame in items:
        rel = f"frames/{idx:06d}.jpg"
        cv2.imwrite(str(out / rel), frame)
        meta.append({"idx": idx, "t": round(float(t), 2), "path": rel})
        frames.append(frame)
    embs = [embedder.images(frames[i:i + chunk]) for i in range(0, len(frames), chunk)]
    emb = np.concatenate(embs) if embs else np.zeros((0, 512), np.float32)
    np.save(out / "embeddings.npy", emb)
    (out / "frames.json").write_text(json.dumps(meta, indent=2))
    return len(meta), emb.shape[1]


def main():
    ap = argparse.ArgumentParser(description="OpenEye Copilot indexer")
    ap.add_argument("--video", help="video file to index")
    ap.add_argument("--synthetic", action="store_true", help="no video needed")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between keyframes")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--out", default="out/index")
    args = ap.parse_args()
    if not args.video and not args.synthetic:
        raise SystemExit("pass --video clip.mp4 or --synthetic")

    items = list(synthetic_frames() if args.synthetic
                 else sample_frames(args.video, args.every, args.max_frames))
    embedder = ClipEmbedder()
    count, dim = build_index(args.out, items, embedder)
    print(f"indexed {count} frames (dim {dim}) -> {args.out}")


if __name__ == "__main__":
    main()
```

Embedding in chunks of 16 bounds memory; the JPEGs are your thumbnails *and* your UI images *and* your VLM inputs later. One artifact, three uses.

## Step 4 — Search (`copilot/search.py`)

```python
"""OpenEye Copilot — natural-language search over an index built by indexer.py.

    python copilot/search.py "a person crossing the street" --index out/index
"""
import argparse
import json
from pathlib import Path

import numpy as np

from embedder import ClipEmbedder


class VideoIndex:
    """Cosine similarity is just a dot product — both sides are L2-normalized."""

    def __init__(self, index_dir):
        self.dir = Path(index_dir)
        self.emb = np.load(self.dir / "embeddings.npy")
        self.meta = json.loads((self.dir / "frames.json").read_text())
        assert len(self.meta) == len(self.emb), "index files are out of sync — rebuild"

    def search(self, embedder, query, k=6):
        q = embedder.texts([query])[0]                  # (D,)
        sims = self.emb @ q                             # (N,)
        order = np.argsort(sims)[::-1][:k]
        return [{"idx": self.meta[i]["idx"], "t": self.meta[i]["t"], "score": float(sims[i]),
                 "path": str(self.dir / self.meta[i]["path"])} for i in order]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--index", default="../out/index")
    ap.add_argument("-k", type=int, default=6)
    args = ap.parse_args()

    index = VideoIndex(args.index)
    embedder = ClipEmbedder()
    for r in index.search(embedder, args.query, args.k):
        print(f"{r['t']:7.1f}s   score {r['score']:.3f}   {r['path']}")
```

The `assert` guarding metadata/embedding alignment has caught more real bugs in production than any unit test I've written: append one frame to the index without rebuilding and every timestamp after it silently shifts.

## Step 5 — Build an index and search it

```bash
cd copilot
python indexer.py --synthetic --out ../out/index      # ~20 s incl. model load
python search.py "a red scene" --index ../out/index -k 5
```

Expected: five hits, all `red` frames, timestamps spread across the clip. Then try real footage:

```bash
python indexer.py --video ../clip.mp4 --every 2 --out ../out/index
python search.py "a person walking a dog" --index ../out/index
```

Keep queries **visual and concrete** ("a red car", "someone in a yellow jacket", "a doorway at night") — CLIP is bad at negation, counting, and abstractions. That isn't a bug you can prompt-engineer away; it's the training objective.

## Step 6 — Verify with a ground-truth check

The synthetic clip knows the answer: frame `idx` is `NAMES[idx % 5]`. That turns "looks right" into recall@5, a real metric.

```python
"""Run from copilot/:  python -c "exec(open('eval_snippet.py').read())"  — or paste into a REPL."""
import numpy as np
from embedder import ClipEmbedder
from search import VideoIndex

NAMES = ["red", "green", "blue", "yellow", "purple"]     # matches indexer.synthetic_frames

index = VideoIndex("../out/index")
embedder = ClipEmbedder()

scores = []
for query, want in [("a red scene", "red"), ("a green scene", "green"),
                    ("a blue scene", "blue"), ("a yellow scene", "yellow"),
                    ("a purple scene", "purple")]:
    results = index.search(embedder, query, k=5)
    hits = sum(NAMES[r["idx"] % 5] == want for r in results)
    scores.append(hits / 5)
    print(f"{query:20s} recall@5 = {hits}/5")

mean = float(np.mean(scores))
print(f"mean recall@5: {mean:.2f}   (random baseline = 0.20)")
assert mean > 0.6, "index or embeddings look broken"
```

For real footage, label ~20 frames with captions by hand and compute recall@5 the same way. That's the honest number; anything else is demo-ware.

## Step 7 — UI (`copilot/app.py`)

```python
"""OpenEye Copilot — Gradio search UI. Run: python copilot/app.py --index out/index"""
import argparse

import gradio as gr

from embedder import ClipEmbedder
from search import VideoIndex

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="../out/index")
args = ap.parse_args()

index = VideoIndex(args.index)
embedder = ClipEmbedder()


def run_query(query, k):
    if not query.strip():
        return []
    results = index.search(embedder, query, int(k))
    return [(r["path"], f"t = {r['t']:.1f}s   score {r['score']:.3f}") for r in results]


with gr.Blocks(title="OpenEye Copilot") as demo:
    gr.Markdown("# OpenEye Copilot\nAsk your footage in plain English.")
    query = gr.Textbox(label="Query", placeholder="a person crossing the street")
    k = gr.Slider(1, 12, value=6, step=1, label="Results")
    out = gr.Gallery(label="Top matches", columns=3)
    query.submit(run_query, [query, k], out)
    gr.Examples([["a red scene"], ["a blue scene"], ["a green scene"]], inputs=query)

demo.launch()
```

```bash
cd copilot && python app.py --index ../out/index
```

## Step 8 — Optional: vision-LLM answers over the top hit

Retrieval gives you *where*; a vision LLM answers *what happened there*. Any OpenAI-compatible endpoint works — here, AI Gateway, which routes to most providers behind one key (`AI_GATEWAY_API_KEY`) and one URL:

```python
import base64
import json
import os
import urllib.request
from pathlib import Path


def describe(image_path, question, model=None):
    """Ask any vision LLM about a frame. Set VLM_MODEL to any provider/model id you have access to."""
    model = model or os.environ.get("VLM_MODEL", "openai/gpt-4o-mini")
    img = base64.b64encode(Path(image_path).read_bytes()).decode()
    body = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
            ],
        }],
    }).encode()
    req = urllib.request.Request(
        "https://ai-gateway.vercel.sh/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {os.environ['AI_GATEWAY_API_KEY']}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


# usage:
# top = index.search(embedder, "a red scene", k=1)[0]
# print(describe(top["path"], "Describe this scene in one sentence."))
```

More frames per answer beat one frame: send the top-3 as multiple `image_url` blocks and ask *"what changes across these frames?"* — that's video Q&A with a still-image model.

## Step 9 — Optional: deploy the search box

The index is two small files plus JPEGs, so deployment is genuinely simple:

1. `vercel blob` (or S3) to host `embeddings.npy`, `frames.json`, and the thumbnails.
2. A Next.js route that loads the index and runs `emb @ text_embedding` — for the text embedding, either precompute offline or call an embeddings endpoint through AI Gateway; the VLM question from step 8 goes through the same key.
3. Build the UI in the browser: query box, thumbnail timeline, click-to-seek.

Keep the heavy lifting (embedding hours of footage) offline; only search and the VLM calls belong in a request path.

## Step 10 — Limits and how to grow this

| Limitation | Why | Upgrade |
|---|---|---|
| Misses small objects | CLIP embeds the whole frame | crop with the Sentinel detector first, embed crops |
| Bad at "two people", "not red", reading text | CLIP training objective is global image–text alignment | add OCR / VLM captions per frame; query the captions |
| Memory grows with N frames | `embeddings.npy` is dense | faiss / hnswlib approximate search above ~10⁶ frames |
| Whole-frame matching loses temporal info | it's per-frame retrieval | temporal windows: embed 4-frame stacks, or aggregate top-k into "moments" |
| Every redeployment re-embeds if bytes change | JPEG re-encode is lossy | store raw frames or hash-stable encoding if you compare across builds |

## Stretch goals

- **Fuse with Sentinel (capstone 1):** filter search results by events — "people who entered the left lane after 14:00" = zone_enter events ∩ visual search results.
- **Audio track:** transcribe with `faster-whisper`, index text alongside frames, retrieve by dialogue or visual similarity from one query bar.
- **Caption-assisted search:** run a VLM over keyframes offline to write one caption each; embed captions with a text model; retrieval quality jumps on counting/relations.
- **Face clusters:** detect faces (exercise 7), cluster embeddings, rename "John" — then search "John near the door".
- **Better encoder:** swap in SigLIP or a Chinese-CLIP variant; the index format doesn't change, only `embedder.py`'s model name.
