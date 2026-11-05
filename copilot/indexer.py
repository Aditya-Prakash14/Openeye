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
