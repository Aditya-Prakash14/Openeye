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
