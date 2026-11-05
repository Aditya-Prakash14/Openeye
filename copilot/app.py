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
