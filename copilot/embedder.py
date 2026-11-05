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
