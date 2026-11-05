# OpenEye — Computer Vision & AI, Hands-On

A complete course: **10 exercises with full solutions** and **2 advanced capstone projects** with step-by-step guides.

Every exercise runs **offline by default** (synthetic data, no downloads) so nothing blocks you, with optional real-data and pretrained-model tracks when you want them.

## What you need

- Python 3.10+, ~20 minutes per exercise
- Core track: `pip install -r requirements.txt`
- Advanced track (ex. 8–10 + capstones): `pip install -r requirements-advanced.txt`

## How to use this repo

1. Open an exercise file in `exercises/`.
2. Read **Tasks**, write code, run it.
3. Only then expand the **Solution** toggle — it has complete runnable code and notes.
4. Every solution ends with `assert`s so you *know* your result is correct.

Syntax-check every code block in the course with:

```bash
python scripts/check_solutions.py
```

## Curriculum

| # | Exercise | You build | Needs |
|---|----------|-----------|-------|
| 1 | [Image fundamentals](exercises/01-image-fundamentals.md) | Numpy/OpenCV basics: color spaces, geometry, histograms | core |
| 2 | [Convolution & filtering](exercises/02-convolution-and-filtering.md) | Your own `convolve2d`, blur/sharpen/Sobel, denoising | core |
| 3 | [Canny & Hough](exercises/03-canny-and-hough.md) | Canny edge detector from scratch, Hough line detection | core |
| 4 | [Contours & shapes](exercises/04-contours-and-shapes.md) | Shape classifier, coin counter, moments/centroids | core |
| 5 | [Features & panorama](exercises/05-feature-matching-and-panorama.md) | ORB + ratio test + RANSAC homography, stitching | core |
| 6 | [Calibration & pose](exercises/06-calibration-and-pose.md) | `calibrateCamera`, undistortion, `solvePnP` pose | core |
| 7 | [Faces](exercises/07-face-detection-and-eigenfaces.md) | Haar detection + PCA eigenfaces recognition | core |
| 8 | [Detection & mAP](exercises/08-detection-nms-and-map.md) | Multi-scale detector, NMS, AP evaluation (YOLO optional) | core |
| 9 | [Segmentation](exercises/09-unet-segmentation.md) | Train a U-Net, Dice/IoU metrics | torch |
| 10 | [CLIP & zero-shot](exercises/10-clip-zero-shot-and-attention.md) | Tiny CLIP from scratch, zero-shot, attention maps | torch |

## Capstones

| Project | What you ship |
|---------|---------------|
| [1. Sentinel — real-time tracking & zone analytics](capstones/01-sentinel-realtime-analytics.md) | Detection → tracking → polygon zones → speed → event stream → dashboard |
| [2. Copilot — natural-language video search](capstones/02-copilot-video-search.md) | Keyframes → CLIP embedding index → NL search → timeline UI → optional VLM Q&A |

Capstones compose: Copilot can filter its search by events produced by Sentinel.
Capstone 1 runs fully offline (synthetic scene, motion/HOG detectors). Capstone 2 needs the one-time CLIP download (~600 MB) for real embeddings, and its indexing/search plumbing can be exercised offline.

## Verified

Every exercise solution in this repo was executed end-to-end (macOS, Python 3.13, CPU) — each one ends in `assert`s you can re-run. Both capstone pipelines were smoke-tested: Sentinel produces events and an annotated video from its synthetic scene; Copilot's index/search plumbing was tested with a stub embedder. Re-check all code blocks any time:

```bash
python scripts/check_solutions.py
```

## Suggested pace (8 weeks)

| Week | Work |
|------|------|
| 1 | Exercises 1–2 + setup |
| 2 | Exercises 3–4 |
| 3 | Exercise 5 |
| 4 | Exercise 6 |
| 5 | Exercise 7 |
| 6 | Exercise 8 |
| 7 | Exercises 9–10 |
| 8 | Capstones 1 and 2 |

## Optional downloads (only when you want real data)

| What | Where |
|------|-------|
| Sample photos | `cv2.samples.findFile("lena.jpg")` auto-downloads from the OpenCV repo |
| Olivetti faces (ex. 7) | `sklearn.datasets.fetch_olivetti_faces()` (~4 MB) |
| YOLOv8-nano (ex. 8, capstone 1) | auto-downloads via `ultralytics` (~6 MB) |
| CLIP ViT-B/32 (ex. 10, capstone 2) | `openai/clip-vit-base-patch32` via `transformers` (~600 MB) |

Every exercise works without any of these.
