# Exercise 8 — Detection, NMS & mAP — solution
# Generated from exercises/08-detection-nms-and-map.md by scripts/extract_solutions.py.
# Run:  python solutions/08-detection-nms-and-map.py   (writes output files to the cwd)

import cv2
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ---- 1. Scene, targets, distractors ----
H, W = 480, 640
canvas = rng.integers(60, 190, (H, W, 3), np.uint8)
canvas = cv2.GaussianBlur(canvas, (9, 9), 0)          # cluttered texture

tpl = np.full((40, 40, 3), 235, np.uint8)              # the target: bright square patch
cv2.rectangle(tpl, (12, 12), (27, 27), (30, 30, 30), -1)   # dark inset = its signature

def paste(canvas, tpl, center, scale, angle):
    h, w = tpl.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    size = int(np.ceil(np.hypot(h, w) * scale)) + 2
    M[0, 2] += (size - w) / 2
    M[1, 2] += (size - h) / 2
    obj = cv2.warpAffine(tpl, M, (size, size), borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (size, size),
                          flags=cv2.INTER_NEAREST)
    x0, y0 = int(center[0] - size / 2), int(center[1] - size / 2)
    roi = canvas[y0:y0 + size, x0:x0 + size]
    roi[mask > 0] = obj[mask > 0]
    ys, xs = np.where(mask > 0)                        # tight box of the pasted object
    return (float(x0 + xs.min()), float(y0 + ys.min()),
            float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))

gt = [paste(canvas, tpl, c, s, a) for c, s, a in
      [((110, 120), 1.0, 0), ((320, 90), 0.7, -8), ((480, 160), 1.45, 8),
       ((200, 350), 1.15, -6), ((430, 380), 0.85, 6), ((90, 260), 1.3, 4)]]

for cx, cy in [(250, 220), (560, 300), (150, 420)]:      # distractors (must NOT fire)
    cv2.circle(canvas, (cx, cy), 28, (30, 30, 30), -1)               # dark disc...
    cv2.line(canvas, (cx - 20, cy), (cx + 20, cy), (240, 240, 240), 6)  # ...with a bright bar

# ---- 2. Multi-scale template-matching detector ----
scene_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

candidates = []          # (score, (x, y, w, h))
for scale in np.linspace(0.55, 1.6, 8):
    t = cv2.resize(tpl_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if min(t.shape) < 8 or t.shape[0] >= H or t.shape[1] >= W:
        continue
    resp = cv2.matchTemplate(scene_gray, t, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(resp > 0.6)
    scores = resp[ys, xs]
    for i in np.argsort(scores)[-80:]:                  # cap candidates per scale
        candidates.append((float(scores[i]),
                           (float(xs[i]), float(ys[i]), float(t.shape[1]), float(t.shape[0]))))
print("candidates before NMS:", len(candidates))

# ---- 3. IoU + NMS (own implementation) ----
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (aw * ah + bw * bh - inter + 1e-9)

def nms(boxes, scores, thr=0.3):
    order = list(np.argsort(scores)[::-1])
    keep = []
    while order:
        i = order[0]
        keep.append(i)
        order = [j for j in order[1:] if iou(boxes[i], boxes[j]) < thr]
    return keep

order = np.argsort([s for s, _ in candidates])[::-1]
boxes_all = [b for _, b in candidates]
scores_all = np.array([s for s, _ in candidates])
keep = nms(boxes_all, scores_all, thr=0.3)
dets = [(float(scores_all[i]), boxes_all[i]) for i in keep]
print("detections after NMS:", len(dets))

# ---- 4. Evaluation: greedy matching + all-point AP ----
def average_precision(dets, gts, iou_thr=0.5):
    dets = sorted(dets, key=lambda d: -d[0])            # highest score first
    matched = np.zeros(len(gts), bool)
    tp, fp = [], []
    for _, box in dets:
        best, best_iou = -1, iou_thr
        for i, g in enumerate(gts):
            if matched[i]:
                continue
            v = iou(box, g)
            if v >= best_iou:
                best, best_iou = i, v
        if best >= 0:
            matched[best] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp / len(gts)
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]      # make PR monotonic
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))

ap = average_precision(dets, gt, iou_thr=0.5)
print(f"AP@0.5: {ap:.3f}  (ground-truth objects: {len(gt)}, detections: {len(dets)})")
assert ap > 0.8, "detector too weak — check thresholds or texture"

# ---- 5. Visualize ----
vis = canvas.copy()
for (x, y, w, h) in gt:
    cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
for score, (x, y, w, h) in dets:
    cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
    cv2.putText(vis, f"{score:.2f}", (int(x), int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

plt.figure(figsize=(11, 8))
plt.imshow(vis[..., ::-1])
plt.title(f"green = ground truth, red = detections, AP@0.5 = {ap:.2f}")
plt.axis("off")
plt.tight_layout()
plt.savefig("ex08.png", dpi=110)
print("saved ex08.png")

# ---- 6. Appendix: the same eval harness, applied to YOLO ----
try:
    from ultralytics import YOLO                     # optional dependency
    model = YOLO("yolov8n.pt")                       # downloads ~6 MB once
    r = model.predict(cv2.samples.findFile("bus.jpg"), conf=0.35, verbose=False)[0]
    yolo_dets = [(float(c), (float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
                 for (x1, y1, x2, y2, c) in r.boxes.data[:, :5].tolist()]
    print(f"YOLO found {len(yolo_dets)} objects (COCO classes) — "
          "annotate a few as ground truth and this harness scores them the same way.")
except Exception as e:
    print("YOLO appendix skipped:", e)
