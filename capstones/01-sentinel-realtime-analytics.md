# Capstone 1 — OpenEye Sentinel: Real-Time Tracking & Zone Analytics

**What you ship:** a complete video-analytics system. Point it at a camera or a video file and it detects people, tracks them with persistent IDs, counts entries/exits per zone, estimates speed, and writes a machine-readable event stream you can dashboard, query, and build business logic on top of.

This is exactly the shape of commercial "smart camera" products: *detection → tracking → geometry → events*.

```
 frames ──► Detector ──► IoU Tracker ──► Zones ──► Event log (JSONL) ──► Dashboard
                 (HOG/YOLO/motion)   (IDs, trails)   (enter/exit/dwell/speed)
```

**Files you will create:**

```
sentinel/
  detector.py    # 3 interchangeable detectors behind one interface
  tracker.py     # IoU tracker + speed from trajectories
  zones.py       # polygon zones
  pipeline.py    # the main loop — wires everything
  dashboard.py   # Streamlit view of the event stream
out/
  sentinel.mp4   # annotated video
  events.jsonl   # the event stream
```

---

## Step 1 — Setup

```bash
pip install -r requirements.txt          # core (motion + HOG detectors)
pip install ultralytics streamlit        # optional: YOLO detector + dashboard
```

## Step 2 — Understand the design

Everything is behind **one detector interface**:

```text
detector(frame) -> [(x1, y1, x2, y2, confidence), ...]
```

That single decision means the pipeline never knows — or cares — whether detections come from background subtraction, HOG, or YOLO. Swap models without touching tracking or analytics. Do this in every real system.

## Step 3 — Detectors (`sentinel/detector.py`)

Create `sentinel/detector.py`:

```python
"""OpenEye Sentinel — detectors.

Three interchangeable detectors behind one signature:  frame -> [(x1, y1, x2, y2, conf), ...]
"""
import cv2
import numpy as np


class MotionDetector:
    """Background subtraction. No model, no training. 40+ FPS, perfect for the synthetic self-test."""

    def __init__(self, min_area=400):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16)
        self.min_area = min_area
        self.kernel = np.ones((3, 3), np.uint8)

    def __call__(self, frame):
        mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            dets.append((float(x), float(y), float(x + w), float(y + h), 1.0))
        return dets


class HOGPeopleDetector:
    """Pedestrian detector, fully offline. ~2-6 FPS on CPU — fine to learn the pipeline."""

    def __init__(self, win_stride=(8, 8), scale=1.05, hit_threshold=0.3):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.win_stride = win_stride
        self.scale = scale
        self.hit_threshold = hit_threshold

    def __call__(self, frame):
        rects, weights = self.hog.detectMultiScale(
            frame, winStride=self.win_stride, padding=(8, 8), scale=self.scale)
        return [(float(x), float(y), float(x + w), float(y + h), float(c))
                for (x, y, w, h), c in zip(rects, weights) if c > self.hit_threshold]


class YOLODetector:
    """Learned detector (pip install ultralytics). COCO class 0 = person. Defaults to fast + accurate."""

    def __init__(self, weights="yolov8n.pt", conf=0.35, classes=(0,)):
        from ultralytics import YOLO
        self.model = YOLO(weights)          # downloads ~6 MB on first run
        self.conf = conf
        self.classes = list(classes)

    def __call__(self, frame):
        r = self.model.predict(frame, conf=self.conf, classes=self.classes, verbose=False)[0]
        return [(float(x1), float(y1), float(x2), float(y2), float(c))
                for x1, y1, x2, y2, c in r.boxes.data[:, :5].tolist()]
```

## Step 4 — The tracker (`sentinel/tracker.py`)

Detections are stateless; tracking assigns **persistent IDs across frames**. This is the minimal working version: greedy IoU association + track aging. No Kalman filter, on purpose — add one later if you need smoother trails.

```python
"""OpenEye Sentinel — minimal IoU tracker (SORT without the Kalman filter).

Greedy IoU association + track aging. O(n^2) matching is fine for tens of objects.
"""
import numpy as np


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def track_speed_kmh(track, meters_per_pixel, fps):
    """Speed from the last ~10 centroid displacements (px/frame) → km/h."""
    pts = track["trail"][-11:]
    if len(pts) < 2:
        return 0.0
    deltas = [np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    return float(np.mean(deltas)) * fps * meters_per_pixel * 3.6


class IoUTracker:
    def __init__(self, iou_thr=0.3, max_age=15, min_hits=2, trail_len=60):
        self.tracks = {}
        self.next_id = 1
        self.iou_thr = iou_thr
        self.max_age = max_age      # frames a track survives without a detection
        self.min_hits = min_hits    # hide flickery 1-frame tracks
        self.trail_len = trail_len

    def update(self, dets):
        unmatched = set(range(len(dets)))
        for tr in self.tracks.values():
            best, best_iou = -1, self.iou_thr
            for i in unmatched:
                v = iou(tr["box"], dets[i])
                if v > best_iou:
                    best, best_iou = i, v
            if best >= 0:
                tr["box"] = tuple(dets[best][:4])
                tr["age"] = 0
                tr["hits"] += 1
                cx = (tr["box"][0] + tr["box"][2]) / 2
                cy = (tr["box"][1] + tr["box"][3]) / 2
                tr["trail"].append((cx, cy))
                del tr["trail"][:-self.trail_len]
                unmatched.discard(best)
        for i in unmatched:
            box = tuple(dets[i][:4])
            self.tracks[self.next_id] = {
                "box": box, "age": 0, "hits": 1,
                "trail": [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)],
            }
            self.next_id += 1
        for tid in [t for t, tr in self.tracks.items() if tr["age"] > self.max_age]:
            del self.tracks[tid]
        for tr in self.tracks.values():
            tr["age"] += 1
        return self.tracks
```

**Key parameters to understand:**
- `iou_thr` — minimum overlap to claim a detection for an existing track. Too low → ID switches; too high → tracks die while the object is still there.
- `max_age` — how many frames a track survives occlusion/missed detections before it's dropped.
- `min_hits` — suppress unstable newborn tracks.

## Step 5 — Zones (`sentinel/zones.py`)

Zones are **polygons in fractional coordinates** (0–1), scaled to the real frame on the first frame — resolution-independent config.

```python
"""OpenEye Sentinel — polygon zones.

Zones are defined in *fractional* coordinates (0-1) so the same config works at
any resolution. Call scale_to(w, h) once the real frame size is known.
"""
import cv2
import numpy as np


class Zone:
    def __init__(self, name, polygon, color=(0, 200, 255)):
        self.name = name
        self.fractions = [(float(x), float(y)) for x, y in polygon]
        self.polygon = None
        self.color = color

    def scale_to(self, width, height):
        self.polygon = np.array(
            [[int(x * width), int(y * height)] for x, y in self.fractions], np.int32)
        return self

    def contains(self, point):
        return cv2.pointPolygonTest(
            self.polygon, (float(point[0]), float(point[1])), False) >= 0

    def draw(self, frame):
        cv2.polylines(frame, [self.polygon], True, self.color, 2)
        cx, cy = self.polygon.mean(axis=0).astype(int)
        cv2.putText(frame, self.name, (cx - 45, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
```

`cv2.pointPolygonTest` returns `+1` inside, `0` on the edge, `-1` outside. Testing the **centroid** against the polygon (not the box) is the standard, robust choice.

## Step 6 — The pipeline (`sentinel/pipeline.py`)

The main loop wires it all together: detect → track → zone state machine → events → annotate → write.

```python
"""OpenEye Sentinel — detection -> tracking -> zone analytics -> event stream.

Run:
    python sentinel/pipeline.py --source synthetic --model motion       # self-test, offline
    python sentinel/pipeline.py --source 0 --model hog                  # webcam
    python sentinel/pipeline.py --source clip.mp4 --model yolo          # best quality

Outputs: out/sentinel.mp4 (annotated video) and out/events.jsonl (one JSON event per line).
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from detector import HOGPeopleDetector, MotionDetector, YOLODetector
from tracker import IoUTracker, track_speed_kmh
from zones import Zone

ROOT = Path(__file__).resolve().parent.parent      # repo root, whatever the cwd

# zones as fractions of frame size: (name, polygon)
ZONE_DEFS = [
    ("left-lane", [(0.02, 0.98), (0.30, 0.45), (0.44, 0.45), (0.28, 0.98)]),
    ("right-lane", [(0.56, 0.45), (0.70, 0.45), (0.98, 0.98), (0.72, 0.98)]),
]


def synthetic_source(width=960, height=540, n_frames=400, fps=30.0):
    """Scripted scene: 3 moving rectangles crossing both zones. Lets you verify the whole
    pipeline (and the tracker) without a camera or video file."""
    def make(i):
        frame = np.full((height, width, 3), 30, np.uint8)
        cv2.rectangle(frame, (0, 430), (width, height), (45, 45, 45), -1)
        for k, speed in enumerate((3.5, 5.0, 2.0)):
            x = int((60 + i * speed + k * 320) % (width + 120)) - 60
            y2 = 480 - k * 60
            cv2.rectangle(frame, (x, y2 - 60), (x + 34, y2), (200, 200, 200), -1)
        return frame

    for i in range(n_frames):
        yield make(i), i / fps


def open_source(args):
    if args.source == "synthetic":
        return synthetic_source(n_frames=args.frames)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def gen():
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, i / fps
            i += 1

    return gen()


def build_detector(name):
    if name == "hog":
        return HOGPeopleDetector()
    if name == "yolo":
        return YOLODetector()
    return MotionDetector()


def log(fh, obj):
    fh.write(json.dumps(obj) + "\n")


def main():
    ap = argparse.ArgumentParser(description="OpenEye Sentinel")
    ap.add_argument("--source", default="synthetic", help="synthetic | video file | camera index")
    ap.add_argument("--model", default="motion", choices=["motion", "hog", "yolo"])
    ap.add_argument("--frames", type=int, default=400, help="synthetic mode length")
    ap.add_argument("--fps", type=float, default=30.0, help="used in synthetic mode")
    ap.add_argument("--out", default=str(ROOT / "out" / "sentinel.mp4"))
    ap.add_argument("--events", default=str(ROOT / "out" / "events.jsonl"))
    ap.add_argument("--meters-per-pixel", type=float, default=0.02,
                    help="calibrate: (real distance between two points) / (pixels between them)")
    ap.add_argument("--speed-limit-kmh", type=float, default=20.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.events) or ".", exist_ok=True)

    det = build_detector(args.model)
    tracker = IoUTracker(iou_thr=0.3, max_age=15)
    zones = None
    writer = None
    events_file = open(args.events, "w")
    summary = {name: {"enters": 0, "exits": 0} for name, _ in ZONE_DEFS}
    zone_state = {}    # (track_id, zone_name) -> {"inside": bool, "enter_t": float}
    speeding = {}      # track_id -> bool (log speed events on transition only)

    for frame, t in open_source(args):
        h, w = frame.shape[:2]
        if w != 960:                                   # normalize width for speed + HOG timing
            frame = cv2.resize(frame, (960, int(960 * h / w)))
            h, w = frame.shape[:2]
        if zones is None:
            zones = [Zone(name, poly).scale_to(w, h) for name, poly in ZONE_DEFS]

        dets = det(frame)
        tracks = tracker.update(dets)

        for z in zones:
            z.draw(frame)

        for tid, tr in tracks.items():
            if tr["hits"] < tracker.min_hits:
                continue
            x1, y1, x2, y2 = (int(v) for v in tr["box"])
            color = ((37 * tid) % 155 + 100, (91 * tid) % 155 + 100, (53 * tid) % 155 + 100)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            pts = np.array(tr["trail"], np.int32)
            if len(pts) > 1:
                cv2.polylines(frame, [pts], False, color, 2)

            speed = track_speed_kmh(tr, args.meters_per_pixel, args.fps)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{tid} {speed:.0f}km/h", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            for z in zones:
                key = (tid, z.name)
                st = zone_state.setdefault(key, {"inside": False, "enter_t": 0.0})
                is_in = z.contains((cx, cy))
                if is_in and not st["inside"]:
                    st["inside"], st["enter_t"] = True, t
                    summary[z.name]["enters"] += 1
                    log(events_file, {"t": round(t, 2), "type": "zone_enter",
                                      "track": tid, "zone": z.name})
                elif st["inside"] and not is_in:
                    st["inside"] = False
                    summary[z.name]["exits"] += 1
                    log(events_file, {"t": round(t, 2), "type": "zone_exit",
                                      "track": tid, "zone": z.name,
                                      "dwell_s": round(t - st["enter_t"], 2)})

            over = speed > args.speed_limit_kmh
            if over and not speeding.get(tid, False):
                log(events_file, {"t": round(t, 2), "type": "speed",
                                  "track": tid, "speed_kmh": round(speed, 1)})
            speeding[tid] = over

        # prune state for dead tracks
        live = set(tracks)
        for key in [k for k in zone_state if k[0] not in live]:
            del zone_state[key]
        for tid in [k for k in speeding if k not in live]:
            del speeding[tid]

        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (w, h))
        if writer.isOpened():
            writer.write(frame)
        if args.show:
            cv2.imshow("sentinel", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    events_file.close()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"events        -> {args.events}")
    print(f"annotated video -> {args.out}")
    print("summary:")
    for name, s in summary.items():
        print(f"  {name}: {s['enters']} enters, {s['exits']} exits")


if __name__ == "__main__":
    main()
```

**Three design details worth internalizing:**
1. **State machines, not frame decisions.** "Entered the zone" is a *transition* (`inside` flips false→true), not a per-frame check. That's also why speed violations are logged on transition only — otherwise you'd get 30 duplicate events per second.
2. **Events are the product.** The annotated video is for humans; `events.jsonl` is what a business system consumes (counts, alerts, dwell analytics).
3. **Zones as fractions.** Ship one config, run at 720p or 4K unchanged.

## Step 7 — Run and verify (offline self-test)

```bash
cd sentinel
python pipeline.py --source synthetic --model motion
python pipeline.py --source synthetic --model motion --speed-limit-kmh 8   # trips speed events
```

Expected: a summary with several `left-lane`/`right-lane` enters and exits, an annotated `../out/sentinel.mp4`, and `../out/events.jsonl`. (The scripted "people" move at 4–11 km/h, so the default 20 km/h limit logs no speed events; `--speed-limit-kmh 8` demonstrates them.) Then run the checks:

```python
import json
events = [json.loads(line) for line in open("../out/events.jsonl")]
enters = [e for e in events if e["type"] == "zone_enter"]
exits = [e for e in events if e["type"] == "zone_exit"]
tracks = {e["track"] for e in enters}

assert len(enters) >= 3, "expected several zone crossings in the scripted scene"
assert len(tracks) >= 2, "expected multiple tracked identities"
assert all(e["dwell_s"] >= 0 for e in exits)
print(f"OK: {len(enters)} enters from {len(tracks)} tracks, {len(exits)} exits, "
      f"{sum(1 for e in events if e['type'] == 'speed')} speed events")
```

## Step 8 — Real footage

```bash
cd sentinel
python pipeline.py --source ../clip.mp4 --model yolo --show      # or: --source 0 --model hog
```

**Calibrating `meters-per-pixel`:** pick two points whose real-world distance you know (e.g., a 3.5 m lane marking spanning 175 px) → `3.5 / 175 = 0.02 m/px`. Without calibration the speed numbers are meaningless — geometry always needs a scale.

## Step 9 — Dashboard (`sentinel/dashboard.py`)

```python
"""OpenEye Sentinel dashboard. Run: streamlit run sentinel/dashboard.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

EVENTS = Path(__file__).resolve().parent.parent / "out" / "events.jsonl"

st.title("OpenEye Sentinel")

if not EVENTS.exists():
    st.warning("No events yet. Run: python sentinel/pipeline.py --source synthetic --model motion")
    st.stop()

events = [json.loads(line) for line in EVENTS.read_text().splitlines() if line.strip()]
df = pd.DataFrame(events)

c1, c2, c3 = st.columns(3)
c1.metric("events", len(df))
c1.metric("zone entries", int((df["type"] == "zone_enter").sum()))
exits = df[df["type"] == "zone_exit"]
if len(exits):
    c2.metric("avg dwell (s)", round(exits["dwell_s"].mean(), 1))
c2.metric("speed events", int((df["type"] == "speed").sum()))

left, right = st.columns(2)
with left:
    st.subheader("entries per zone")
    st.bar_chart(df[df["type"] == "zone_enter"]["zone"].value_counts())
with right:
    st.subheader("event types")
    st.bar_chart(df["type"].value_counts())

st.subheader("timeline")
if "zone" in df:
    st.scatter_chart(df[df["type"] != "speed"], x="t", y="zone")

st.subheader("raw events (latest 200)")
st.dataframe(df.tail(200))
```

```bash
streamlit run sentinel/dashboard.py
```

## Step 10 — Know your failure modes

| Symptom | Cause | Fix |
|---|---|---|
| One person gets 5 IDs in 5 seconds | detector flickers; `iou_thr` too high | lower `iou_thr`, raise `max_age`, use YOLO |
| Two people merge into one ID | IoU match steals the neighbor's box | add a motion model (Kalman) — that's what SORT/ByteTrack add |
| Spurious tracks on shadow/waving tree | motion detector noise | raise `min_area`, use HOG/YOLO |
| Speed numbers wrong | `meters-per-pixel` not calibrated | calibrate with a known distance |
| Events at scene entry, not zone entry | zone polygon touches frame edge | inset the polygon |

The honest metric to track is **ID switches per minute**. If you extend this capstone, that's the number to minimize.

## Stretch goals

- **Re-identification:** embed each track's crop (color histogram, or CLIP from capstone 2) and merge tracks whose embeddings match after a gap — surveillance-grade tracking.
- **Directional counting:** a line-crossing counter with direction sign (cross-product of movement vector vs. line) instead of a polygon — count only crossings in one direction.
- **Anomaly alerts:** dwell > 60 s in a zone → alert event. Now you have a product.
- **Model upgrade path:** `supervision` (ByteTrack) and `boxmot` drop-in trackers, RT-DETR detectors.
- **Deploy:** Vercel Python function + Blob for the event stream, or run the pipeline as a Vercel Daemon (declared in `vercel.json` under `daemons`) if you want it always-on next to a dashboard.
