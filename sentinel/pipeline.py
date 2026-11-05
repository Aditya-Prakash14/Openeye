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
