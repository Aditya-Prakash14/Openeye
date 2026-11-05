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
