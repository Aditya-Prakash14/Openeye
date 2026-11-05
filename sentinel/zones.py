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
