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
