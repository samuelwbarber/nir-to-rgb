"""YOLOv8 object detection — COCO 80-class, color-and-texture dependent.

Vegetation looks bright in NIR (inverted from RGB), skin looks pale, and
many object classes have characteristic colors that NIR strips out.
Detection rate and class agreement should drop sharply on NIR direct.
"""

import numpy as np

from .base import BaseEvaluator


def _iou_xyxy(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class YOLOEvaluator(BaseEvaluator):
    name = "yolo"
    requires = ("ultralytics",)
    primary_metric = "matched_iou_vs_rgb"

    def __init__(self, weights="yolov8m.pt", iou_match_threshold=0.5, conf=0.25):
        self.weights = weights
        self.iou_match_threshold = iou_match_threshold
        self.conf = conf

    def setup(self, device):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("ultralytics not installed. pip install ultralytics") from e
        self.model = YOLO(self.weights)
        self.device = device

    def predict(self, rgb_uint8):
        bgr = rgb_uint8[..., ::-1]
        results = self.model.predict(bgr, verbose=False, device=self.device, conf=self.conf)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return {"xyxy": np.zeros((0, 4)), "cls": np.zeros((0,), dtype=int), "conf": np.zeros((0,))}
        return {
            "xyxy": boxes.xyxy.cpu().numpy(),
            "cls": boxes.cls.cpu().numpy().astype(int),
            "conf": boxes.conf.cpu().numpy(),
        }

    def per_condition_stats(self, p):
        n = len(p["cls"])
        return {
            "detection_rate": float(n > 0),
            "n_objects": float(n),
            "mean_conf": float(p["conf"].mean()) if n else float("nan"),
        }

    def vs_reference(self, pred, ref):
        if len(ref["cls"]) == 0:
            return {"matched_iou_vs_rgb": float("nan"), "class_agreement": float("nan"), "recall_vs_rgb": float("nan")}
        if len(pred["cls"]) == 0:
            return {"matched_iou_vs_rgb": 0.0, "class_agreement": 0.0, "recall_vs_rgb": 0.0}

        used = set()
        ious = []
        cls_match = []
        matched = 0
        for ri in range(len(ref["cls"])):
            r_box = ref["xyxy"][ri]
            r_cls = ref["cls"][ri]
            best, best_iou = None, 0.0
            for pi in range(len(pred["cls"])):
                if pi in used:
                    continue
                iou = _iou_xyxy(r_box, pred["xyxy"][pi])
                if iou > best_iou:
                    best_iou = iou; best = pi
            if best is not None and best_iou >= self.iou_match_threshold:
                used.add(best)
                ious.append(best_iou)
                cls_match.append(1.0 if pred["cls"][best] == r_cls else 0.0)
                matched += 1
        return {
            "matched_iou_vs_rgb": float(np.mean(ious)) if ious else 0.0,
            "class_agreement": float(np.mean(cls_match)) if cls_match else 0.0,
            "recall_vs_rgb": matched / len(ref["cls"]),
        }
