"""Torchvision-based evaluators: DeepLabV3, Mask R-CNN, ResNet50.

DeepLabV3:  Pascal VOC semantic segmentation (21 classes incl. background).
Mask R-CNN: COCO instance segmentation + detection.
ResNet50:   ImageNet 1000-class classification.

All are pretrained on natural RGB and rely heavily on color/texture priors
that NIR doesn't satisfy.
"""

import numpy as np
import torch

from .base import BaseEvaluator


def _to_tensor(rgb_uint8, device):
    t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


class DeepLabV3Evaluator(BaseEvaluator):
    name = "deeplab"
    requires = ("torchvision",)
    primary_metric = "miou_vs_rgb"

    def setup(self, device):
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=weights).to(device).eval()
        self.transforms = weights.transforms()
        self.device = device
        self.n_classes = 21

    def predict(self, rgb_uint8):
        x = _to_tensor(rgb_uint8, self.device)
        x = self.transforms(x)
        with torch.no_grad():
            out = self.model(x)["out"]
        seg = out.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return {"seg": seg}

    def per_condition_stats(self, p):
        seg = p["seg"]
        nonbg = (seg != 0).mean()
        n_classes = int(len(np.unique(seg)))
        return {"nonbg_pixel_frac": float(nonbg), "n_classes_present": float(n_classes)}

    def vs_reference(self, pred, ref):
        a = pred["seg"]; b = ref["seg"]
        if a.shape != b.shape:
            return {"miou_vs_rgb": float("nan"), "pixel_acc_vs_rgb": float("nan")}
        pixel_acc = float((a == b).mean())
        ious = []
        present = np.unique(b)
        for c in present:
            am = (a == c); bm = (b == c)
            union = (am | bm).sum()
            if union == 0:
                continue
            ious.append((am & bm).sum() / union)
        miou = float(np.mean(ious)) if ious else float("nan")
        return {"miou_vs_rgb": miou, "pixel_acc_vs_rgb": pixel_acc}


class MaskRCNNEvaluator(BaseEvaluator):
    name = "maskrcnn"
    requires = ("torchvision",)
    primary_metric = "matched_iou_vs_rgb"

    def __init__(self, score_threshold=0.5, iou_match_threshold=0.5):
        self.score_threshold = score_threshold
        self.iou_match_threshold = iou_match_threshold

    def setup(self, device):
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
        )
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
        self.device = device

    def predict(self, rgb_uint8):
        x = _to_tensor(rgb_uint8, self.device)
        with torch.no_grad():
            out = self.model(x)[0]
        keep = out["scores"].cpu().numpy() >= self.score_threshold
        return {
            "boxes": out["boxes"][keep].cpu().numpy(),
            "labels": out["labels"][keep].cpu().numpy().astype(int),
            "scores": out["scores"][keep].cpu().numpy(),
        }

    def per_condition_stats(self, p):
        n = len(p["labels"])
        return {
            "detection_rate": float(n > 0),
            "n_objects": float(n),
            "mean_conf": float(p["scores"].mean()) if n else float("nan"),
        }

    def vs_reference(self, pred, ref):
        if len(ref["labels"]) == 0:
            return {"matched_iou_vs_rgb": float("nan"), "class_agreement": float("nan"), "recall_vs_rgb": float("nan")}
        if len(pred["labels"]) == 0:
            return {"matched_iou_vs_rgb": 0.0, "class_agreement": 0.0, "recall_vs_rgb": 0.0}

        from .yolo import _iou_xyxy
        used = set()
        ious, cls_match = [], []
        matched = 0
        for ri in range(len(ref["labels"])):
            r_box = ref["boxes"][ri]
            r_lbl = ref["labels"][ri]
            best, best_iou = None, 0.0
            for pi in range(len(pred["labels"])):
                if pi in used:
                    continue
                iou = _iou_xyxy(r_box, pred["boxes"][pi])
                if iou > best_iou:
                    best_iou = iou; best = pi
            if best is not None and best_iou >= self.iou_match_threshold:
                used.add(best)
                ious.append(best_iou)
                cls_match.append(1.0 if pred["labels"][best] == r_lbl else 0.0)
                matched += 1
        return {
            "matched_iou_vs_rgb": float(np.mean(ious)) if ious else 0.0,
            "class_agreement": float(np.mean(cls_match)) if cls_match else 0.0,
            "recall_vs_rgb": matched / len(ref["labels"]),
        }


class ResNet50Evaluator(BaseEvaluator):
    name = "resnet50"
    requires = ("torchvision",)
    primary_metric = "top1_agreement"

    def __init__(self, top_k=5):
        self.top_k = top_k

    def setup(self, device):
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights).to(device).eval()
        self.transforms = weights.transforms()
        self.device = device

    def predict(self, rgb_uint8):
        x = _to_tensor(rgb_uint8, self.device)
        x = self.transforms(x)
        with torch.no_grad():
            logits = self.model(x)[0]
        probs = torch.softmax(logits, dim=0)
        topk = probs.topk(self.top_k)
        return {
            "topk_idx": topk.indices.cpu().numpy().astype(int),
            "topk_prob": topk.values.cpu().numpy(),
        }

    def per_condition_stats(self, p):
        return {"top1_prob": float(p["topk_prob"][0])}

    def vs_reference(self, pred, ref):
        top1_match = float(pred["topk_idx"][0] == ref["topk_idx"][0])
        ref_top1 = ref["topk_idx"][0]
        topk_contains_match = float(ref_top1 in pred["topk_idx"])
        return {"top1_agreement": top1_match, f"top{self.top_k}_agreement": topk_contains_match}
