"""MiDaS depth estimation — included as a contrast case.

NIR contains a lot of geometric/texture information that depth nets rely on,
so MiDaS may actually survive NIR input reasonably well. Useful as a
sanity-check: if your translator hurts depth estimation but helps everything
else, the translation may be doing something pathological.
"""

import numpy as np
import torch

from .base import BaseEvaluator


def _spearman(a, b):
    a = a.flatten(); b = b.flatten()
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    n = len(a)
    if n < 2:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


class MiDaSEvaluator(BaseEvaluator):
    name = "midas"
    requires = ("torch", "timm")
    primary_metric = "depth_rank_corr_vs_rgb"

    def __init__(self, model_type="MiDaS_small"):
        self.model_type = model_type

    def setup(self, device):
        try:
            import timm  # noqa: F401  (MiDaS hub model needs timm)
        except ImportError as e:
            raise RuntimeError("timm not installed (required by MiDaS). pip install timm") from e
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
        self.model.to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform if self.model_type == "MiDaS_small" else transforms.dpt_transform
        self.device = device

    def predict(self, rgb_uint8):
        x = self.transform(rgb_uint8).to(self.device)
        with torch.no_grad():
            pred = self.model(x)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb_uint8.shape[:2], mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()
        return {"depth": pred.astype(np.float32)}

    def per_condition_stats(self, p):
        d = p["depth"]
        return {"depth_mean": float(d.mean()), "depth_std": float(d.std())}

    def vs_reference(self, pred, ref):
        a = pred["depth"]; b = ref["depth"]
        if a.shape != b.shape:
            return {"depth_rank_corr_vs_rgb": float("nan"), "depth_rel_l1_vs_rgb": float("nan")}
        a_n = (a - a.mean()) / (a.std() + 1e-6)
        b_n = (b - b.mean()) / (b.std() + 1e-6)
        rel_l1 = float(np.abs(a_n - b_n).mean())
        rank_corr = _spearman(a, b)
        return {"depth_rank_corr_vs_rgb": rank_corr, "depth_rel_l1_vs_rgb": rel_l1}
