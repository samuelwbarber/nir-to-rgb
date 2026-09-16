import json
import re
import random
from pathlib import Path

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from torch.utils.data import Dataset


def load_mp_cache(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"mediapipe cache not found: {p}. "
            f"Run scripts/precompute_mediapipe_boxes.py first."
        )
    with open(p) as f:
        data = json.load(f)
    return data.get("items", data)


def discover_pairs(nir_dir, rgb_dir, extensions):
    nir_dir, rgb_dir = Path(nir_dir), Path(rgb_dir)
    exts = {e.lower() for e in extensions}
    nir_index = {}
    for p in nir_dir.rglob("*"):
        if p.suffix.lower() in exts:
            nir_index[p.stem] = p
    pairs = []
    for p in rgb_dir.rglob("*"):
        if p.suffix.lower() in exts and p.stem in nir_index:
            pairs.append((nir_index[p.stem], p))
    pairs.sort(key=lambda x: x[0].as_posix())
    return pairs


def split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed=42, scene_regex=None):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)

    if scene_regex:
        pat = re.compile(scene_regex)
        scenes = {}
        for nir, rgb in pairs:
            m = pat.search(nir.stem)
            key = m.group(0) if m else nir.stem
            scenes.setdefault(key, []).append((nir, rgb))
        scene_keys = list(scenes.keys())
        rng.shuffle(scene_keys)
        n = len(scene_keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_keys = scene_keys[:n_train]
        val_keys = scene_keys[n_train:n_train + n_val]
        test_keys = scene_keys[n_train + n_val:]
        flatten = lambda ks: [p for k in ks for p in scenes[k]]
        return flatten(train_keys), flatten(val_keys), flatten(test_keys)

    pairs = list(pairs)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]


def _read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read {path}")
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class PairedNIRRGBDataset(Dataset):
    def __init__(self, pairs, image_size=256, train=True, aug_cfg=None, mp_cache=None):
        self.pairs = pairs
        self.image_size = image_size
        self.train = train
        a = aug_cfg or {}
        self.hflip = a.get("hflip", True)
        self.rotate_max = a.get("rotate_max_deg", 0.0)
        self.scale_min = a.get("scale_min", 1.0)
        self.scale_max = a.get("scale_max", 1.0)
        self.nir_brightness = a.get("nir_brightness", 0.0)
        self.nir_gamma_min = a.get("nir_gamma_min", 1.0)
        self.nir_gamma_max = a.get("nir_gamma_max", 1.0)
        self.mp_cache = mp_cache or {}

    def __len__(self):
        return len(self.pairs)

    def _paired_geom(self, nir, rgb, mask):
        s = self.image_size
        h, w = nir.shape[:2]
        if self.train:
            scale = random.uniform(self.scale_min, self.scale_max)
            short = min(h, w)
            target_short = int(round(s * scale))
            ratio = target_short / short
            new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
            nir = cv2.resize(nir, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if new_h < s or new_w < s:
                ph = max(0, s - new_h)
                pw = max(0, s - new_w)
                nir = cv2.copyMakeBorder(nir, 0, ph, 0, pw, cv2.BORDER_REFLECT)
                rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT)
                mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
                new_h, new_w = nir.shape[:2]
            y = random.randint(0, new_h - s)
            x = random.randint(0, new_w - s)
            nir = nir[y:y + s, x:x + s]
            rgb = rgb[y:y + s, x:x + s]
            mask = mask[y:y + s, x:x + s]
            if self.hflip and random.random() < 0.5:
                nir = nir[:, ::-1]
                rgb = rgb[:, ::-1]
                mask = mask[:, ::-1]
            if self.rotate_max > 0:
                angle = random.uniform(-self.rotate_max, self.rotate_max)
                M = cv2.getRotationMatrix2D((s / 2, s / 2), angle, 1.0)
                nir = cv2.warpAffine(nir, M, (s, s), borderMode=cv2.BORDER_REFLECT)
                rgb = cv2.warpAffine(rgb, M, (s, s), borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (s, s), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            # Aspect-preserving val: resize short side to s, then center-crop s×s.
            # Avoids the anisotropic squash that mismatches train-time crops.
            h, w = nir.shape[:2]
            short = min(h, w)
            if short != s:
                ratio = s / short
                new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
                nir = cv2.resize(nir, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                h, w = new_h, new_w
            y = max(0, (h - s) // 2)
            x = max(0, (w - s) // 2)
            nir = nir[y:y + s, x:x + s]
            rgb = rgb[y:y + s, x:x + s]
            mask = mask[y:y + s, x:x + s]
        return (np.ascontiguousarray(nir),
                np.ascontiguousarray(rgb),
                np.ascontiguousarray(mask))

    def _photometric_nir(self, nir):
        if not self.train:
            return nir
        out = nir.astype(np.float32)
        if self.nir_brightness > 0:
            shift = random.uniform(-self.nir_brightness, self.nir_brightness) * 255.0
            out = np.clip(out + shift, 0, 255)
        if self.nir_gamma_min != 1.0 or self.nir_gamma_max != 1.0:
            gamma = random.uniform(self.nir_gamma_min, self.nir_gamma_max)
            out = np.clip(((out / 255.0) ** gamma) * 255.0, 0, 255)
        return out.astype(np.uint8)

    def _lookup_cache(self, nir_path, rgb_path):
        if not self.mp_cache:
            return None
        return self.mp_cache.get(rgb_path.stem) or self.mp_cache.get(nir_path.stem)

    def __getitem__(self, idx):
        nir_path, rgb_path = self.pairs[idx]
        nir = _read_image(nir_path)
        rgb = _read_image(rgb_path)

        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        entry = self._lookup_cache(nir_path, rgb_path)
        if entry:
            region_box = entry.get("region_bbox")
            if region_box:
                rx1, ry1, rx2, ry2 = region_box
                rx1, ry1 = max(0, rx1), max(0, ry1)
                rx2, ry2 = min(w, rx2), min(h, ry2)
                if rx2 > rx1 and ry2 > ry1:
                    mask[ry1:ry2, rx1:rx2] = 1

            crop_box = entry.get("crop_bbox")
            if crop_box:
                cx1, cy1, cx2, cy2 = crop_box
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(w, cx2), min(h, cy2)
                if cx2 - cx1 >= 16 and cy2 - cy1 >= 16:
                    if nir.shape[:2] == (h, w):
                        nir = nir[cy1:cy2, cx1:cx2]
                    else:
                        nir = cv2.resize(nir, (w, h), interpolation=cv2.INTER_LINEAR)[cy1:cy2, cx1:cx2]
                    rgb = rgb[cy1:cy2, cx1:cx2]
                    mask = mask[cy1:cy2, cx1:cx2]

        nir, rgb, mask = self._paired_geom(nir, rgb, mask)
        nir = self._photometric_nir(nir)
        nir_t = torch.from_numpy(nir.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).contiguous()
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        return {
            "nir": nir_t,
            "rgb": rgb_t,
            "region_mask": mask_t,
            "nir_path": str(nir_path),
            "rgb_path": str(rgb_path),
        }
