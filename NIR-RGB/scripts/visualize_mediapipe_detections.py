"""Render mediapipe detection overlays on (NIR, translated, GT-RGB) triplets.

For each test frame and each mediapipe model, emit a 3-up image whenever any
modality produced a detection. Output goes to <out-dir>/<model>/<stem>.png.

For mp_selfie (where every frame "detects" a mask), only emit when the worst
modality-vs-RGB IoU is below ``--selfie-iou-thresh`` (otherwise every frame
qualifies).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.utils.config import load_config
from src.data.dataset import discover_pairs, split_pairs, _read_image, load_mp_cache
from src.models import build_generator
from src.training.checkpoint import load_checkpoint
from src.eval.downstream.mediapipe_models import (
    MediaPipeHandsEvaluator, MediaPipeFaceEvaluator,
    MediaPipePoseEvaluator, MediaPipeSelfieEvaluator,
)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def _load_uint8(path, size, crop_bbox=None):
    img = _read_image(path)
    if crop_bbox is not None:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = crop_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 >= 16 and y2 - y1 >= 16:
            img = img[y1:y2, x1:x2]
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


def _translate(generator, nir_uint8, device):
    x = torch.from_numpy(nir_uint8.astype(np.float32) / 127.5 - 1.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        y = generator(x)
    y = (y.clamp(-1, 1)[0] + 1) * 127.5
    return y.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def _draw_landmarks(bgr, lms, connections, color_pt, color_ln, point_radius=2):
    h, w = bgr.shape[:2]
    pts = [(int(round(x * w)), int(round(y * h))) for (x, y, _z) in lms]
    for a, b in connections:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(bgr, pts[a], pts[b], color_ln, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(bgr, p, point_radius, color_pt, -1, cv2.LINE_AA)


def _overlay_mask(bgr, mask, color):
    if mask is None:
        return
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.squeeze()
    if m.ndim != 2:
        return
    if m.shape != bgr.shape[:2]:
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    m_bool = m.astype(bool)
    if not m_bool.any():
        return
    color_arr = np.array(color, dtype=np.float32).reshape(1, 3)
    bgr[m_bool] = (color_arr * 0.5 + bgr[m_bool].astype(np.float32) * 0.5).astype(np.uint8)


def _label(panel, text, fg=(255, 255, 255)):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(panel, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fg, 1, cv2.LINE_AA)


def _draw_for_model(bgr, model_name, det):
    """Returns a short status string describing what was drawn."""
    if model_name == "mp_hands":
        hands = (det or {}).get("hands") or []
        for hand in hands:
            _draw_landmarks(bgr, hand, HAND_CONNECTIONS, (0, 255, 0), (0, 200, 0))
        return f"{len(hands)} hand(s)" if hands else "no detection"
    if model_name == "mp_face":
        faces = (det or {}).get("faces") or []
        h, w = bgr.shape[:2]
        for face in faces:
            for (x, y, _z) in face:
                cv2.circle(bgr, (int(x * w), int(y * h)), 1, (0, 200, 255), -1)
        return f"{len(faces)} face(s)" if faces else "no detection"
    if model_name == "mp_pose":
        lms = (det or {}).get("landmarks")
        if lms is None:
            return "no detection"
        _draw_landmarks(bgr, lms, POSE_CONNECTIONS, (0, 255, 255), (0, 200, 200))
        return "pose"
    if model_name == "mp_selfie":
        mask = (det or {}).get("mask")
        if mask is None:
            return "no mask"
        _overlay_mask(bgr, mask, (0, 255, 0))
        return f"frac={float(mask.mean()):.2f}"
    return ""


def _render_triplet(nir_rgb, trans_rgb, gt_rgb, model_name, detections, out_path):
    panels = []
    order = [("NIR direct", nir_rgb), ("Translated", trans_rgb), ("GT RGB", gt_rgb)]
    for label, img_rgb in order:
        bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
        stat = _draw_for_model(bgr, model_name, detections[label])
        _label(bgr, f"{label} | {stat}")
        panels.append(bgr)
    grid = np.concatenate(panels, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


def _has_detection(name, p):
    if name == "mp_hands":
        return bool(p.get("hands"))
    if name == "mp_face":
        return bool(p.get("faces"))
    if name == "mp_pose":
        return p.get("landmarks") is not None
    if name == "mp_selfie":
        return p.get("mask") is not None
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--models", nargs="+",
                   default=["mp_hands", "mp_face", "mp_pose", "mp_selfie"])
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--selfie-iou-thresh", type=float, default=0.7,
                   help="for mp_selfie only: emit a visual when min(IoU(NIR,RGB), "
                        "IoU(trans,RGB)) is below this. Default 0.7.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-mp-cache", action="store_true")
    args = p.parse_args()

    cfg, _ = load_config(args.config)
    device = args.device or (cfg.device if torch.cuda.is_available() else "cpu")
    size = cfg.data.image_size

    mp_cache_path = None if args.no_mp_cache else getattr(cfg.data, "mp_cache", None)
    mp_cache = load_mp_cache(mp_cache_path) if mp_cache_path else {}
    if mp_cache:
        n_with_crop = sum(1 for v in mp_cache.values() if v.get("crop_bbox"))
        print(f"mp_cache: {len(mp_cache)} entries, {n_with_crop} with crop_bbox")

    pairs = discover_pairs(cfg.data.nir_dir, cfg.data.rgb_dir, cfg.data.extensions)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio,
        seed=cfg.split.seed, scene_regex=cfg.split.scene_regex,
    )
    chosen = {"train": train_pairs, "val": val_pairs, "test": test_pairs}[args.split]
    if args.max_samples:
        chosen = chosen[: args.max_samples]
    print(f"running on {len(chosen)} {args.split} samples")

    G = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels).to(device).eval()
    load_checkpoint(args.checkpoint, G, map_location=device)
    print(f"loaded translator from {args.checkpoint}")

    eval_classes = {
        "mp_hands": MediaPipeHandsEvaluator,
        "mp_face": MediaPipeFaceEvaluator,
        "mp_pose": MediaPipePoseEvaluator,
        "mp_selfie": MediaPipeSelfieEvaluator,
    }
    evaluators = {}
    for m in args.models:
        if m not in eval_classes:
            print(f"skipping unknown model {m}")
            continue
        ev = eval_classes[m]()
        ev.setup(device)
        evaluators[m] = ev
        print(f"  {m} ready")

    out_root = Path(args.out_dir)
    counts = {m: 0 for m in evaluators}

    for nir_path, rgb_path in tqdm(chosen, desc="visualize"):
        crop_bbox = None
        if mp_cache:
            entry = mp_cache.get(rgb_path.stem) or mp_cache.get(nir_path.stem)
            if entry:
                crop_bbox = entry.get("crop_bbox")
        try:
            nir_img = _load_uint8(nir_path, size, crop_bbox=crop_bbox)
            rgb_img = _load_uint8(rgb_path, size, crop_bbox=crop_bbox)
        except FileNotFoundError:
            continue
        trans_img = _translate(G, nir_img, device)
        stem = rgb_path.stem

        for name, ev in evaluators.items():
            r_nir = ev.predict(nir_img)
            r_rgb = ev.predict(rgb_img)
            r_trans = ev.predict(trans_img)

            if name == "mp_selfie":
                iou_nir = ev.vs_reference(r_nir, r_rgb).get("iou_vs_rgb", 1.0)
                iou_trans = ev.vs_reference(r_trans, r_rgb).get("iou_vs_rgb", 1.0)
                if min(iou_nir, iou_trans) >= args.selfie_iou_thresh:
                    continue
            else:
                if not any(_has_detection(name, r) for r in (r_nir, r_rgb, r_trans)):
                    continue

            detections = {"NIR direct": r_nir, "Translated": r_trans, "GT RGB": r_rgb}
            # Translator false positive (only meaningful for detection-based models): trans
            # fired and ground-truth RGB did not. Route those to trans_fp/ for inspection.
            trans_fp = (name in ("mp_hands", "mp_face", "mp_pose")
                        and _has_detection(name, r_trans)
                        and not _has_detection(name, r_rgb))
            sub = "trans_fp" if trans_fp else ""
            out_path = (out_root / name / sub / f"{stem}.png") if sub else (out_root / name / f"{stem}.png")
            try:
                _render_triplet(nir_img, trans_img, rgb_img, name, detections, out_path)
                counts[name] += 1
                if trans_fp:
                    counts.setdefault(f"{name}/trans_fp", 0)
                    counts[f"{name}/trans_fp"] += 1
            except Exception as e:
                print(f"  [skip] {name}/{stem}: {type(e).__name__}: {e}")

    print("\nWritten:")
    for m, c in counts.items():
        print(f"  {m}: {c} visuals → {out_root / m}")


if __name__ == "__main__":
    main()
