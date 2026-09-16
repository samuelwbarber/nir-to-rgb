"""Run pretrained models on (NIR direct, translated, GT-RGB) and report
how badly NIR breaks each one — and how much of that gap your translator
closes.

The headline number per task is `gap_closure`:
    (translated - nir) / (rgb - nir)

1.0 = matches RGB performance. 0.0 = no improvement over raw NIR.
Negative = translator made things worse than feeding NIR directly.

Usage examples:

    # baseline (no translator, just NIR-vs-RGB to see how broken NIR is)
    python scripts/eval_downstream.py --config configs/baseline.yaml \
        --models mp_hands mp_face yolo deeplab resnet50 \
        --max-samples 200

    # with a trained translator
    python scripts/eval_downstream.py --config configs/teacher_nafnet.yaml \
        --checkpoint experiments/phase2_teacher_nafnet32/checkpoints/final.pth \
        --models all --max-samples 500
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
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
from src.eval.downstream import REGISTRY, ALL, SKIN_HEAVY, COLOR_HEAVY, GEOMETRIC


def _resolve_model_names(names):
    out = []
    for n in names:
        if n == "all":
            out.extend(ALL)
        elif n == "skin":
            out.extend(SKIN_HEAVY)
        elif n == "color":
            out.extend(COLOR_HEAVY)
        elif n == "geometric":
            out.extend(GEOMETRIC)
        elif n in REGISTRY:
            out.append(n)
        else:
            raise ValueError(f"unknown model: {n}")
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


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


def _aggregate(metric_dicts):
    """Average each numeric field across a list of dicts, ignoring NaNs."""
    bag = defaultdict(list)
    for d in metric_dicts:
        for k, v in d.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            bag[k].append(v)
    return {k: float(np.mean(v)) for k, v in bag.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="omit for NIR-vs-RGB baseline (no translator)")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--models", nargs="+", default=["all"],
                   help="model names from registry, or 'all' / 'skin' / 'color' / 'geometric'")
    p.add_argument("--size", type=int, default=None,
                   help="resize images to this square size (default: cfg.data.image_size)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--mp-cache", type=str, default=None,
                   help="path to mp_cache.json from precompute_mediapipe_boxes.py. "
                        "When provided, smart-crop NIR/RGB to the cached bbox before resizing, "
                        "matching the training preprocessing. Defaults to cfg.data.mp_cache.")
    p.add_argument("--no-mp-cache", action="store_true",
                   help="ignore mp_cache even if cfg.data.mp_cache is set")
    args = p.parse_args()

    cfg, _ = load_config(args.config)
    device = args.device or (cfg.device if torch.cuda.is_available() else "cpu")
    size = args.size or cfg.data.image_size
    model_names = _resolve_model_names(args.models)
    print(f"models: {model_names}")
    print(f"device: {device}, image size: {size}")

    mp_cache_path = None if args.no_mp_cache else (args.mp_cache or getattr(cfg.data, "mp_cache", None))
    mp_cache = load_mp_cache(mp_cache_path) if mp_cache_path else {}
    if mp_cache:
        n_with_crop = sum(1 for v in mp_cache.values() if v.get("crop_bbox"))
        print(f"mp_cache: {len(mp_cache)} entries, {n_with_crop} with crop_bbox "
              f"(smart-cropping inputs before {size}x{size} resize)")

    pairs = discover_pairs(cfg.data.nir_dir, cfg.data.rgb_dir, cfg.data.extensions)
    if not pairs:
        raise SystemExit(f"no pairs found in {cfg.data.nir_dir} / {cfg.data.rgb_dir}")
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio,
        seed=cfg.split.seed, scene_regex=cfg.split.scene_regex,
    )
    chosen = {"train": train_pairs, "val": val_pairs, "test": test_pairs}[args.split]
    if args.max_samples:
        chosen = chosen[: args.max_samples]
    print(f"evaluating on {len(chosen)} {args.split} samples")

    generator = None
    if args.checkpoint:
        generator = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels).to(device).eval()
        load_checkpoint(args.checkpoint, generator, map_location=device)
        print(f"loaded translator from {args.checkpoint}")
    else:
        print("no checkpoint — running NIR-direct vs GT-RGB baseline only")

    print("setting up evaluators…")
    evaluators = []
    for name in model_names:
        cls = REGISTRY[name]
        try:
            ev = cls()
            t0 = time.time()
            ev.setup(device)
            print(f"  {name:12s}  loaded in {time.time() - t0:.1f}s")
            evaluators.append(ev)
        except Exception as e:
            print(f"  {name:12s}  SKIPPED ({type(e).__name__}: {e})")

    if not evaluators:
        raise SystemExit("no evaluators loaded — install the dependencies and try again")

    per_image = {ev.name: {"nir_stats": [], "rgb_stats": [], "trans_stats": [],
                           "nir_vs_rgb": [], "trans_vs_rgb": []} for ev in evaluators}

    for nir_path, rgb_path in tqdm(chosen, desc="eval"):
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
        trans_img = _translate(generator, nir_img, device) if generator is not None else None

        for ev in evaluators:
            r_nir = ev.predict(nir_img)
            r_rgb = ev.predict(rgb_img)
            per_image[ev.name]["nir_stats"].append(ev.per_condition_stats(r_nir))
            per_image[ev.name]["rgb_stats"].append(ev.per_condition_stats(r_rgb))
            per_image[ev.name]["nir_vs_rgb"].append(ev.vs_reference(r_nir, r_rgb))
            if trans_img is not None:
                r_trans = ev.predict(trans_img)
                per_image[ev.name]["trans_stats"].append(ev.per_condition_stats(r_trans))
                per_image[ev.name]["trans_vs_rgb"].append(ev.vs_reference(r_trans, r_rgb))

    print("\n" + "=" * 80)
    print(f"DOWNSTREAM EVALUATION — n={len(chosen)} ({args.split})")
    if generator is not None:
        print(f"checkpoint: {args.checkpoint}")
    print("=" * 80)

    summary = {}
    for ev in evaluators:
        block = per_image[ev.name]
        nir_stats = _aggregate(block["nir_stats"])
        rgb_stats = _aggregate(block["rgb_stats"])
        nir_ref = _aggregate(block["nir_vs_rgb"])
        trans_stats = _aggregate(block["trans_stats"]) if block["trans_stats"] else None
        trans_ref = _aggregate(block["trans_vs_rgb"]) if block["trans_vs_rgb"] else None

        s = {
            "primary_metric": ev.primary_metric,
            "nir_per_condition": nir_stats,
            "rgb_per_condition": rgb_stats,
            "nir_vs_rgb": nir_ref,
            "translated_per_condition": trans_stats,
            "translated_vs_rgb": trans_ref,
        }

        primary = ev.primary_metric
        nir_p = nir_ref.get(primary, nir_stats.get(primary, float("nan")))
        rgb_p = rgb_stats.get(primary, 1.0)
        trans_p = (trans_ref or {}).get(primary, (trans_stats or {}).get(primary, float("nan"))) if trans_stats is not None else None

        gap = None
        if trans_p is not None and not math.isnan(trans_p):
            denom = rgb_p - nir_p
            if abs(denom) > 1e-9:
                gap = (trans_p - nir_p) / denom
        s["gap_closure"] = gap

        summary[ev.name] = s

        print(f"\n[{ev.name}]  primary metric: {primary}")
        print(f"  NIR direct  : {nir_p:.4f}")
        print(f"  GT RGB      : {rgb_p:.4f}  (ceiling)")
        if trans_p is not None and not math.isnan(trans_p):
            print(f"  Translated  : {trans_p:.4f}")
            if gap is not None:
                print(f"  → gap closure: {gap:+.2%}  (1.0 = matches RGB; 0.0 = no improvement)")
        print(f"  full nir_vs_rgb     : {nir_ref}")
        if trans_ref:
            print(f"  full trans_vs_rgb   : {trans_ref}")
        print(f"  per-condition (NIR) : {nir_stats}")
        print(f"  per-condition (RGB) : {rgb_stats}")
        if trans_stats:
            print(f"  per-condition (TRA) : {trans_stats}")

    out = Path(args.out_json) if args.out_json else (
        Path("experiments") / cfg.project.experiment / f"downstream_{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"n_samples": len(chosen), "checkpoint": args.checkpoint,
                   "models": model_names, "summary": summary}, f, indent=2)
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
