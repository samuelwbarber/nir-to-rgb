#!/usr/bin/env python3
"""Render test-set NIR / predicted RGB / GT RGB grids for a checkpoint."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from tqdm import tqdm

from scripts.eval_downstream import _load_uint8, _translate
from src.data.dataset import discover_pairs, split_pairs, load_mp_cache
from src.models import build_generator
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config


def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _save_grid(nir, pred, rgb, out_path):
    grid = np.concatenate([
        _label(nir, "NIR input"),
        _label(pred, "Predicted RGB"),
        _label(rgb, "GT RGB"),
    ], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg, _ = load_config(args.config)
    device = args.device or (cfg.device if torch.cuda.is_available() else "cpu")
    size = args.size or cfg.data.image_size

    mp_cache_path = getattr(cfg.data, "mp_cache", None)
    mp_cache = load_mp_cache(mp_cache_path) if mp_cache_path else {}
    pairs = discover_pairs(cfg.data.nir_dir, cfg.data.rgb_dir, cfg.data.extensions)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        cfg.split.train_ratio,
        cfg.split.val_ratio,
        cfg.split.test_ratio,
        seed=cfg.split.seed,
        scene_regex=cfg.split.scene_regex,
    )
    chosen = {"train": train_pairs, "val": val_pairs, "test": test_pairs}[args.split]
    if args.limit:
        chosen = chosen[: args.limit]

    generator = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels).to(device).eval()
    load_checkpoint(args.checkpoint, generator, map_location=device)

    out_dir = Path(args.out_dir)
    manifest = []
    print(f"device: {device}")
    print(f"rendering {len(chosen)} {args.split} samples to {out_dir}")
    for idx, (nir_path, rgb_path) in enumerate(tqdm(chosen, desc="render")):
        crop_bbox = None
        if mp_cache:
            entry = mp_cache.get(rgb_path.stem) or mp_cache.get(nir_path.stem)
            if entry:
                crop_bbox = entry.get("crop_bbox")
        nir = _load_uint8(nir_path, size, crop_bbox=crop_bbox)
        rgb = _load_uint8(rgb_path, size, crop_bbox=crop_bbox)
        pred = _translate(generator, nir, device)
        out_path = out_dir / f"{idx:05d}_{nir_path.stem}.png"
        _save_grid(nir, pred, rgb, out_path)
        manifest.append(f"{idx:05d}\t{nir_path}\t{rgb_path}\t{out_path}\n")

    (out_dir / "manifest.tsv").write_text("idx\tnir\trgb\tgrid\n" + "".join(manifest))
    print(f"wrote {len(chosen)} grids")
    print(f"manifest: {out_dir / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
