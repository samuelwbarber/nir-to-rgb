import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2

from src.utils.config import load_config
from src.data.dataset import discover_pairs, split_pairs, PairedNIRRGBDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/baseline.yaml")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--out", type=str, default="experiments/_dataset_check.png")
    args = p.parse_args()

    cfg, _ = load_config(args.config)
    pairs = discover_pairs(cfg.data.nir_dir, cfg.data.rgb_dir, cfg.data.extensions)
    print(f"discovered {len(pairs)} pairs")
    if not pairs:
        print("no pairs found. check that nir_dir and rgb_dir contain files with matching stems.")
        return

    train, val, test = split_pairs(
        pairs, cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio,
        seed=cfg.split.seed, scene_regex=cfg.split.scene_regex,
    )
    print(f"split: train={len(train)} val={len(val)} test={len(test)}")

    ds = PairedNIRRGBDataset(train, image_size=cfg.data.image_size, train=True, aug_cfg=vars(cfg.augment))
    n = min(args.n_samples, len(ds))
    rows = []
    for i in range(n):
        item = ds[i]
        nir = ((item["nir"].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        rgb = ((item["rgb"].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        rows.append(np.concatenate([nir, rgb], axis=1))
    grid = np.concatenate(rows, axis=0)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid_bgr)
    print(f"wrote sample grid to {out}")


if __name__ == "__main__":
    main()
