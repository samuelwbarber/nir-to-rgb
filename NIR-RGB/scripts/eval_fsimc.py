"""Compute FSIMc on a val/test split using a trained checkpoint."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import piq

from src.utils.config import load_config
from src.data.dataset import discover_pairs, split_pairs, PairedNIRRGBDataset
from src.models import build_generator
from src.training.checkpoint import load_checkpoint


def to_01(t):
    return t.clamp(-1, 1) * 0.5 + 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    cfg, _ = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = discover_pairs(cfg.data.nir_dir, cfg.data.rgb_dir, cfg.data.extensions)
    train_p, val_p, test_p = split_pairs(
        pairs, cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio,
        seed=cfg.split.seed, scene_regex=cfg.split.scene_regex,
    )
    chosen = {"train": train_p, "val": val_p, "test": test_p}[args.split]
    ds = PairedNIRRGBDataset(chosen, image_size=cfg.data.image_size, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    G = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels).to(device)
    load_checkpoint(args.checkpoint, G, map_location=device)
    G.eval()

    total_fsimc = 0.0
    n = 0
    for batch in tqdm(loader, desc=f"FSIMc ({args.split})"):
        nir = batch["nir"].to(device)
        rgb = batch["rgb"].to(device)
        with torch.no_grad():
            fake = G(nir)
        fake_01 = to_01(fake)
        rgb_01 = to_01(rgb)
        fsimc = piq.fsim(fake_01, rgb_01, chromatic=True, data_range=1.0)
        total_fsimc += float(fsimc) * nir.size(0)
        n += nir.size(0)

    print(f"\nFSIMc on {args.split} (n={n}): {total_fsimc / n:.4f}")


if __name__ == "__main__":
    main()
