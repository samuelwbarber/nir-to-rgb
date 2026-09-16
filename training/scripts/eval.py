import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_config
from src.data.dataset import discover_pairs, split_pairs, PairedNIRRGBDataset
from src.models import build_generator
from src.training.checkpoint import load_checkpoint, save_grid
from src.eval.metrics import compute_psnr, compute_ssim, LPIPSWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--save-samples", type=int, default=16)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    cfg, _ = load_config(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    pairs = discover_pairs(cfg.data.nir_dir, cfg.data.rgb_dir, cfg.data.extensions)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio,
        seed=cfg.split.seed, scene_regex=cfg.split.scene_regex,
    )
    chosen = {"train": train_pairs, "val": val_pairs, "test": test_pairs}[args.split]
    ds = PairedNIRRGBDataset(chosen, image_size=cfg.data.image_size, train=False)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    G = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels).to(device)
    load_checkpoint(args.checkpoint, G, map_location=device)
    G.eval()

    lpips_metric = LPIPSWrapper(net=cfg.eval.lpips_net).to(device) if "lpips" in cfg.eval.metrics else None

    out_dir = Path(args.out_dir) if args.out_dir else Path("experiments") / cfg.project.experiment / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    psnr_sum = ssim_sum = lpips_sum = 0.0
    n = 0
    saved = 0
    for batch in tqdm(loader):
        nir = batch["nir"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        with torch.no_grad():
            fake = G(nir)
        bs = nir.size(0)
        psnr_sum += compute_psnr(fake, rgb) * bs
        ssim_sum += compute_ssim(fake, rgb) * bs
        if lpips_metric is not None:
            lpips_sum += lpips_metric(fake, rgb) * bs
        n += bs
        if saved < args.save_samples:
            take = min(args.save_samples - saved, bs)
            save_grid(nir[:take], fake[:take], rgb[:take], out_dir / f"sample_{saved:04d}.png")
            saved += take

    print(f"[{args.split}] n={n}")
    print(f"  PSNR  = {psnr_sum / n:.3f}")
    print(f"  SSIM  = {ssim_sum / n:.4f}")
    if lpips_metric is not None:
        print(f"  LPIPS = {lpips_sum / n:.4f}")


if __name__ == "__main__":
    main()
