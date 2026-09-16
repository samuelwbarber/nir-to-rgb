"""Benchmark a checkpoint against the raw (untranslated) input baseline.

Scores both the model output G(x) and the raw input x against the RGB ground
truth on PSNR/SSIM/LPIPS plus the foundation-model embedding cosines
(clip_cos / dinov2_cos) that this recipe actually optimises. Answers the
question "how much better is the translation than feeding raw NIR/thermal
straight into an RGB-trained model?".
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_config
from src.data.dataset import discover_pairs, split_pairs, PairedNIRRGBDataset
from src.models import build_generator
from src.training.checkpoint import load_checkpoint
from src.eval.metrics import compute_psnr, compute_ssim, LPIPSWrapper
from src.training.losses import FeatureMapLoss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()

    cfg, _ = load_config(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    bs = args.batch_size or cfg.train.batch_size

    pairs = discover_pairs(cfg.data.nir_dir, cfg.data.rgb_dir, cfg.data.extensions)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio,
        seed=cfg.split.seed, scene_regex=cfg.split.scene_regex,
    )
    chosen = {"train": train_pairs, "val": val_pairs, "test": test_pairs}[args.split]
    ds = PairedNIRRGBDataset(chosen, image_size=cfg.data.image_size, train=False)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.train.num_workers)

    G = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels).to(device)
    load_checkpoint(args.checkpoint, G, map_location=device)
    G.eval()

    lpips_metric = LPIPSWrapper(net=cfg.eval.lpips_net).to(device)

    fm = None
    extractors_cfg = getattr(cfg.loss, "feature_map_extractors", None)
    if extractors_cfg:
        input_size = int(getattr(cfg.loss, "feature_map_input_size", 224))
        fm = FeatureMapLoss(extractors_cfg=extractors_cfg, input_size=input_size).to(device)

    acc = {k: {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "cos": {}} for k in ("raw", "translated")}
    n = 0

    @torch.no_grad()
    def cosines(x, rgb):
        if fm is None:
            return {}
        xe = fm.pooled_embeddings(x)
        re = fm.pooled_embeddings(rgb)
        return {k: float(F.cosine_similarity(xe[k].float(), re[k].float(), dim=-1).mean()) for k in xe}

    for batch in tqdm(loader, desc=args.split):
        nir = batch["nir"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        with torch.no_grad():
            fake = G(nir)
        b = nir.size(0)
        for name, img in (("raw", nir), ("translated", fake)):
            acc[name]["psnr"] += compute_psnr(img, rgb) * b
            acc[name]["ssim"] += compute_ssim(img, rgb) * b
            acc[name]["lpips"] += lpips_metric(img, rgb) * b
            for k, v in cosines(img, rgb).items():
                acc[name]["cos"][k] = acc[name]["cos"].get(k, 0.0) + v * b
        n += b

    cos_keys = sorted(acc["raw"]["cos"].keys())
    print(f"\n=== {args.split} (n={n}) — raw vs translated ===")
    header = f"{'metric':<22}{'raw':>12}{'translated':>14}{'delta':>12}"
    print(header)
    print("-" * len(header))
    rows = [("PSNR", "psnr", False), ("SSIM", "ssim", False), ("LPIPS", "lpips", True)]
    for label, key, lower_better in rows:
        r = acc["raw"][key] / n
        t = acc["translated"][key] / n
        d = t - r
        arrow = "↓" if lower_better else "↑"
        print(f"{label+' '+arrow:<22}{r:>12.4f}{t:>14.4f}{d:>+12.4f}")
    for k in cos_keys:
        r = acc["raw"]["cos"][k] / n
        t = acc["translated"]["cos"][k] / n
        print(f"{k+'_cos ↑':<22}{r:>12.4f}{t:>14.4f}{t-r:>+12.4f}")


if __name__ == "__main__":
    main()
