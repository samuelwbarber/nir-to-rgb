"""Compare PSNR / SSIM / LPIPS for baseline, 02, 03, 04, 06 on the held-out
test split.

All five share NAFNet64 architecture and the same seed-42 80/10/10 random
split of data/nir + data/rgb. mp_cache (smartcrop) is forced OFF for all
models so every model sees identical input frames.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import yaml
import lpips
from pytorch_msssim import ssim as ssim_metric
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import PairedNIRRGBDataset, discover_pairs, split_pairs
from src.models.nafnet import NAFNet


MODELS = [
    {
        "name": "ablation_01_baseline",
        "config": PROJECT_ROOT / "configs/ablation_01_baseline.yaml",
        "ckpt": Path("/vol/bitbucket/sb1522/vm-backup/experiments/ablation_01_baseline/checkpoints/best.pth"),
    },
    {
        "name": "ablation_02_smartcrop",
        "config": PROJECT_ROOT / "configs/ablation_02_smartcrop.yaml",
        "ckpt": Path("/vol/bitbucket/sb1522/vm-backup/experiments/ablation_02_smartcrop/checkpoints/best.pth"),
    },
    {
        "name": "ablation_03_region_l1_l1max_50",
        "config": PROJECT_ROOT / "configs/ablation_03_region_l1.yaml",
        "ckpt": PROJECT_ROOT / "experiments/ablation_03_region_l1_l1max_50/checkpoints/best.pth",
    },
    {
        "name": "ablation_04_nir_grad_l1max_50",
        "config": PROJECT_ROOT / "configs/ablation_04_nir_grad.yaml",
        "ckpt": PROJECT_ROOT / "experiments/ablation_04_nir_grad_l1max_50/checkpoints/best.pth",
    },
    {
        "name": "ablation_06_featuremap_diverse_50_ganoff_from_best",
        "config": PROJECT_ROOT / "configs/ablation_06_featuremap_diverse_50.yaml",
        "ckpt": PROJECT_ROOT / "experiments/ablation_06_featuremap_diverse_50_ganoff_from_best/checkpoints/best.pth",
    },
]

OUT_DIR = PROJECT_ROOT / "experiments/eval_compare_01_02_03_04_06"
BATCH_SIZE = 8
NUM_WORKERS = 4


def build_test_loader(cfg):
    data = cfg["data"]
    split = cfg["split"]
    pairs = discover_pairs(
        PROJECT_ROOT / data["nir_dir"],
        PROJECT_ROOT / data["rgb_dir"],
        data["extensions"],
    )
    _, _, test = split_pairs(
        pairs,
        split["train_ratio"], split["val_ratio"], split["test_ratio"],
        seed=split["seed"], scene_regex=split.get("scene_regex"),
    )
    ds = PairedNIRRGBDataset(
        test,
        image_size=data["image_size"],
        train=False,
        aug_cfg=None,
        mp_cache=None,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def build_generator(cfg, ckpt_path, device):
    g = cfg["model"]["generator"]
    G = NAFNet(
        in_channels=cfg["data"]["in_channels"],
        out_channels=cfg["data"]["out_channels"],
        width=g["width"],
        enc_blk_nums=tuple(g["enc_blk_nums"]),
        middle_blk_num=g["middle_blk_num"],
        dec_blk_nums=tuple(g["dec_blk_nums"]),
        dropout=g.get("dropout", 0.0),
        global_residual=g.get("global_residual", False),
        output_tanh=g.get("output_tanh", True),
    )
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    G.load_state_dict(state["G"])
    return G.eval().to(device), state.get("epoch", -1)


def to_unit(x):
    return (x.clamp(-1, 1) + 1) * 0.5


def psnr_per_image(pred01, tgt01):
    mse = ((pred01 - tgt01) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    return -10.0 * torch.log10(mse)


def evaluate_model(G, loader, lpips_fn, device):
    psnrs, ssims, lpips_scores, paths = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            nir = batch["nir"].to(device, non_blocking=True)
            rgb = batch["rgb"].to(device, non_blocking=True)
            fake = G(nir)
            fake_u, rgb_u = to_unit(fake), to_unit(rgb)
            psnrs.extend(psnr_per_image(fake_u, rgb_u).cpu().tolist())
            ssims.extend(ssim_metric(fake_u, rgb_u, data_range=1.0,
                                     size_average=False).cpu().tolist())
            lpips_scores.extend(
                lpips_fn(fake.clamp(-1, 1), rgb.clamp(-1, 1)).flatten().cpu().tolist()
            )
            paths.extend(batch["rgb_path"])
            if (i + 1) % 20 == 0 or (i + 1) == len(loader):
                done = min((i + 1) * BATCH_SIZE, len(loader.dataset))
                print(f"  {done}/{len(loader.dataset)}", flush=True)
    return psnrs, ssims, lpips_scores, paths


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    lpips_fn = lpips.LPIPS(net="alex").eval().to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # Test split is identical across all 5 (same data, same seed=42).
    baseline_cfg = yaml.safe_load(open(MODELS[0]["config"]))
    loader = build_test_loader(baseline_cfg)
    n_test = len(loader.dataset)
    print(f"test set: {n_test} pairs (mp_cache=OFF for all models)")

    results = {}
    canonical_paths = None
    for m in MODELS:
        name = m["name"]
        print(f"\n=== {name} ===")
        print(f"  ckpt: {m['ckpt']}")
        cfg = yaml.safe_load(open(m["config"]))
        G, epoch = build_generator(cfg, m["ckpt"], device)
        print(f"  loaded epoch={epoch}")
        psnrs, ssims, lps, paths = evaluate_model(G, loader, lpips_fn, device)
        if canonical_paths is None:
            canonical_paths = paths
        else:
            assert paths == canonical_paths, f"order mismatch for {name}"
        ps_t = torch.tensor(psnrs); ss_t = torch.tensor(ssims); lp_t = torch.tensor(lps)
        print(f"  PSNR  {ps_t.mean():7.4f} ± {ps_t.std(unbiased=False):.4f}")
        print(f"  SSIM  {ss_t.mean():7.4f} ± {ss_t.std(unbiased=False):.4f}")
        print(f"  LPIPS {lp_t.mean():7.4f} ± {lp_t.std(unbiased=False):.4f}")
        results[name] = {
            "epoch": epoch,
            "psnr": psnrs, "ssim": ssims, "lpips": lps,
        }
        del G
        torch.cuda.empty_cache()

    # per_image.csv
    per_img = OUT_DIR / "per_image.csv"
    cols = []
    for m in MODELS:
        cols += [f"{m['name']}_psnr", f"{m['name']}_ssim", f"{m['name']}_lpips"]
    with open(per_img, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rgb_path"] + cols)
        for i, path in enumerate(canonical_paths):
            row = [path]
            for m in MODELS:
                r = results[m["name"]]
                row += [r["psnr"][i], r["ssim"][i], r["lpips"][i]]
            w.writerow(row)
    print(f"\nwrote {per_img}")

    # summary.csv
    summary = OUT_DIR / "summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "epoch", "n",
                    "psnr_mean", "psnr_std",
                    "ssim_mean", "ssim_std",
                    "lpips_mean", "lpips_std"])
        for m in MODELS:
            r = results[m["name"]]
            ps = torch.tensor(r["psnr"]); ss = torch.tensor(r["ssim"]); lp = torch.tensor(r["lpips"])
            w.writerow([
                m["name"], r["epoch"], len(ps),
                f"{ps.mean():.6f}", f"{ps.std(unbiased=False):.6f}",
                f"{ss.mean():.6f}", f"{ss.std(unbiased=False):.6f}",
                f"{lp.mean():.6f}", f"{lp.std(unbiased=False):.6f}",
            ])
    print(f"wrote {summary}")

    print("\n=== SUMMARY (mp_cache OFF, n={}) ===".format(n_test))
    print(f"{'model':<55} {'epoch':>5} {'PSNR':>9} {'SSIM':>9} {'LPIPS':>9}")
    for m in MODELS:
        r = results[m["name"]]
        ps = torch.tensor(r["psnr"]); ss = torch.tensor(r["ssim"]); lp = torch.tensor(r["lpips"])
        print(f"{m['name']:<55} {r['epoch']:>5} {ps.mean():9.4f} {ss.mean():9.4f} {lp.mean():9.4f}")


if __name__ == "__main__":
    main()
