"""Unified evaluation across raw-NIR + 5 ablations on the held-out test split.

Metrics per (prediction, GT-RGB) pair:
  - PSNR, SSIM, LPIPS (pixel / perceptual)
  - DINOv2 cosine, CLIP cosine (semantic / structural features)
  - ADE20K mIoU (SegFormer-B0 trained on ADE150 — covers trees/buildings/cars/persons/sky/road...)
  - Depth MAE in normalized [0,1] DepthAnything space (per-image min-max)
  - MUSIQ no-reference image quality (run on prediction only, no GT comparison)

Aggregate per entry:
  - CMMD in CLIP image-embedding space (RBF, σ=10 per Jayasumana et al. 2024)

raw_nir is included as a floor reference — what every model has to beat.

mp_cache (smartcrop) is OFF for all entries so every image is identical input.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import lpips
from pytorch_msssim import ssim as ssim_metric
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import PairedNIRRGBDataset, discover_pairs, split_pairs
from src.models.nafnet import NAFNet


ENTRIES = [
    {"name": "raw_nir", "config": PROJECT_ROOT / "configs/ablation_01_baseline.yaml", "ckpt": None},
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
    {
        "name": "ablation_08_foundation_ensemble_aggressive",
        "config": PROJECT_ROOT / "configs/ablation_08_foundation_ensemble_aggressive.yaml",
        "ckpt": PROJECT_ROOT / "experiments/ablation_08_foundation_ensemble_aggressive/checkpoints/best.pth",
    },
    {
        "name": "ablation_09_backtranslation",
        "config": PROJECT_ROOT / "configs/ablation_09_backtranslation.yaml",
        "ckpt": PROJECT_ROOT / "experiments/ablation_09_backtranslation/checkpoints/best.pth",
    },
]

# Skip entries whose checkpoint doesn't exist yet (lets eval_all.py run cleanly
# at any point during the pipeline — only completed runs are evaluated).
ENTRIES = [e for e in ENTRIES if e["ckpt"] is None or e["ckpt"].exists()]

OUT_DIR = PROJECT_ROOT / "experiments/eval_all_v2"
BATCH_SIZE = 4
NUM_WORKERS = 4

DINOV2_HF = "facebook/dinov2-small"
CLIP_HF = "openai/clip-vit-base-patch16"
ADE_HF = "nvidia/segformer-b0-finetuned-ade-512-512"
DEPTH_HF = "LiheYoung/depth-anything-small-hf"

DINOV2_IN = 224
CLIP_IN = 224
ADE_IN = 512
ADE_NUM_CLASSES = 150
DEPTH_IN = 224
MASK_HW = (256, 256)
DEPTH_HW = (256, 256)
CMMD_SIGMA = 10.0

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def to_unit(x):
    return (x.clamp(-1, 1) + 1) * 0.5


def _normalize(x01, mean, std, device):
    m = mean.to(device).view(1, 3, 1, 1)
    s = std.to(device).view(1, 3, 1, 1)
    return (x01 - m) / s


def prep(x_neg11, size, mean, std, device):
    x = to_unit(x_neg11)
    x = F.interpolate(x, size=size, mode="bilinear",
                      align_corners=False, antialias=True)
    return _normalize(x, mean, std, device)


def dinov2_embed(model, x_neg11, device):
    with torch.no_grad():
        out = model(pixel_values=prep(x_neg11, DINOV2_IN, IMAGENET_MEAN, IMAGENET_STD, device))
    return out.pooler_output  # (B, 384)


def clip_embed(model, x_neg11, device):
    with torch.no_grad():
        vis = model.vision_model(pixel_values=prep(x_neg11, CLIP_IN, CLIP_MEAN, CLIP_STD, device))
        emb = model.visual_projection(vis.pooler_output)
    return emb  # (B, 512)


def ade_mask(model, x_neg11, device, out_hw):
    with torch.no_grad():
        out = model(pixel_values=prep(x_neg11, ADE_IN, IMAGENET_MEAN, IMAGENET_STD, device))
    logits = F.interpolate(out.logits, size=out_hw, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)  # (B, H, W) long, in [0, 149]


def depth_map(model, x_neg11, device, out_hw):
    with torch.no_grad():
        out = model(pixel_values=prep(x_neg11, DEPTH_IN, IMAGENET_MEAN, IMAGENET_STD, device))
    d = out.predicted_depth  # (B, H, W)
    if d.dim() == 3:
        d = d.unsqueeze(1)  # (B, 1, H, W)
    d = F.interpolate(d, size=out_hw, mode="bilinear", align_corners=False)
    return d.squeeze(1)  # (B, H, W)


def normalize_per_image(d, eps=1e-8):
    # d: (B, H, W). Per-image min-max to [0, 1].
    B = d.size(0)
    flat = d.reshape(B, -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    nd = (flat - mn) / (mx - mn + eps)
    return nd.reshape_as(d)


def psnr_per_image(pred01, tgt01):
    mse = ((pred01 - tgt01) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    return -10.0 * torch.log10(mse)


def per_image_miou(pred, gt, num_classes):
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        g = (gt == c)
        if not (p.any() or g.any()):
            continue
        inter = (p & g).sum().float()
        union = (p | g).sum().float()
        ious.append((inter / union).item())
    return sum(ious) / len(ious) if ious else 1.0


def cmmd_rbf(x, y, sigma=CMMD_SIGMA, chunk=512):
    # x: (n, D), y: (m, D). Returns scalar MMD² estimate with Gaussian kernel.
    x = x.float(); y = y.float()
    two_sigma_sq = 2.0 * sigma * sigma

    def sum_kernel(a, b):
        s = 0.0
        n = a.size(0); m = b.size(0)
        for i in range(0, n, chunk):
            ai = a[i:i + chunk]
            d2 = (ai.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)
            s += torch.exp(-d2 / two_sigma_sq).sum().item()
        return s

    n = x.size(0); m = y.size(0)
    kxx = sum_kernel(x, x) / (n * n)
    kyy = sum_kernel(y, y) / (m * m)
    kxy = sum_kernel(x, y) / (n * m)
    return kxx + kyy - 2.0 * kxy


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
    ds = PairedNIRRGBDataset(test, image_size=data["image_size"],
                              train=False, aug_cfg=None, mp_cache=None)
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    from transformers import (
        AutoModel,
        CLIPModel,
        SegformerForSemanticSegmentation,
        AutoModelForDepthEstimation,
    )
    import pyiqa

    print(f"loading foundation models ...")
    dinov2 = AutoModel.from_pretrained(DINOV2_HF).eval().to(device)
    clip = CLIPModel.from_pretrained(CLIP_HF).eval().to(device)
    ade = SegformerForSemanticSegmentation.from_pretrained(ADE_HF).eval().to(device)
    depth = AutoModelForDepthEstimation.from_pretrained(DEPTH_HF).eval().to(device)
    musiq = pyiqa.create_metric("musiq", device=device)
    lpips_fn = lpips.LPIPS(net="alex").eval().to(device)
    for m in [dinov2, clip, ade, depth, lpips_fn]:
        for p in m.parameters():
            p.requires_grad_(False)

    baseline_cfg = yaml.safe_load(open(ENTRIES[1]["config"]))  # ablation_01
    loader = build_test_loader(baseline_cfg)
    n_test = len(loader.dataset)
    print(f"test set: {n_test} pairs (mp_cache OFF)")

    # ---- Pass 1: cache GT-RGB features/masks/depth/musiq ----
    print("\n=== caching GT-RGB foundation outputs ===")
    gt_dinov2 = []
    gt_clip = []
    gt_mask = []
    gt_depth = []
    gt_musiq = []
    canonical_paths = []
    for i, batch in enumerate(loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt_dinov2.append(dinov2_embed(dinov2, rgb, device).cpu())
        gt_clip.append(clip_embed(clip, rgb, device).cpu())
        gt_mask.append(ade_mask(ade, rgb, device, MASK_HW).to(torch.uint8).cpu())
        gt_depth.append(normalize_per_image(depth_map(depth, rgb, device, DEPTH_HW)).cpu())
        with torch.no_grad():
            gt_musiq.append(musiq(to_unit(rgb)).flatten().cpu())
        canonical_paths.extend(batch["rgb_path"])
        if (i + 1) % 25 == 0 or (i + 1) == len(loader):
            done = min((i + 1) * BATCH_SIZE, n_test)
            print(f"  GT {done}/{n_test}", flush=True)
    gt_dinov2 = torch.cat(gt_dinov2)  # (N, 384)
    gt_clip = torch.cat(gt_clip)  # (N, 512)
    gt_mask = torch.cat(gt_mask)  # (N, H, W) uint8
    gt_depth = torch.cat(gt_depth)  # (N, H, W) fp32
    gt_musiq = torch.cat(gt_musiq)  # (N,)
    print(f"cached: dinov2 {tuple(gt_dinov2.shape)}, clip {tuple(gt_clip.shape)}, mask {tuple(gt_mask.shape)}, depth {tuple(gt_depth.shape)}")
    print(f"GT MUSIQ: {gt_musiq.mean():.2f} ± {gt_musiq.std(unbiased=False):.2f}")

    # Also record GT pixel-domain metrics — trivially PSNR=inf, SSIM=1, LPIPS=0 — skip.

    # ---- Per-entry pass ----
    results = {}
    pred_clip_embeds = {}  # for CMMD
    for entry in ENTRIES:
        name = entry["name"]
        print(f"\n=== {name} ===")
        cfg = yaml.safe_load(open(entry["config"]))
        G = None
        epoch = None
        if entry["ckpt"] is not None:
            G, epoch = build_generator(cfg, entry["ckpt"], device)
            print(f"  loaded epoch={epoch}")
        else:
            print("  (raw NIR — no generator)")

        psnrs, ssims, lpipss = [], [], []
        dn_coses, cl_coses, mious, dmaes, mscores = [], [], [], [], []
        clip_embeds_for_cmmd = []
        idx = 0
        for i, batch in enumerate(loader):
            nir = batch["nir"].to(device, non_blocking=True)
            rgb = batch["rgb"].to(device, non_blocking=True)
            with torch.no_grad():
                pred = nir if G is None else G(nir)
            B = nir.size(0)

            # Pixel-domain
            pred_u, rgb_u = to_unit(pred), to_unit(rgb)
            psnrs.extend(psnr_per_image(pred_u, rgb_u).cpu().tolist())
            ssims.extend(ssim_metric(pred_u, rgb_u, data_range=1.0,
                                     size_average=False).cpu().tolist())
            with torch.no_grad():
                lpipss.extend(
                    lpips_fn(pred.clamp(-1, 1), rgb.clamp(-1, 1)).flatten().cpu().tolist()
                )

            # Foundation features
            p_dn = dinov2_embed(dinov2, pred, device)
            p_cl = clip_embed(clip, pred, device)
            p_ma = ade_mask(ade, pred, device, MASK_HW)
            p_de = normalize_per_image(depth_map(depth, pred, device, DEPTH_HW))
            with torch.no_grad():
                p_mq = musiq(to_unit(pred)).flatten()

            g_dn = gt_dinov2[idx:idx + B].to(device)
            g_cl = gt_clip[idx:idx + B].to(device)
            g_de = gt_depth[idx:idx + B].to(device)

            dn_coses.extend(F.cosine_similarity(p_dn, g_dn, dim=-1).cpu().tolist())
            cl_coses.extend(F.cosine_similarity(p_cl, g_cl, dim=-1).cpu().tolist())
            for b in range(B):
                mious.append(per_image_miou(p_ma[b].long(),
                                            gt_mask[idx + b].to(device).long(),
                                            ADE_NUM_CLASSES))
            dmaes.extend((p_de - g_de).abs().mean(dim=(1, 2)).cpu().tolist())
            mscores.extend(p_mq.cpu().tolist())

            clip_embeds_for_cmmd.append(p_cl.cpu())
            idx += B
            if (i + 1) % 25 == 0 or (i + 1) == len(loader):
                done = min((i + 1) * BATCH_SIZE, n_test)
                print(f"  {done}/{n_test}", flush=True)

        pred_clip_embeds[name] = torch.cat(clip_embeds_for_cmmd)

        def stat(v):
            t = torch.tensor(v)
            return t.mean().item(), t.std(unbiased=False).item()

        psnr_m, psnr_s = stat(psnrs)
        ssim_m, ssim_s = stat(ssims)
        lp_m, lp_s = stat(lpipss)
        dn_m, dn_s = stat(dn_coses)
        cl_m, cl_s = stat(cl_coses)
        mi_m, mi_s = stat(mious)
        dm_m, dm_s = stat(dmaes)
        ms_m, ms_s = stat(mscores)

        print(f"  PSNR  {psnr_m:7.4f} ± {psnr_s:.4f}")
        print(f"  SSIM  {ssim_m:7.4f} ± {ssim_s:.4f}")
        print(f"  LPIPS {lp_m:7.4f} ± {lp_s:.4f}")
        print(f"  DINOv2 cos  {dn_m:7.4f} ± {dn_s:.4f}")
        print(f"  CLIP   cos  {cl_m:7.4f} ± {cl_s:.4f}")
        print(f"  ADE20K mIoU {mi_m:7.4f} ± {mi_s:.4f}")
        print(f"  Depth MAE   {dm_m:7.4f} ± {dm_s:.4f}")
        print(f"  MUSIQ       {ms_m:7.4f} ± {ms_s:.4f}")

        results[name] = {
            "epoch": epoch if epoch is not None else "",
            "psnr": psnrs, "ssim": ssims, "lpips": lpipss,
            "dinov2_cos": dn_coses, "clip_cos": cl_coses,
            "ade_miou": mious, "depth_mae": dmaes, "musiq": mscores,
        }
        if G is not None:
            del G
            torch.cuda.empty_cache()

    # ---- CMMD per entry ----
    print("\n=== computing CMMD in CLIP space (σ={}) ===".format(CMMD_SIGMA))
    gt_clip_cpu = gt_clip.cpu()
    cmmd = {}
    for name in [e["name"] for e in ENTRIES]:
        v = cmmd_rbf(pred_clip_embeds[name], gt_clip_cpu)
        cmmd[name] = v
        print(f"  {name:<55} CMMD = {v:.6f}")

    # ---- CSV writes ----
    METRICS_PER_IMG = ["psnr", "ssim", "lpips", "dinov2_cos", "clip_cos",
                       "ade_miou", "depth_mae", "musiq"]

    cols = []
    for e in ENTRIES:
        for mk in METRICS_PER_IMG:
            cols.append(f"{e['name']}_{mk}")

    per_img = OUT_DIR / "per_image.csv"
    with open(per_img, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rgb_path"] + cols)
        for i, path in enumerate(canonical_paths):
            row = [path]
            for e in ENTRIES:
                r = results[e["name"]]
                for mk in METRICS_PER_IMG:
                    row.append(r[mk][i])
            w.writerow(row)
    print(f"\nwrote {per_img}")

    summary = OUT_DIR / "summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        header = ["entry", "epoch", "n"]
        for mk in METRICS_PER_IMG:
            header += [f"{mk}_mean", f"{mk}_std"]
        header += ["cmmd"]
        w.writerow(header)
        for e in ENTRIES:
            name = e["name"]
            r = results[name]
            row = [name, r["epoch"], len(r["psnr"])]
            for mk in METRICS_PER_IMG:
                t = torch.tensor(r[mk])
                row += [f"{t.mean().item():.6f}", f"{t.std(unbiased=False).item():.6f}"]
            row.append(f"{cmmd[name]:.6f}")
            w.writerow(row)
    print(f"wrote {summary}")

    # Pretty table
    print(f"\n=== SUMMARY (n={n_test}) — GT MUSIQ ref = {gt_musiq.mean():.2f} ===")
    hdr = f"{'entry':<55} {'ep':>4} {'PSNR':>7} {'SSIM':>6} {'LPIPS':>6} {'DINOv2':>7} {'CLIP':>6} {'ADE':>6} {'D-MAE':>6} {'MUSIQ':>6} {'CMMD':>8}"
    print(hdr)
    for e in ENTRIES:
        name = e["name"]; r = results[name]
        means = {mk: torch.tensor(r[mk]).mean().item() for mk in METRICS_PER_IMG}
        ep = r["epoch"] if r["epoch"] != "" else "-"
        print(f"{name:<55} {str(ep):>4} {means['psnr']:7.3f} {means['ssim']:6.3f} {means['lpips']:6.3f} {means['dinov2_cos']:7.4f} {means['clip_cos']:6.4f} {means['ade_miou']:6.4f} {means['depth_mae']:6.4f} {means['musiq']:6.2f} {cmmd[name]:8.5f}")


if __name__ == "__main__":
    main()
