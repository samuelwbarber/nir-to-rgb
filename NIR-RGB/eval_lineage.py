"""Eval the teacher lineage + new from-scratch runs on first 200 test pairs.

Metrics per (translated, GT-RGB) pair:
  - PSNR, SSIM, LPIPS                                  (pixel / perceptual)
  - DINOv2 cos, CLIP cos                               (semantic features)
  - Per-class Dice: tree, person, building, road       (downstream segmentation)
       Backbone: SegFormer-B0 finetuned on ADE20K-512
       Dice is recorded only when the GT mask for that class has >= 5% of pixels.

Entries: phase2_latest, ablation_06/best, ablation_08/best, l1boost/best, pix2pix/best.
Outputs: per_image.csv + summary.csv  (no image overlays).
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

from src.utils.config import load_config
from src.data.dataset import PairedNIRRGBDataset, discover_pairs, split_pairs
from src.models import build_generator
from src.training.checkpoint import load_checkpoint


# --- entries -----------------------------------------------------------------
ENTRIES = [
    {
        "name": "phase2_teacher_nafnet64",
        "config": PROJECT_ROOT / "experiments/_archive/phase2_teacher_nafnet64/config.yaml",
        "ckpt":   PROJECT_ROOT / "checkpoints_seed/phase2_latest.pth",
    },
    {
        "name": "ablation_06_featuremap_diverse_50_ganoff_from_best",
        "config": PROJECT_ROOT / "configs/ablation_06_featuremap_diverse_50.yaml",
        "ckpt":   PROJECT_ROOT / "experiments/ablation_06_featuremap_diverse_50_ganoff_from_best/checkpoints/best.pth",
    },
    {
        "name": "ablation_08_foundation_ensemble_aggressive",
        "config": PROJECT_ROOT / "experiments/ablation_08_foundation_ensemble_aggressive/config.yaml",
        "ckpt":   PROJECT_ROOT / "experiments/ablation_08_foundation_ensemble_aggressive/checkpoints/best.pth",
    },
    {
        "name": "ablation_08_nir_l1boost_gan_msssim_bf16_200_seed8065",
        "config": PROJECT_ROOT / "experiments/ablation_08_nir_l1boost_gan_msssim_bf16_200_seed8065/config.yaml",
        "ckpt":   PROJECT_ROOT / "experiments/ablation_08_nir_l1boost_gan_msssim_bf16_200_seed8065/checkpoints/best.pth",
    },
    {
        "name": "pix2pix_nir_200",
        "config": PROJECT_ROOT / "experiments/pix2pix_nir_200/config.yaml",
        "ckpt":   PROJECT_ROOT / "experiments/pix2pix_nir_200/checkpoints/best.pth",
    },
]
ENTRIES = [e for e in ENTRIES if e["ckpt"].exists() and e["config"].exists()]

OUT_DIR = PROJECT_ROOT / "experiments/eval_lineage"
N_SAMPLES = 200
BATCH_SIZE = 4
NUM_WORKERS = 4

# --- foundation models -------------------------------------------------------
DINOV2_HF = "facebook/dinov2-small"
CLIP_HF   = "openai/clip-vit-base-patch16"
SEGF_HF   = "nvidia/segformer-b0-finetuned-ade-512-512"

DINOV2_IN = 224
CLIP_IN   = 224
SEGF_IN   = 512
MASK_HW   = (256, 256)

# ADE20K class indices (Hugging Face SegFormer-B0-ADE & BEiT-ADE share id2label)
ADE_TREE     = 4
ADE_PERSON   = 12
ADE_ROAD     = 6
ADE_BUILDING = 1
CLASS_IDS    = {"tree": ADE_TREE, "person": ADE_PERSON, "building": ADE_BUILDING, "road": ADE_ROAD}
MIN_GT_FRAC  = 0.05  # only score a class when GT covers >=5% pixels

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])
CLIP_MEAN     = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD      = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def to_unit(x):
    return (x.clamp(-1, 1) + 1) * 0.5


def prep(x_neg11, size, mean, std, device):
    x = to_unit(x_neg11)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    m = mean.to(device).view(1, 3, 1, 1)
    s = std.to(device).view(1, 3, 1, 1)
    return (x - m) / s


def dinov2_embed(model, x_neg11, device):
    with torch.no_grad():
        return model(pixel_values=prep(x_neg11, DINOV2_IN, IMAGENET_MEAN, IMAGENET_STD, device)).pooler_output


def clip_embed(model, x_neg11, device):
    with torch.no_grad():
        vis = model.vision_model(pixel_values=prep(x_neg11, CLIP_IN, CLIP_MEAN, CLIP_STD, device))
        return model.visual_projection(vis.pooler_output)


def seg_mask(model, x_neg11, in_size, device, out_hw):
    with torch.no_grad():
        out = model(pixel_values=prep(x_neg11, in_size, IMAGENET_MEAN, IMAGENET_STD, device))
    logits = F.interpolate(out.logits, size=out_hw, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)  # (B, H, W) long


def psnr_per_image(p01, t01):
    mse = ((p01 - t01) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    return -10.0 * torch.log10(mse)


def per_image_dice(pred, gt, class_id):
    """Per-image Dice for one class. pred/gt: (B,H,W) long. Returns list of len B; entries are float in [0,1]
    or None if the class doesn't have >= MIN_GT_FRAC of GT pixels for that image."""
    out = []
    B, H, W = gt.shape
    total = H * W
    for b in range(B):
        g = (gt[b] == class_id)
        gt_frac = g.float().mean().item()
        if gt_frac < MIN_GT_FRAC:
            out.append(None)
            continue
        p = (pred[b] == class_id)
        inter = (p & g).sum().float()
        denom = p.sum().float() + g.sum().float()
        if denom.item() == 0:
            out.append(0.0)
        else:
            out.append((2.0 * inter / denom).item())
    return out


def build_test_loader(cfg_obj):
    pairs = discover_pairs(cfg_obj.data.nir_dir, cfg_obj.data.rgb_dir, cfg_obj.data.extensions)
    _, _, test = split_pairs(
        pairs,
        cfg_obj.split.train_ratio, cfg_obj.split.val_ratio, cfg_obj.split.test_ratio,
        seed=cfg_obj.split.seed, scene_regex=getattr(cfg_obj.split, "scene_regex", None),
    )
    test = test[:N_SAMPLES]
    ds = PairedNIRRGBDataset(test, image_size=cfg_obj.data.image_size,
                              train=False, aug_cfg=None, mp_cache=None)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def load_generator(entry, device):
    cfg_obj, _ = load_config(str(entry["config"]))
    G = build_generator(cfg_obj, cfg_obj.data.in_channels, cfg_obj.data.out_channels).to(device).eval()
    # Load weights only; use EMA if present (val_ema is what training reports).
    state = torch.load(str(entry["ckpt"]), map_location="cpu", weights_only=False)
    if "G_ema" in state and state["G_ema"] is not None:
        G.load_state_dict(state["G_ema"])
        ema_used = True
    else:
        G.load_state_dict(state["G"])
        ema_used = False
    return G, cfg_obj, int(state.get("epoch", -1)), ema_used


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"entries: {[e['name'] for e in ENTRIES]}", flush=True)

    from transformers import AutoModel, CLIPModel, SegformerForSemanticSegmentation
    print("loading foundation models ...", flush=True)
    dinov2 = AutoModel.from_pretrained(DINOV2_HF).eval().to(device)
    clip   = CLIPModel.from_pretrained(CLIP_HF).eval().to(device)
    segf   = SegformerForSemanticSegmentation.from_pretrained(SEGF_HF).eval().to(device)
    lpips_fn = lpips.LPIPS(net="alex").eval().to(device)
    for m in (dinov2, clip, segf, lpips_fn):
        for p in m.parameters():
            p.requires_grad_(False)

    # Use any entry's config to build the test loader — split is identical
    # (same data dirs, same seed=42, same 80/10/10).
    base_cfg_obj, _ = load_config(str(ENTRIES[0]["config"]))
    loader = build_test_loader(base_cfg_obj)
    n_test = len(loader.dataset)
    print(f"test set: {n_test} pairs (first {N_SAMPLES} of seed-42 split)", flush=True)

    # ----- pass 1: cache GT embeddings + GT segmentation masks -----
    print("\n=== caching GT-RGB foundation outputs ===", flush=True)
    gt_dinov2, gt_clip, gt_segf, gt_paths = [], [], [], []
    for i, batch in enumerate(loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt_dinov2.append(dinov2_embed(dinov2, rgb, device).cpu())
        gt_clip.append(clip_embed(clip, rgb, device).cpu())
        gt_segf.append(seg_mask(segf, rgb, SEGF_IN, device, MASK_HW).to(torch.uint8).cpu())
        gt_paths.extend(batch["rgb_path"])
        done = min((i + 1) * BATCH_SIZE, n_test)
        if (i + 1) % 10 == 0 or done == n_test:
            print(f"  GT {done}/{n_test}", flush=True)
    gt_dinov2 = torch.cat(gt_dinov2)
    gt_clip   = torch.cat(gt_clip)
    gt_segf   = torch.cat(gt_segf).long()

    # ----- per-entry pass -----
    per_image_rows = []
    summary_rows = []

    for entry in ENTRIES:
        name = entry["name"]
        print(f"\n=== {name} ===", flush=True)
        G, cfg_obj, epoch, ema_used = load_generator(entry, device)
        for p in G.parameters():
            p.requires_grad_(False)
        print(f"  loaded epoch={epoch}  (weights: {'G_ema' if ema_used else 'G'})", flush=True)

        psnrs, ssims, lpipss, dn_coses, cl_coses = [], [], [], [], []
        dice = {f"segf_{cls}": [] for cls in CLASS_IDS}

        idx = 0
        for i, batch in enumerate(loader):
            nir = batch["nir"].to(device, non_blocking=True)
            rgb = batch["rgb"].to(device, non_blocking=True)
            with torch.no_grad():
                pred = G(nir)
            B = nir.size(0)

            # pixel metrics — operate on [0,1] domain
            p01 = to_unit(pred)
            t01 = to_unit(rgb)
            psnrs.extend(psnr_per_image(p01, t01).cpu().tolist())
            with torch.no_grad():
                ssims.extend([ssim_metric(p01[b:b+1], t01[b:b+1], data_range=1.0, size_average=True).item() for b in range(B)])
                lp = lpips_fn(pred, rgb).flatten().cpu().tolist()
            lpipss.extend(lp)

            # feature embeddings vs cached GT
            pred_dn = dinov2_embed(dinov2, pred, device)
            pred_cl = clip_embed(clip, pred, device)
            gt_dn   = gt_dinov2[idx:idx + B].to(device)
            gt_cl   = gt_clip[idx:idx + B].to(device)
            dn_coses.extend(F.cosine_similarity(pred_dn, gt_dn, dim=-1).cpu().tolist())
            cl_coses.extend(F.cosine_similarity(pred_cl, gt_cl, dim=-1).cpu().tolist())

            # segmentation Dice
            pred_segf = seg_mask(segf, pred, SEGF_IN, device, MASK_HW)
            gt_segf_b = gt_segf[idx:idx + B].to(device)
            for cls_name, cls_id in CLASS_IDS.items():
                dice[f"segf_{cls_name}"].extend(per_image_dice(pred_segf, gt_segf_b, cls_id))

            idx += B
            done = min(idx, n_test)
            if (i + 1) % 10 == 0 or done == n_test:
                print(f"  pred {done}/{n_test}", flush=True)

        # per-image rows
        for j, path in enumerate(gt_paths):
            row = {"model": name, "epoch": epoch, "ema": ema_used, "rgb_path": str(path),
                   "psnr": psnrs[j], "ssim": ssims[j], "lpips": lpipss[j],
                   "dinov2_cos": dn_coses[j], "clip_cos": cl_coses[j]}
            for k, v in dice.items():
                row[k] = "" if v[j] is None else v[j]
            per_image_rows.append(row)

        # summary row — averages
        def mean_skip_none(xs):
            xs = [x for x in xs if x is not None]
            return (sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)

        s = {"model": name, "epoch": epoch, "ema": ema_used, "n": len(psnrs),
             "psnr_mean": sum(psnrs) / len(psnrs),
             "ssim_mean": sum(ssims) / len(ssims),
             "lpips_mean": sum(lpipss) / len(lpipss),
             "dinov2_cos_mean": sum(dn_coses) / len(dn_coses),
             "clip_cos_mean":   sum(cl_coses) / len(cl_coses)}
        for k, v in dice.items():
            m, n = mean_skip_none(v)
            s[f"{k}_dice"] = m
            s[f"{k}_n"] = n
        summary_rows.append(s)

        print(f"  PSNR={s['psnr_mean']:.3f} SSIM={s['ssim_mean']:.4f} LPIPS={s['lpips_mean']:.4f} "
              f"DINOv2={s['dinov2_cos_mean']:.4f} CLIP={s['clip_cos_mean']:.4f}", flush=True)
        for k in dice:
            print(f"    {k}_dice = {s[f'{k}_dice']:.4f}  (n={s[f'{k}_n']})", flush=True)

        del G
        torch.cuda.empty_cache()

    # ----- write CSVs -----
    pi_path = OUT_DIR / "per_image.csv"
    su_path = OUT_DIR / "summary.csv"
    with open(pi_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader()
        w.writerows(per_image_rows)
    with open(su_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nwrote {pi_path}\nwrote {su_path}", flush=True)


if __name__ == "__main__":
    main()
