"""Downstream-task eval: DINOv2 cosine similarity + Cityscapes mIoU.

For each of the 5 ablations, generate fake-RGB from the test-split NIR and
compare against the GT RGB along two content-agnostic axes:

  - DINOv2 (ViT-S/14) CLS-embedding cosine similarity per pair  [perceptual]
  - SegFormer-B0 trained on Cityscapes: per-image mIoU vs GT mask  [downstream]

Cityscapes covers car / building / vegetation / road / sky / person /
sidewalk / fence / pole / etc. — much better suited to mixed street/outdoor
content than COCO detection, which has no classes for trees / houses /
buildings.

Test split is identical to eval_compare.py (seed=42, mp_cache=OFF).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
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

OUT_DIR = PROJECT_ROOT / "experiments/eval_downstream_01_02_03_04_06"
BATCH_SIZE = 4
NUM_WORKERS = 4

DINOV2_HF = "facebook/dinov2-small"
DINOV2_INPUT = 224
SEG_HF = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
SEG_INPUT = 512           # square upscale of 256x256 source
SEG_NUM_CLASSES = 19
MASK_CACHE_HW = (256, 256)  # resolution we store + compare masks at

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def to_unit(x):
    return (x.clamp(-1, 1) + 1) * 0.5


def imagenet_normalize(x01, device):
    mean = IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
    std = IMAGENET_STD.to(device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def prep_dinov2(x_neg11, device):
    x = to_unit(x_neg11)
    x = F.interpolate(x, size=DINOV2_INPUT, mode="bilinear",
                      align_corners=False, antialias=True)
    return imagenet_normalize(x, device)


def prep_seg(x_neg11, device):
    x = to_unit(x_neg11)
    x = F.interpolate(x, size=SEG_INPUT, mode="bilinear",
                      align_corners=False, antialias=True)
    return imagenet_normalize(x, device)


def dinov2_embed(model, x_neg11, device):
    with torch.no_grad():
        out = model(pixel_values=prep_dinov2(x_neg11, device))
    return out.pooler_output  # (B, 384)


def seg_predict_mask(model, x_neg11, device, out_hw):
    with torch.no_grad():
        out = model(pixel_values=prep_seg(x_neg11, device))
    logits = out.logits  # (B, num_classes, h/4, w/4)
    logits = F.interpolate(logits, size=out_hw, mode="bilinear",
                           align_corners=False)
    return logits.argmax(dim=1)  # (B, H, W) long


def cos_sim(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def per_image_miou(pred, gt, num_classes):
    # both (H, W) long. Average IoU over classes present in either mask.
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

    from transformers import AutoModel, SegformerForSemanticSegmentation

    print(f"loading {DINOV2_HF} ...")
    dinov2 = AutoModel.from_pretrained(DINOV2_HF).eval().to(device)
    for p in dinov2.parameters():
        p.requires_grad_(False)
    print(f"loading {SEG_HF} ...")
    seg = SegformerForSemanticSegmentation.from_pretrained(SEG_HF).eval().to(device)
    for p in seg.parameters():
        p.requires_grad_(False)

    baseline_cfg = yaml.safe_load(open(MODELS[0]["config"]))
    loader = build_test_loader(baseline_cfg)
    n_test = len(loader.dataset)
    print(f"test set: {n_test} pairs (mp_cache OFF for all)")

    # Pass 1: cache GT-RGB DINOv2 embeddings and Cityscapes masks
    print("\n=== caching GT-RGB DINOv2 embeddings + SegFormer masks ===")
    gt_embeds = []
    gt_masks = []
    canonical_paths = []
    for i, batch in enumerate(loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        e = dinov2_embed(dinov2, rgb, device).cpu()
        m = seg_predict_mask(seg, rgb, device, MASK_CACHE_HW).to(torch.uint8).cpu()
        gt_embeds.append(e)
        gt_masks.append(m)
        canonical_paths.extend(batch["rgb_path"])
        if (i + 1) % 25 == 0 or (i + 1) == len(loader):
            done = min((i + 1) * BATCH_SIZE, n_test)
            print(f"  GT {done}/{n_test}", flush=True)
    gt_embeds = torch.cat(gt_embeds, dim=0)
    gt_masks = torch.cat(gt_masks, dim=0)
    print(f"cached: embeddings {tuple(gt_embeds.shape)}, masks {tuple(gt_masks.shape)} ({gt_masks.element_size() * gt_masks.numel() / 1e6:.0f} MB)")

    # Per-model passes
    results = {}
    for m in MODELS:
        name = m["name"]
        print(f"\n=== {name} ===")
        cfg = yaml.safe_load(open(m["config"]))
        G, epoch = build_generator(cfg, m["ckpt"], device)
        print(f"  loaded epoch={epoch}")

        per_cos, per_miou = [], []
        idx = 0
        for i, batch in enumerate(loader):
            nir = batch["nir"].to(device, non_blocking=True)
            with torch.no_grad():
                fake = G(nir)
            B = nir.size(0)
            f_e = dinov2_embed(dinov2, fake, device)
            g_e = gt_embeds[idx:idx + B].to(device)
            per_cos.extend(cos_sim(f_e, g_e).cpu().tolist())
            f_m = seg_predict_mask(seg, fake, device, MASK_CACHE_HW)
            for b in range(B):
                per_miou.append(
                    per_image_miou(f_m[b].long(),
                                   gt_masks[idx + b].to(device).long(),
                                   SEG_NUM_CLASSES)
                )
            idx += B
            if (i + 1) % 25 == 0 or (i + 1) == len(loader):
                done = min((i + 1) * BATCH_SIZE, n_test)
                print(f"  {done}/{n_test}", flush=True)

        ct = torch.tensor(per_cos); mt = torch.tensor(per_miou)
        print(f"  DINOv2 cos  {ct.mean():.4f} ± {ct.std(unbiased=False):.4f}")
        print(f"  CS mIoU     {mt.mean():.4f} ± {mt.std(unbiased=False):.4f}")
        results[name] = {"epoch": epoch, "cos": per_cos, "miou": per_miou}
        del G
        torch.cuda.empty_cache()

    # Write CSVs
    cols = []
    for m in MODELS:
        cols += [f"{m['name']}_dinov2_cos", f"{m['name']}_cs_miou"]
    per_img = OUT_DIR / "per_image_downstream.csv"
    with open(per_img, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rgb_path"] + cols)
        for i, path in enumerate(canonical_paths):
            row = [path]
            for m in MODELS:
                r = results[m["name"]]
                row += [r["cos"][i], r["miou"][i]]
            w.writerow(row)
    print(f"\nwrote {per_img}")

    summary = OUT_DIR / "summary_downstream.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "epoch", "n",
                    "dinov2_cos_mean", "dinov2_cos_std",
                    "cs_miou_mean", "cs_miou_std"])
        for m in MODELS:
            r = results[m["name"]]
            ct = torch.tensor(r["cos"]); mt = torch.tensor(r["miou"])
            w.writerow([
                m["name"], r["epoch"], len(ct),
                f"{ct.mean():.6f}", f"{ct.std(unbiased=False):.6f}",
                f"{mt.mean():.6f}", f"{mt.std(unbiased=False):.6f}",
            ])
    print(f"wrote {summary}")

    print(f"\n=== SUMMARY (n={n_test}) ===")
    print(f"{'model':<55} {'epoch':>5} {'DINOv2 cos':>11} {'CS mIoU':>9}")
    for m in MODELS:
        r = results[m["name"]]
        ct = torch.tensor(r["cos"]); mt = torch.tensor(r["miou"])
        print(f"{m['name']:<55} {r['epoch']:>5} {ct.mean():11.4f} {mt.mean():9.4f}")


if __name__ == "__main__":
    main()
