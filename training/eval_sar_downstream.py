"""WHU-SAR-VV downstream eval: GT optical vs raw SAR vs fake optical.

Generator checkpoint: whu_sar_vv_ablation_08_bf16_b4_msssim_200/best.pth.
Test data: WHU-SEN-City-Paper-VV (Wuhan, Sentinel-1 VV SAR + Sentinel-2 RGB,
~10 m/px), 50/50 train/test split via scene_regex '^(train|test)'.

Domain-appropriate benchmarks (no finetuning):
  1. EuroSAT 10-class land cover classifier
       mrm8488/convnext-tiny-finetuned-eurosat
       classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
                Pasture, PermanentCrop, Residential, River, SeaLake
       Per-image top-1 — record (raw_sar==gt_optical), (fake_optical==gt_optical).

  2. RemoteCLIP-ViT-B/32 image embedding cosine vs GT optical
       chendelong/RemoteCLIP — CLIP architecture pretrained on aerial/satellite.
       Drop-in replacement for vanilla CLIP cos that knows aerial.

  3. Restor TCD tree-cover binary semantic seg (>=5% gate)
       restor/tcd-segformer-mit-b5 — genuine aerial-trained semantic seg.
       Per-image Dice for the `tree` class against the GT-optical-predicted mask.

Outputs: experiments/eval_sar_downstream/{per_image,summary}.csv + console table.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.data.dataset import PairedNIRRGBDataset, discover_pairs, split_pairs
from src.models import build_generator


SAR_CONFIG = PROJECT_ROOT / "configs/whu_sar_vv_ablation_08_bf16_b4_msssim_200.yaml"
SAR_CKPT   = PROJECT_ROOT / "experiments/whu_sar_vv_ablation_08_bf16_b4_msssim_200/checkpoints/best.pth"
OUT_DIR    = PROJECT_ROOT / "experiments/eval_sar_downstream"

N_SAMPLES   = 200
BATCH_SIZE  = 4
NUM_WORKERS = 4
MIN_GT_FRAC = 0.05

EUROSAT_HF    = "mrm8488/convnext-tiny-finetuned-eurosat"
REMOTECLIP_HF = "chendelong/RemoteCLIP"
REMOTECLIP_ARCH = "ViT-B-32"
REMOTECLIP_FILE = "RemoteCLIP-ViT-B-32.pt"
TCD_HF        = "restor/tcd-segformer-mit-b5"
TCD_TREE_ID   = 1
TCD_IN_SIZE   = 512
TCD_OUT_HW    = (256, 256)

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


def eurosat_topk(model, x_neg11, device):
    """Return (B,) long tensor of top-1 class indices."""
    with torch.no_grad():
        out = model(pixel_values=prep(x_neg11, 224, IMAGENET_MEAN, IMAGENET_STD, device))
    return out.logits.argmax(dim=1)


def remoteclip_embed(model, x_neg11, device):
    with torch.no_grad():
        return model.encode_image(prep(x_neg11, 224, CLIP_MEAN, CLIP_STD, device))


def tcd_mask(model, x_neg11, device):
    with torch.no_grad():
        out = model(pixel_values=prep(x_neg11, TCD_IN_SIZE, IMAGENET_MEAN, IMAGENET_STD, device))
    logits = F.interpolate(out.logits, size=TCD_OUT_HW, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


def per_image_dice(pred, gt, class_id=TCD_TREE_ID):
    out = []
    B = gt.shape[0]
    for b in range(B):
        g = (gt[b] == class_id)
        if g.float().mean().item() < MIN_GT_FRAC:
            out.append(None); continue
        p = (pred[b] == class_id)
        inter = (p & g).sum().float()
        denom = p.sum().float() + g.sum().float()
        out.append(0.0 if denom.item() == 0 else float(2.0 * inter / denom))
    return out


def build_test_loader(cfg_obj):
    pairs = discover_pairs(cfg_obj.data.nir_dir, cfg_obj.data.rgb_dir, cfg_obj.data.extensions)
    _, test, _ = split_pairs(
        pairs,
        cfg_obj.split.train_ratio, cfg_obj.split.val_ratio, cfg_obj.split.test_ratio,
        seed=cfg_obj.split.seed, scene_regex=getattr(cfg_obj.split, "scene_regex", None),
    )
    test = test[:N_SAMPLES]
    ds = PairedNIRRGBDataset(test, image_size=cfg_obj.data.image_size,
                              train=False, aug_cfg=None, mp_cache=None)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def load_generator(cfg_obj, ckpt_path, device):
    G = build_generator(cfg_obj, cfg_obj.data.in_channels, cfg_obj.data.out_channels).to(device).eval()
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "G_ema" in state and state["G_ema"] is not None:
        G.load_state_dict(state["G_ema"]); ema = True
    else:
        G.load_state_dict(state["G"]); ema = False
    for p in G.parameters():
        p.requires_grad_(False)
    return G, int(state.get("epoch", -1)), ema


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    cfg_obj, _ = load_config(str(SAR_CONFIG))
    loader = build_test_loader(cfg_obj)
    n_test = len(loader.dataset)
    print(f"WHU-SAR-VV test: first {n_test} pairs (split.seed={cfg_obj.split.seed}, "
          f"scene_regex={cfg_obj.split.scene_regex!r}, val_ratio takes 'test' scenes)",
          flush=True)

    G, epoch, ema = load_generator(cfg_obj, SAR_CKPT, device)
    print(f"generator: whu_sar_vv_ablation_08_bf16_b4_msssim_200 best.pth  "
          f"(epoch={epoch}, ema={ema})", flush=True)

    print("loading downstream models ...", flush=True)
    from transformers import AutoModelForImageClassification, SegformerForSemanticSegmentation
    import open_clip
    from huggingface_hub import hf_hub_download

    print(f"  EuroSAT: {EUROSAT_HF}", flush=True)
    es = AutoModelForImageClassification.from_pretrained(EUROSAT_HF).eval().to(device)
    es_labels = [es.config.id2label[i] for i in range(len(es.config.id2label))]
    for p in es.parameters(): p.requires_grad_(False)

    print(f"  RemoteCLIP: {REMOTECLIP_HF} ({REMOTECLIP_ARCH})", flush=True)
    rc, _, _ = open_clip.create_model_and_transforms(REMOTECLIP_ARCH)
    ckpt_path = hf_hub_download(REMOTECLIP_HF, REMOTECLIP_FILE)
    rc.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    rc = rc.eval().to(device)
    for p in rc.parameters(): p.requires_grad_(False)

    print(f"  Restor TCD: {TCD_HF}", flush=True)
    tcd = SegformerForSemanticSegmentation.from_pretrained(TCD_HF).eval().to(device)
    for p in tcd.parameters(): p.requires_grad_(False)
    print("all downstream models on device.", flush=True)

    inputs = ("gt_optical", "raw_sar", "fake_optical")

    # Per-image storage.
    rgb_paths_all = []
    es_pred = {inp: [] for inp in inputs}         # int class per image
    rc_cos  = {inp: [] for inp in inputs}         # cosine vs GT optical per image (gt_optical is 1.0)
    tcd_dc  = {inp: [] for inp in inputs}         # Dice vs GT-optical TCD mask per image (gt_optical is 1.0)

    idx = 0
    for i, batch in enumerate(loader):
        sar  = batch["nir"].to(device, non_blocking=True)        # SAR (3-ch)
        opt  = batch["rgb"].to(device, non_blocking=True)        # GT optical
        with torch.no_grad():
            fake = G(sar)
        streams = {"gt_optical": opt, "raw_sar": sar, "fake_optical": fake}

        # EuroSAT top-1
        es_preds = {inp: eurosat_topk(es, streams[inp], device).cpu() for inp in inputs}
        for inp in inputs:
            es_pred[inp].extend(es_preds[inp].tolist())

        # RemoteCLIP cosine vs GT
        rc_embs = {inp: remoteclip_embed(rc, streams[inp], device) for inp in inputs}
        gt_emb = rc_embs["gt_optical"]
        for inp in inputs:
            cos = F.cosine_similarity(rc_embs[inp], gt_emb, dim=-1).cpu().tolist()
            rc_cos[inp].extend(cos)

        # TCD tree Dice vs GT-optical TCD mask
        gt_mask = tcd_mask(tcd, opt, device)
        for inp in inputs:
            m = tcd_mask(tcd, streams[inp], device)
            tcd_dc[inp].extend(per_image_dice(m, gt_mask, TCD_TREE_ID))

        rgb_paths_all.extend(batch["rgb_path"])
        idx += sar.size(0)
        done = min(idx, n_test)
        if (i + 1) % 5 == 0 or done == n_test:
            print(f"  {done}/{n_test}", flush=True)

    # --- per-image CSV -----
    pi_path = OUT_DIR / "per_image.csv"
    fields = ["rgb_path"]
    for inp in inputs:
        fields += [f"eurosat_class__{inp}",
                   f"remoteclip_cos__{inp}",
                   f"tcd_tree_dice__{inp}"]
    with open(pi_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for j, p in enumerate(rgb_paths_all):
            row = {"rgb_path": str(p)}
            for inp in inputs:
                row[f"eurosat_class__{inp}"]  = es_labels[es_pred[inp][j]]
                row[f"remoteclip_cos__{inp}"] = rc_cos[inp][j]
                d = tcd_dc[inp][j]
                row[f"tcd_tree_dice__{inp}"]  = "" if d is None else d
            w.writerow(row)

    # --- summary aggregates -----
    def mean_n(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs)/len(xs), len(xs)) if xs else (float("nan"), 0)

    n = len(rgb_paths_all)
    # EuroSAT: top-1 agreement w/ gt_optical's predicted class
    gt_es = es_pred["gt_optical"]
    es_acc = {inp: sum(int(a==b) for a,b in zip(es_pred[inp], gt_es))/n for inp in inputs}
    # EuroSAT: class distribution per stream (for KL/qualitative)
    from collections import Counter
    es_dist = {inp: Counter(es_labels[c] for c in es_pred[inp]) for inp in inputs}

    rc_mean = {inp: sum(rc_cos[inp])/n for inp in inputs}
    tcd_mean = {inp: mean_n(tcd_dc[inp]) for inp in inputs}

    su_path = OUT_DIR / "summary.csv"
    with open(su_path, "w", newline="") as f:
        keys = ["metric", "gt_optical", "raw_sar", "fake_optical", "gain"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        def gain(gt, raw, fake):
            return (fake - raw) / (gt - raw) if (gt - raw) > 1e-6 else float("nan")
        w.writerow({"metric": "eurosat_top1_agreement",
                    "gt_optical": es_acc["gt_optical"],
                    "raw_sar": es_acc["raw_sar"],
                    "fake_optical": es_acc["fake_optical"],
                    "gain": gain(es_acc["gt_optical"], es_acc["raw_sar"], es_acc["fake_optical"])})
        w.writerow({"metric": "remoteclip_cos_vs_gt",
                    "gt_optical": rc_mean["gt_optical"],
                    "raw_sar": rc_mean["raw_sar"],
                    "fake_optical": rc_mean["fake_optical"],
                    "gain": gain(rc_mean["gt_optical"], rc_mean["raw_sar"], rc_mean["fake_optical"])})
        w.writerow({"metric": f"tcd_tree_dice (n={tcd_mean['gt_optical'][1]} of {n} after 5% gate)",
                    "gt_optical": tcd_mean["gt_optical"][0],
                    "raw_sar": tcd_mean["raw_sar"][0],
                    "fake_optical": tcd_mean["fake_optical"][0],
                    "gain": gain(tcd_mean["gt_optical"][0], tcd_mean["raw_sar"][0], tcd_mean["fake_optical"][0])})

    # --- pretty print -----
    print("\n=== WHU-SAR-VV downstream  (first 200 test pairs) ===")
    print(f"{'metric':<32} {'GT opt':>8} {'raw SAR':>9} {'fake opt':>9}  gain")
    print("-" * 76)
    g1 = (es_acc['fake_optical']-es_acc['raw_sar']) / max(1e-6, es_acc['gt_optical']-es_acc['raw_sar'])
    g2 = (rc_mean['fake_optical']-rc_mean['raw_sar']) / max(1e-6, rc_mean['gt_optical']-rc_mean['raw_sar'])
    tdg = tcd_mean['gt_optical'][0]; tdr = tcd_mean['raw_sar'][0]; tdf = tcd_mean['fake_optical'][0]
    g3 = (tdf-tdr) / max(1e-6, tdg-tdr) if (tdg-tdr)>0 else float('nan')
    print(f"{'EuroSAT top-1 vs GT class':<32} {es_acc['gt_optical']:>8.3f} "
          f"{es_acc['raw_sar']:>9.3f} {es_acc['fake_optical']:>9.3f}  {g1:>+.2%}")
    print(f"{'RemoteCLIP cosine vs GT opt':<32} {rc_mean['gt_optical']:>8.4f} "
          f"{rc_mean['raw_sar']:>9.4f} {rc_mean['fake_optical']:>9.4f}  {g2:>+.2%}")
    print(f"{'TCD tree Dice (>=5% gate, n='+str(tcd_mean['gt_optical'][1])+')':<32} "
          f"{tdg:>8.4f} {tdr:>9.4f} {tdf:>9.4f}  {g3:>+.2%}")

    print("\nEuroSAT class distribution (top-1 counts):")
    print(f"  {'class':<22} {'GT':>6} {'raw':>6} {'fake':>6}")
    all_classes = sorted(set().union(*[d.keys() for d in es_dist.values()]))
    for c in all_classes:
        print(f"  {c:<22} {es_dist['gt_optical'][c]:>6} "
              f"{es_dist['raw_sar'][c]:>6} {es_dist['fake_optical'][c]:>6}")

    print(f"\nwrote {pi_path}\nwrote {su_path}", flush=True)


if __name__ == "__main__":
    main()
