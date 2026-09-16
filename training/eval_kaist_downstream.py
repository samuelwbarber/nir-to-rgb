"""KAIST downstream Dice eval: GT RGB vs raw thermal vs fake RGB.

For the kaist_08_ganoff_200 best checkpoint, run translation on the first 200
KAIST test pairs (seed=42 split, same as training) and compare against raw
thermal across four pretrained semantic-seg models:

  - SegFormer-B0 ADE20K-512        (classes: tree, person, building, road)
  - SegFormer-B0 Cityscapes-512    (classes: road, building, vegetation, sky, person, car)
  - DeepLabV3-ResNet50 Pascal VOC  (classes: car, person, bicycle, bus, motorbike)
  - UPerNet-ConvNeXt-Small ADE20K  (classes: tree, person, building, road)

Per-image Dice for each class is recorded only when the GT-RGB mask for that
class covers >= MIN_GT_FRAC (5%) of pixels — same gate as eval_lineage.py.

Inputs evaluated per seg model:
  - gt_rgb      : sanity (Dice vs itself = 1.0)
  - raw_thermal : floor reference (3-channel thermal fed directly to seg model)
  - fake_rgb    : G(thermal) where G is kaist_08_ganoff_200/best.pth

Outputs:
  experiments/eval_kaist_downstream/per_image.csv
  experiments/eval_kaist_downstream/summary.csv
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


KAIST_CONFIG = PROJECT_ROOT / "configs/kaist_08_ganoff_200.yaml"
KAIST_CKPT   = PROJECT_ROOT / "experiments/kaist_08_ganoff_200/checkpoints/best.pth"
OUT_DIR      = PROJECT_ROOT / "experiments/eval_kaist_downstream"

N_SAMPLES   = 200
BATCH_SIZE  = 4
NUM_WORKERS = 4
MASK_HW     = (256, 256)
MIN_GT_FRAC = 0.05

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

# --- seg-model registry ------------------------------------------------------
# Each entry describes a frozen pretrained semantic segmentation model and the
# subset of its label space we want Dice for on KAIST automotive scenes.

SEG_MODELS = [
    {
        "name": "segf_ade",
        "kind": "segformer",
        "hf":   "nvidia/segformer-b0-finetuned-ade-512-512",
        "in_size": 512,
        "classes": {"tree": 4, "person": 12, "building": 1, "road": 6},
    },
    {
        "name": "segf_cs",
        "kind": "segformer",
        "hf":   "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
        "in_size": 512,
        "classes": {"road": 0, "building": 2, "vegetation": 8, "sky": 10,
                    "person": 11, "car": 13},
    },
    {
        "name": "deeplab_voc",
        "kind": "deeplab_tv",
        "hf":   None,
        "in_size": 520,
        "classes": {"car": 7, "person": 15, "bicycle": 2, "bus": 6, "motorbike": 14},
    },
    {
        "name": "upernet_ade",
        "kind": "upernet",
        "hf":   "openmmlab/upernet-convnext-small",
        "in_size": 512,
        "classes": {"tree": 4, "person": 12, "building": 1, "road": 6},
    },
]


def to_unit(x):
    return (x.clamp(-1, 1) + 1) * 0.5


def prep(x_neg11, size, device):
    x = to_unit(x_neg11)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    m = IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
    s = IMAGENET_STD.to(device).view(1, 3, 1, 1)
    return (x - m) / s


def seg_mask(model_entry, model, x_neg11, device, out_hw):
    """Return (B, H, W) long mask of argmax class ids at out_hw resolution."""
    kind = model_entry["kind"]
    in_size = model_entry["in_size"]
    pixel_values = prep(x_neg11, in_size, device)
    with torch.no_grad():
        if kind in ("segformer", "upernet"):
            out = model(pixel_values=pixel_values)
            logits = out.logits                                  # (B, C, H', W')
        elif kind == "deeplab_tv":
            out = model(pixel_values)                            # tv Dict[Tensor]
            logits = out["out"]
        else:
            raise ValueError(kind)
    logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


def per_image_dice(pred, gt, class_id):
    """Per-image Dice for one class. Returns list len B; entries float or None
    (None when GT covers <MIN_GT_FRAC for that image)."""
    out = []
    B, H, W = gt.shape
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


def load_seg_model(entry, device):
    kind = entry["kind"]
    if kind == "segformer":
        from transformers import SegformerForSemanticSegmentation
        m = SegformerForSemanticSegmentation.from_pretrained(entry["hf"])
    elif kind == "upernet":
        from transformers import UperNetForSemanticSegmentation
        m = UperNetForSemanticSegmentation.from_pretrained(entry["hf"])
    elif kind == "deeplab_tv":
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
        m = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(kind)
    m = m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    cfg_obj, _ = load_config(str(KAIST_CONFIG))
    loader = build_test_loader(cfg_obj)
    n_test = len(loader.dataset)
    print(f"KAIST test: first {n_test} pairs of seed={cfg_obj.split.seed} split", flush=True)

    G, epoch, ema = load_generator(cfg_obj, KAIST_CKPT, device)
    print(f"generator: kaist_08_ganoff_200 best.pth  (epoch={epoch}, ema={ema})", flush=True)

    seg_loaded = {}
    print("loading seg models ...", flush=True)
    for entry in SEG_MODELS:
        print(f"  {entry['name']} ({entry['hf'] or 'torchvision'})", flush=True)
        seg_loaded[entry["name"]] = load_seg_model(entry, device)
    print("all seg models on device.", flush=True)

    inputs = ("gt_rgb", "raw_thermal", "fake_rgb")
    # dice[(seg_name, cls_name, input_name)] -> list of per-image dice values
    dice = {}
    for s in SEG_MODELS:
        for cls in s["classes"]:
            for inp in inputs:
                dice[(s["name"], cls, inp)] = []
    rgb_paths_all = []

    idx = 0
    for i, batch in enumerate(loader):
        thermal = batch["nir"].to(device, non_blocking=True)   # [-1,1], 3ch
        rgb     = batch["rgb"].to(device, non_blocking=True)   # [-1,1]
        with torch.no_grad():
            fake = G(thermal)

        streams = {"gt_rgb": rgb, "raw_thermal": thermal, "fake_rgb": fake}

        for s in SEG_MODELS:
            sm = seg_loaded[s["name"]]
            mask_gt = seg_mask(s, sm, rgb, device, MASK_HW)
            mask_th = seg_mask(s, sm, thermal, device, MASK_HW)
            mask_fk = seg_mask(s, sm, fake, device, MASK_HW)
            masks = {"gt_rgb": mask_gt, "raw_thermal": mask_th, "fake_rgb": mask_fk}
            for cls_name, cls_id in s["classes"].items():
                for inp in inputs:
                    dice[(s["name"], cls_name, inp)].extend(
                        per_image_dice(masks[inp], mask_gt, cls_id))

        rgb_paths_all.extend(batch["rgb_path"])
        idx += thermal.size(0)
        done = min(idx, n_test)
        if (i + 1) % 5 == 0 or done == n_test:
            print(f"  {done}/{n_test}", flush=True)

    # --- per-image CSV ------------------------------------------------------
    pi_path = OUT_DIR / "per_image.csv"
    fieldnames = ["rgb_path"]
    for s in SEG_MODELS:
        for cls in s["classes"]:
            for inp in inputs:
                fieldnames.append(f"{s['name']}__{cls}__{inp}")

    with open(pi_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for j, path in enumerate(rgb_paths_all):
            row = {"rgb_path": str(path)}
            for s in SEG_MODELS:
                for cls in s["classes"]:
                    for inp in inputs:
                        v = dice[(s["name"], cls, inp)][j]
                        row[f"{s['name']}__{cls}__{inp}"] = "" if v is None else v
            w.writerow(row)

    # --- summary CSV --------------------------------------------------------
    def mean_n(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)

    su_path = OUT_DIR / "summary.csv"
    su_rows = []
    for s in SEG_MODELS:
        for cls in s["classes"]:
            row = {"seg_model": s["name"], "class": cls}
            n_gate = None
            for inp in inputs:
                m, n = mean_n(dice[(s["name"], cls, inp)])
                row[f"dice_{inp}"]  = m
                row[f"n_{inp}"]     = n
                if inp == "gt_rgb":
                    n_gate = n
            su_rows.append(row)

    with open(su_path, "w", newline="") as f:
        keys = ["seg_model", "class",
                "dice_gt_rgb", "n_gt_rgb",
                "dice_raw_thermal", "n_raw_thermal",
                "dice_fake_rgb", "n_fake_rgb"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(su_rows)

    # --- pretty print -------------------------------------------------------
    print("\n=== KAIST downstream Dice (first 200 test pairs, MIN_GT_FRAC=0.05) ===")
    print(f"{'seg_model':<14} {'class':<12} {'n':>4}  "
          f"{'GT RGB':>8} {'raw_thermal':>12} {'fake_rgb':>10}  gain")
    print("-" * 78)
    for s in SEG_MODELS:
        for cls in s["classes"]:
            gt_d, gt_n = mean_n(dice[(s["name"], cls, "gt_rgb")])
            th_d, _    = mean_n(dice[(s["name"], cls, "raw_thermal")])
            fk_d, _    = mean_n(dice[(s["name"], cls, "fake_rgb")])
            gain = (fk_d - th_d) / (gt_d - th_d) if (gt_d - th_d) > 1e-6 else float("nan")
            print(f"{s['name']:<14} {cls:<12} {gt_n:>4}  "
                  f"{gt_d:>8.4f} {th_d:>12.4f} {fk_d:>10.4f}  {gain:>+.2%}")

    print(f"\nwrote {pi_path}\nwrote {su_path}", flush=True)


if __name__ == "__main__":
    main()
