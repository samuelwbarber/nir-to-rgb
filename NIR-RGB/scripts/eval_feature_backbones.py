"""Compare raw NIR and translated images with RGB-pretrained feature backbones.

This script treats the feature embedding from the real RGB image as the
reference and reports cosine similarity for:
  - raw NIR
  - ablation 01 translated image
  - ablation 02 translated image

It is intentionally separate from eval_downstream.py because these models are
feature encoders, not task heads with boxes/masks/keypoints.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.dataset import discover_pairs, split_pairs, _read_image
from src.models import build_generator
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config


MODEL_SPECS = {
    "clip_timm": {
        "timm_name": "vit_base_patch32_clip_224",
        "img_size": 224,
    },
    "dinov2_timm": {
        "timm_name": "vit_small_patch14_dinov2",
        "img_size": 518,
    },
    "sam_timm": {
        "timm_name": "samvit_base_patch16",
        "img_size": 1024,
    },
}


def load_uint8(path, size):
    img = _read_image(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def translate(generator, nir_uint8, device):
    x = torch.from_numpy(nir_uint8.astype(np.float32) / 127.5 - 1.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        y = generator(x)
    y = (y.clamp(-1, 1)[0] + 1) * 127.5
    return y.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def make_transform():
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def transform(img_uint8):
        x = torch.from_numpy(img_uint8.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0)
        return (x - mean) / std

    return transform


def embed(model, transform, img_uint8, device):
    x = transform(img_uint8).to(device)
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            y = model.forward_features(x)
        else:
            y = model(x)
        if isinstance(y, dict):
            y = y.get("x_norm_clstoken", next(iter(y.values())))
        if y.ndim == 4:
            y = y.mean(dim=(2, 3))
        elif y.ndim == 3:
            y = y[:, 0]
        y = y.flatten(1)
        return F.normalize(y.float(), dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config01", required=True)
    parser.add_argument("--checkpoint01", required=True)
    parser.add_argument("--config02", required=True)
    parser.add_argument("--checkpoint02", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    import timm

    cfg01, _ = load_config(args.config01)
    cfg02, _ = load_config(args.config02)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    pairs = discover_pairs(cfg01.data.nir_dir, cfg01.data.rgb_dir, cfg01.data.extensions)
    train, val, test = split_pairs(
        pairs,
        cfg01.split.train_ratio,
        cfg01.split.val_ratio,
        cfg01.split.test_ratio,
        seed=cfg01.split.seed,
        scene_regex=cfg01.split.scene_regex,
    )
    chosen = {"train": train, "val": val, "test": test}[args.split]
    if args.max_samples:
        chosen = chosen[: args.max_samples]

    gen01 = build_generator(cfg01, cfg01.data.in_channels, cfg01.data.out_channels).to(device).eval()
    gen02 = build_generator(cfg02, cfg02.data.in_channels, cfg02.data.out_channels).to(device).eval()
    load_checkpoint(args.checkpoint01, gen01, map_location=device)
    load_checkpoint(args.checkpoint02, gen02, map_location=device)

    transform = make_transform()
    results = {}

    print(f"device: {device}")
    print(f"evaluating {len(chosen)} samples")
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        print(f"\nloading {model_key}: {spec['timm_name']}")
        model = timm.create_model(spec["timm_name"], pretrained=True, num_classes=0).to(device).eval()

        sums = {"raw_rgb": 1.0, "raw_nir": 0.0, "conv01": 0.0, "conv02": 0.0}
        count = 0
        size = spec["img_size"]
        for nir_path, rgb_path in tqdm(chosen, desc=model_key):
            nir = load_uint8(nir_path, cfg01.data.image_size)
            rgb = load_uint8(rgb_path, cfg01.data.image_size)
            conv01 = translate(gen01, nir, device)
            conv02 = translate(gen02, nir, device)

            nir = cv2.resize(nir, (size, size), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            conv01 = cv2.resize(conv01, (size, size), interpolation=cv2.INTER_LINEAR)
            conv02 = cv2.resize(conv02, (size, size), interpolation=cv2.INTER_LINEAR)

            e_rgb = embed(model, transform, rgb, device)
            sums["raw_nir"] += float((embed(model, transform, nir, device) * e_rgb).sum().item())
            sums["conv01"] += float((embed(model, transform, conv01, device) * e_rgb).sum().item())
            sums["conv02"] += float((embed(model, transform, conv02, device) * e_rgb).sum().item())
            count += 1

        row = {k: (v if k == "raw_rgb" else v / count) for k, v in sums.items()}
        row["primary_metric"] = "embedding_cosine_vs_rgb"
        row["n_samples"] = count
        row["improvement01_vs_nir"] = row["conv01"] - row["raw_nir"]
        row["improvement02_vs_nir"] = row["conv02"] - row["raw_nir"]
        results[model_key] = row
        print(
            f"{model_key:12s} raw RGB {row['raw_rgb']:.4f}  raw NIR {row['raw_nir']:.4f}  "
            f"01 {row['conv01']:.4f}  02 {row['conv02']:.4f}"
        )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
