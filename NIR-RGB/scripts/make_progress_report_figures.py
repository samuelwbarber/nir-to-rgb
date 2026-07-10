#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_downstream import _load_uint8, _translate
from src.data.dataset import discover_pairs as repo_discover_pairs
from src.data.dataset import load_mp_cache, split_pairs
from src.models import build_generator
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config

EXP = Path("/home/sbarber9876/experiments")
OUT = ROOT / "docs" / "figures"


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit_crop(img: np.ndarray, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]


def resize_square(img: np.ndarray, size=420) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def discover_pairs():
    nir_dir = ROOT / "data" / "nir"
    rgb_dir = ROOT / "data" / "rgb"
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    nir = {p.stem: p for p in nir_dir.iterdir() if p.suffix.lower() in exts}
    pairs = []
    for p in rgb_dir.iterdir():
        if p.suffix.lower() in exts and p.stem in nir:
            pairs.append((p.stem, nir[p.stem], p))
    return sorted(pairs, key=lambda x: x[0])


def load_cache(name="mp_cache.json"):
    with open(ROOT / "data" / name) as f:
        data = json.load(f)
    return data.get("items", data)


def savefig(path: Path, fig):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def half_half(nir: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    mid = out.shape[1] // 2
    out[:, :mid] = nir[:, :mid]
    cv2.line(out, (mid, 0), (mid, out.shape[0]), (255, 255, 255), 4)
    cv2.line(out, (mid, 0), (mid, out.shape[0]), (20, 20, 20), 1)
    return out


def make_dataset_pairs():
    pairs = discover_pairs()
    chosen = []
    preferred = ["20260505_122104_654322_a473e239", "20260505_135001_493941_cf8c21ee", "20260504_200224_1d2f5c44"]
    by_stem = {s: (s, n, r) for s, n, r in pairs}
    for stem in preferred:
        if stem in by_stem:
            chosen.append(by_stem[stem])
    if len(chosen) < 3:
        chosen.extend([pairs[len(pairs) // 5], pairs[len(pairs) // 2], pairs[4 * len(pairs) // 5]])
    chosen = chosen[:3]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    for row, (stem, nir_p, rgb_p) in enumerate(chosen):
        nir = read_rgb(nir_p)
        rgb = read_rgb(rgb_p)
        h, w = rgb.shape[:2]
        side = int(min(h, w) * 0.78)
        x = (w - side) // 2
        y = (h - side) // 2
        nir_c = resize_square(nir[y:y + side, x:x + side])
        rgb_c = resize_square(rgb[y:y + side, x:x + side])
        composite = half_half(nir_c, rgb_c)
        axes[row].imshow(composite)
        axes[row].set_xticks([])
        axes[row].set_yticks([])
        axes[row].set_title(stem[:22] + "...", fontsize=9)
        axes[row].text(0.02, 0.94, "NIR", transform=axes[row].transAxes, color="white", fontsize=11, weight="bold",
                       bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
        axes[row].text(0.84, 0.94, "RGB", transform=axes[row].transAxes, color="white", fontsize=11, weight="bold",
                       bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
    fig.suptitle("Pixel-paired training examples after blur filtering + SIFT/RANSAC alignment", fontsize=14, weight="bold")
    fig.text(0.5, 0.01, "Each tile is one aligned pair: left half NIR, right half RGB, same pixel grid.", ha="center", fontsize=10)
    savefig(OUT / "dataset_pixel_pairs.png", fig)


def make_baseline_arch():
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.axis("off")
    def box(x, y, w, h, txt, color, fs=10):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="#111827", lw=1.3, transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, weight="bold", transform=ax.transAxes)
    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=2.0, color="#111827"))

    box(0.04, 0.55, 0.14, 0.20, "NIR input\n3 x 256 x 256", "#dbeafe")
    ax.add_patch(plt.Rectangle((0.27, 0.42), 0.46, 0.42, fc="#f8fafc", ec="#111827", lw=1.8, transform=ax.transAxes))
    ax.text(0.50, 0.79, "NAFNet-64 teacher", ha="center", va="center", fontsize=13, weight="bold", transform=ax.transAxes)
    box(0.30, 0.57, 0.09, 0.13, "intro\nconv", "#e0f2fe", fs=9)
    box(0.42, 0.57, 0.10, 0.13, "encoder\n2,2,4,8", "#dcfce7", fs=9)
    box(0.55, 0.57, 0.08, 0.13, "middle\n12", "#bbf7d0", fs=9)
    box(0.42, 0.44, 0.10, 0.10, "decoder\n2,2,2,2", "#dcfce7", fs=8)
    box(0.55, 0.44, 0.08, 0.10, "ending\n+tanh", "#e0f2fe", fs=8)
    box(0.82, 0.55, 0.14, 0.20, "predicted\nRGB", "#dbeafe")
    for a, b in [
        ((0.18, 0.65), (0.27, 0.65)),
        ((0.39, 0.635), (0.42, 0.635)),
        ((0.52, 0.635), (0.55, 0.635)),
        ((0.59, 0.57), (0.48, 0.54)),
        ((0.52, 0.49), (0.55, 0.49)),
        ((0.73, 0.65), (0.82, 0.65)),
    ]:
        arrow(a, b)
    ax.text(0.04, 0.26, "Teacher: NAFNet width 64, 116.0M parameters", fontsize=11, weight="bold", transform=ax.transAxes)
    ax.text(0.04, 0.17, "Training objective compares predicted RGB with ground-truth RGB:", fontsize=10, weight="bold", transform=ax.transAxes)
    ax.text(0.04, 0.10, "100 L1 + 20 VGG perceptual + 3 MS-SSIM + 0.5 GAN (PatchGAN)", fontsize=10, transform=ax.transAxes)
    ax.set_title("Teacher architecture", fontsize=16, weight="bold")
    savefig(OUT / "baseline_architecture.png", fig)


def gap_data(exp_name):
    with open(EXP / exp_name / "downstream_test.json") as f:
        data = json.load(f)
    out = []
    for task, block in data["summary"].items():
        gap = block.get("gap_closure")
        if gap is None:
            continue
        out.append((task, gap * 100.0))
    return out


def _primary_value(block, condition):
    primary = block["primary_metric"]
    if condition == "nir":
        return block["nir_vs_rgb"].get(primary, block["nir_per_condition"].get(primary, np.nan))
    return block["translated_vs_rgb"].get(primary, block["translated_per_condition"].get(primary, np.nan))


def make_downstream_chart():
    tasks = ["deeplab", "resnet50", "midas", "yolo", "maskrcnn", "mp_hands", "mp_pose", "mp_selfie"]
    labels = {
        "deeplab": "DeepLab",
        "resnet50": "ResNet50",
        "midas": "MiDaS",
        "yolo": "YOLO",
        "maskrcnn": "Mask R-CNN",
        "mp_hands": "MP hands",
        "mp_pose": "MP pose",
        "mp_selfie": "MP selfie",
    }
    with open(EXP / "ablation_01_baseline" / "downstream_test.json") as f:
        summary = json.load(f)["summary"]
    raw = [_primary_value(summary[t], "nir") for t in tasks]
    trans = [_primary_value(summary[t], "translated") for t in tasks]
    y = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.barh(y + 0.18, raw, height=0.32, label="raw NIR", color="#64748b")
    ax.barh(y - 0.18, trans, height=0.32, label="01 translated RGB", color="#2563eb")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[t] for t in tasks])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Primary downstream score against RGB reference (higher is better)")
    ax.set_title("01 teacher: raw NIR vs translated RGB on downstream models (200 test images)", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right")
    savefig(OUT / "downstream_gap_closure.png", fig)


def psnr(pred, target):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    mse = np.mean((pred - target) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def make_baseline_test_examples():
    cfg, _ = load_config(str(EXP / "ablation_01_baseline" / "config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = repo_discover_pairs(cfg.data.nir_dir, cfg.data.rgb_dir, cfg.data.extensions)
    _, _, test_pairs = split_pairs(
        pairs,
        cfg.split.train_ratio,
        cfg.split.val_ratio,
        cfg.split.test_ratio,
        seed=cfg.split.seed,
        scene_regex=cfg.split.scene_regex,
    )
    cache = load_mp_cache(ROOT / "data" / "mp_cache.json")
    rows = []
    for nir_p, rgb_p in test_pairs:
        entry = cache.get(rgb_p.stem) or cache.get(nir_p.stem)
        area = 1.0
        if entry and entry.get("crop_bbox"):
            img = read_rgb(nir_p)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = entry["crop_bbox"]
            area = max(0, x2 - x1) * max(0, y2 - y1) / float(w * h)
        rows.append((abs(area - 0.13), area, nir_p, rgb_p))
    rows.sort()
    selected = [(n, r) for _, _, n, r in rows[:4]]

    generator = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels).to(device).eval()
    load_checkpoint(EXP / "ablation_01_baseline" / "checkpoints" / "best.pth", generator, map_location=device)

    fig, axes = plt.subplots(len(selected), 3, figsize=(8.7, 10.2))
    psnrs = []
    for i, (nir_p, rgb_p) in enumerate(selected):
        nir = _load_uint8(nir_p, cfg.data.image_size, crop_bbox=None)
        rgb = _load_uint8(rgb_p, cfg.data.image_size, crop_bbox=None)
        pred = _translate(generator, nir, device)
        p = psnr(pred, rgb)
        psnrs.append(p)
        for ax, img, title in zip(
            axes[i],
            [nir, pred, rgb],
            ["raw NIR", f"01 translated\nPSNR {p:.1f} dB", "GT RGB"],
        ):
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=10, weight="bold")
            elif "PSNR" in title:
                ax.set_title(f"PSNR {p:.1f} dB", fontsize=9)
        axes[i, 0].set_ylabel(nir_p.stem[:18] + "...", fontsize=8)
    fig.suptitle(f"01 teacher test-set translations, mean PSNR {np.mean(psnrs):.1f} dB", fontsize=14, weight="bold")
    savefig(OUT / "baseline_test_examples.png", fig)


def best_box(cache, key, min_ratio=0.03, max_ratio=0.45):
    pairs = discover_pairs()
    rows = []
    for stem, nir_p, rgb_p in pairs:
        entry = cache.get(stem)
        if not entry or not entry.get(key):
            continue
        img = read_rgb(nir_p)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = entry[key]
        ratio = max(0, x2 - x1) * max(0, y2 - y1) / float(w * h)
        if min_ratio <= ratio <= max_ratio:
            rows.append((abs(ratio - 0.18), ratio, stem, nir_p, rgb_p, entry[key]))
    rows.sort()
    return rows[0]


def make_smartcrop():
    cache = load_cache("mp_cache.json")
    _, ratio, stem, nir_p, rgb_p, box = best_box(cache, "crop_bbox", min_ratio=0.04, max_ratio=0.35)
    nir = read_rgb(nir_p)
    rgb = read_rgb(rgb_p)
    x1, y1, x2, y2 = [int(v) for v in box]
    draw = nir.copy()
    cv2.rectangle(draw, (x1, y1), (x2, y2), (255, 80, 0), max(3, nir.shape[1] // 220))
    crop_n = fit_crop(nir, box)
    crop_r = fit_crop(rgb, box)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for ax, img, title in zip(
        axes,
        [draw, resize_square(crop_n), resize_square(crop_r)],
        ["full NIR + crop box", "NIR crop resized to 256", "paired RGB crop"],
    ):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, weight="bold")
    fig.suptitle(f"02 smart crop example: {stem} ({ratio*100:.1f}% of frame kept before resize)", fontsize=13, weight="bold")
    savefig(OUT / "smartcrop_example.png", fig)


def make_region_l1():
    cache = load_cache("mp_cache_no_crop.json")
    _, ratio, stem, nir_p, rgb_p, box = best_box(cache, "region_bbox", min_ratio=0.02, max_ratio=0.28)
    nir = read_rgb(nir_p)
    h, w = nir.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    overlay = nir.copy()
    mask = np.zeros((h, w), np.float32)
    mask[y1:y2, x1:x2] = 1.0
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay = np.where(mask[..., None] > 0, (0.58 * overlay + 0.42 * red).astype(np.uint8), overlay)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 30, 30), max(3, w // 220))
    weight = 1.0 + 4.0 * mask
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    axes[0].imshow(nir)
    axes[0].set_title("NIR frame", weight="bold", fontsize=10)
    axes[1].imshow(overlay)
    axes[1].set_title("MediaPipe region box", weight="bold", fontsize=10)
    im = axes[2].imshow(weight, cmap="inferno", vmin=1, vmax=5)
    axes[2].set_title("03 L1 weight map (1x -> 5x)", weight="bold", fontsize=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(f"03 region-weighted L1: {stem} ({ratio*100:.1f}% of pixels boosted)", fontsize=13, weight="bold")
    savefig(OUT / "region_l1_overlay.png", fig)


def make_nir_grad():
    pairs = discover_pairs()
    by_stem = {s: (s, n, r) for s, n, r in pairs}
    stem, nir_p, _ = by_stem.get("20260505_122104_654322_a473e239", pairs[len(pairs) // 2])
    nir = read_rgb(nir_p)
    small = cv2.resize(nir, (420, 420), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    avg = cv2.blur(gray, (7, 7))
    sq = cv2.blur(gray * gray, (7, 7))
    std = np.sqrt(np.maximum(sq - avg * avg, 0))
    norm = std / max(float(std.max()), 1e-6)
    weight = 0.2 + 0.8 * norm
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    axes[0].imshow(small)
    axes[0].set_title("NIR input", weight="bold", fontsize=10)
    axes[1].imshow(norm, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("local NIR variation", weight="bold", fontsize=10)
    im = axes[2].imshow(weight, cmap="viridis", vmin=0.2, vmax=1.0)
    axes[2].set_title("L1 weight map (0.2x -> 1x)", weight="bold", fontsize=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("NIR-gradient L1: reduce penalty where NIR contains little predictive signal", fontsize=13, weight="bold")
    savefig(OUT / "nir_gradient_l1.png", fig)


def make_featuremap():
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.axis("off")
    def box(x, y, w, h, txt, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="#111827", lw=1.2, transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10, weight="bold", transform=ax.transAxes)
    def arr(a, b):
        ax.annotate("", xy=b, xytext=a, xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.6, color="#111827"))
    box(0.04, 0.58, 0.14, 0.22, "NIR", "#dbeafe")
    box(0.24, 0.58, 0.18, 0.22, "generator", "#dcfce7")
    box(0.48, 0.72, 0.16, 0.18, "pred RGB", "#dbeafe")
    box(0.48, 0.38, 0.16, 0.18, "GT RGB", "#ffedd5")
    box(0.70, 0.70, 0.22, 0.20, "frozen ResNet50\nstem/layer1/layer2", "#f8fafc")
    box(0.70, 0.36, 0.22, 0.20, "frozen DeepLabV3\nlow/mid features", "#f8fafc")
    box(0.70, 0.08, 0.22, 0.16, "activation\nmatching loss", "#fee2e2")
    arr((0.18, 0.69), (0.24, 0.69))
    arr((0.42, 0.69), (0.48, 0.81))
    arr((0.56, 0.72), (0.70, 0.80))
    arr((0.56, 0.56), (0.70, 0.46))
    arr((0.56, 0.72), (0.70, 0.46))
    arr((0.56, 0.38), (0.70, 0.80))
    arr((0.81, 0.36), (0.81, 0.24))
    arr((0.81, 0.70), (0.81, 0.24))
    arr((0.70, 0.16), (0.34, 0.58))
    ax.text(0.04, 0.14, "L1 weight 25; feature-map weight ramps 0 -> 50 over first 5 epochs.\nGoal: optimize for RGB-trained model activations, not only pixel similarity.", fontsize=10, transform=ax.transAxes)
    ax.set_title("Feature-map loss: backprop through frozen downstream models", fontsize=14, weight="bold")
    savefig(OUT / "featuremap_loss.png", fig)


def make_distill():
    fig, ax = plt.subplots(figsize=(9.5, 2.7))
    ax.axis("off")
    vals = [116.0, 6.0]
    labels = ["teacher\nNAFNet-64", "student\nNAFNet-24"]
    colors = ["#2563eb", "#16a34a"]
    ax.barh([1, 0], vals, color=colors, height=0.46)
    for y, v, lab in zip([1, 0], vals, labels):
        ax.text(v + 2, y, f"{lab}: {v:.1f}M params", va="center", fontsize=11, weight="bold")
    ax.set_xlim(0, 130)
    ax.set_yticks([])
    ax.set_xlabel("parameters (M)")
    ax.set_title("Distillation target: keep teacher behaviour, deploy small student", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    savefig(OUT / "distillation_params.png", fig)


def latest_distill_metrics():
    tb_dir = ROOT / "experiments" / "phase4_student_nafnet24_distill" / "tb"
    latest = {}
    for ev in sorted(tb_dir.glob("events.*")):
        ea = EventAccumulator(str(ev))
        ea.Reload()
        for tag in ["val/psnr_gt", "val/psnr_vs_teacher", "val/ssim_gt", "val/ssim_vs_teacher", "val/lpips_gt"]:
            if tag not in ea.Tags().get("scalars", []):
                continue
            val = ea.Scalars(tag)[-1]
            if tag not in latest or val.step >= latest[tag][0]:
                latest[tag] = (val.step, float(val.value))
    return latest


def make_distill_progress():
    student_cfg, _ = load_config(str(ROOT / "experiments" / "phase4_student_nafnet24_distill" / "config.yaml"))
    teacher_cfg, _ = load_config(str(ROOT / "experiments" / "phase3_teacher_nafnet64_smartcrop_fp32" / "config.yaml"))
    device = "cpu"

    pairs = repo_discover_pairs(student_cfg.data.nir_dir, student_cfg.data.rgb_dir, student_cfg.data.extensions)
    _, _, test_pairs = split_pairs(
        pairs,
        student_cfg.split.train_ratio,
        student_cfg.split.val_ratio,
        student_cfg.split.test_ratio,
        seed=student_cfg.split.seed,
        scene_regex=student_cfg.split.scene_regex,
    )
    cache = load_mp_cache(ROOT / "data" / "mp_cache.json")
    ranked = []
    for nir_p, rgb_p in test_pairs:
        entry = cache.get(rgb_p.stem) or cache.get(nir_p.stem)
        area = 0.0
        if entry and entry.get("crop_bbox"):
            img = read_rgb(nir_p)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = entry["crop_bbox"]
            area = max(0, x2 - x1) * max(0, y2 - y1) / float(w * h)
        ranked.append((abs(area - 0.12), nir_p, rgb_p))
    ranked.sort()
    selected = [(n, r) for _, n, r in ranked[:3]]

    student = build_generator(student_cfg, student_cfg.data.in_channels, student_cfg.data.out_channels).to(device).eval()
    teacher = build_generator(teacher_cfg, teacher_cfg.data.in_channels, teacher_cfg.data.out_channels).to(device).eval()
    load_checkpoint(ROOT / "experiments" / "phase4_student_nafnet24_distill" / "checkpoints" / "latest.pth", student, map_location=device)
    load_checkpoint(ROOT / "experiments" / "phase3_teacher_nafnet64_smartcrop_fp32" / "checkpoints" / "latest.pth", teacher, map_location=device)

    fig, axes = plt.subplots(len(selected), 4, figsize=(10.2, 8.1))
    for out_i, (nir_p, rgb_p) in enumerate(selected):
        nir = _load_uint8(nir_p, student_cfg.data.image_size, crop_bbox=None)
        rgb = _load_uint8(rgb_p, student_cfg.data.image_size, crop_bbox=None)
        student_rgb = _translate(student, nir, device)
        teacher_rgb = _translate(teacher, nir, device)
        for ax, img, title in zip(
            axes[out_i],
            [nir, student_rgb, teacher_rgb, rgb],
            ["NIR", "student RGB", "teacher RGB", "GT RGB"],
        ):
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if out_i == 0:
                ax.set_title(title, fontsize=11, weight="bold")
        axes[out_i, 0].set_ylabel(nir_p.stem[:18] + "...", fontsize=8)
    m = latest_distill_metrics()
    epoch = int(m.get("val/psnr_gt", (53, 0))[0])
    psnr_gt = m["val/psnr_gt"][1]
    psnr_t = m["val/psnr_vs_teacher"][1]
    ssim_gt = m["val/ssim_gt"][1]
    ssim_t = m["val/ssim_vs_teacher"][1]
    fig.suptitle(
        f"Student distillation progress, epoch {epoch}: PSNR_GT {psnr_gt:.2f} dB, "
        f"PSNR_teacher {psnr_t:.2f} dB",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        0.02,
        f"Validation: SSIM_GT {ssim_gt:.3f}, SSIM_teacher {ssim_t:.3f}. "
        "Student is learning to match the teacher while staying anchored to GT RGB.",
        ha="center",
        fontsize=9,
    )
    savefig(OUT / "distillation_progress.png", fig)

    (OUT / "distillation_metrics.tex").write_text(
        f"Epoch & PSNR to GT & PSNR to teacher & SSIM to GT & SSIM to teacher \\\\\n"
        f"\\midrule\n"
        f"{epoch} & {psnr_gt:.2f} dB & {psnr_t:.2f} dB & {ssim_gt:.3f} & {ssim_t:.3f} \\\\\n"
    )


def write_metrics_tex():
    with open(EXP / "ablation_01_baseline" / "downstream_test.json") as f:
        d1 = json.load(f)
    with open(EXP / "ablation_02_smartcrop" / "downstream_test.json") as f:
        d2 = json.load(f)
    useful = ["deeplab", "resnet50", "midas", "yolo", "maskrcnn"]
    names = {"deeplab": "DeepLab", "resnet50": "ResNet50", "midas": "MiDaS", "yolo": "YOLO", "maskrcnn": "Mask R-CNN"}
    lines = []
    for task in useful:
        b1 = d1["summary"][task]
        b2 = d2["summary"][task]
        metric = b1["primary_metric"].replace("_", "\\_")
        raw = b1["nir_vs_rgb"].get(b1["primary_metric"], b1["nir_per_condition"].get(b1["primary_metric"], 0))
        tra = b1["translated_vs_rgb"].get(b1["primary_metric"], b1["translated_per_condition"].get(b1["primary_metric"], 0))
        gap1 = b1["gap_closure"] * 100
        gap2 = b2["gap_closure"] * 100
        lines.append(f"{names[task]} & {metric} & {raw:.3f} & {tra:.3f} & {gap1:+.1f}\\% & {gap2:+.1f}\\% \\\\")
    (OUT / "metrics_rows.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_dataset_pairs()
    make_baseline_arch()
    make_downstream_chart()
    make_baseline_test_examples()
    make_smartcrop()
    make_region_l1()
    make_nir_grad()
    make_featuremap()
    make_distill()
    make_distill_progress()
    write_metrics_tex()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
