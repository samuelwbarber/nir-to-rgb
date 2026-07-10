#!/usr/bin/env python3
"""Multi-model distillation comparison patch grid.

Columns = test images from manifest (default: 5 images used for best_random_5_grid).
Rows (top to bottom, all labelled):
  NIR input
  GT RGB
  Teacher  NAFNet64       (ablation_08_foundation_ensemble_aggressive, best.pth)
  NAFNet32 FP32           (student_hailo_30fps, model.onnx)
  NAFNet32 INT8 quantized (student_hailo_30fps, model_q.onnx)
  NAFNet16 FP32           (student_08_distill_downstream, model.onnx)
  NAFNet16 INT8 quantized (student_08_distill_downstream, model_q.onnx)

Usage:
    python scripts/render_distill_comparison_grid.py \\
        [--manifest experiments/ablation_08_foundation_ensemble_aggressive/best_random_5_grid.manifest.tsv] \\
        [--out experiments/distill_comparison_grid.png] \\
        [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch

from scripts.eval_downstream import _load_uint8, _translate
from src.models import build_generator
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config


ROW_LABELS = [
    "NIR input",
    "GT RGB",
    "Teacher NAFNet64\n(ablation_08)",
    "NAFNet32 FP32\n(Hailo-8L student)",
    "NAFNet32 INT8\n(Hailo-8L quant)",
    "NAFNet16 FP32\n(RPi5 student)",
    "NAFNet16 INT8\n(RPi5 quant)",
]

LABEL_W = 170
PAD = 4


def _draw_label(h: int, w: int, text: str) -> np.ndarray:
    cell = np.zeros((h, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    lines = text.split("\n")
    line_h = cv2.getTextSize("A", font, scale, thick)[0][1] + 8
    total = len(lines) * line_h
    y = (h - total) // 2 + line_h - 2
    for line in lines:
        cv2.putText(cell, line.strip(), (6, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h
    return cell


def _onnx_translate(sess, inp_name: str, nir_rgb: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB -> model -> uint8 HWC RGB via ONNX."""
    x = nir_rgb.astype(np.float32) / 127.5 - 1.0
    x = x.transpose(2, 0, 1)[np.newaxis]
    out = sess.run(None, {inp_name: x})[0]
    out = (np.clip(out[0], -1.0, 1.0) + 1.0) * 127.5
    return out.transpose(1, 2, 0).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default="experiments/ablation_08_foundation_ensemble_aggressive/best_random_5_grid.manifest.tsv",
    )
    p.add_argument("--out", default="experiments/distill_comparison_grid.png")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---- manifest ----
    manifest_path = ROOT / args.manifest
    pairs: list[tuple[Path, Path]] = []
    with open(manifest_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pairs.append((ROOT / parts[1], ROOT / parts[2]))
    print(f"loaded {len(pairs)} images from {manifest_path.name}")

    # ---- teacher (PyTorch) ----
    t_cfg, _ = load_config(str(ROOT / "experiments" / "ablation_08_foundation_ensemble_aggressive" / "config.yaml"))
    teacher = build_generator(t_cfg, t_cfg.data.in_channels, t_cfg.data.out_channels).to(device).eval()
    load_checkpoint(
        str(ROOT / "experiments" / "ablation_08_foundation_ensemble_aggressive" / "checkpoints" / "best.pth"),
        teacher, map_location=device,
    )
    print("teacher loaded (NAFNet64)")

    # ---- ONNX sessions ----
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _sess(path):
        s = ort.InferenceSession(str(ROOT / path), providers=providers)
        return s, s.get_inputs()[0].name

    s32_fp32_sess, s32_fp32_in = _sess("experiments/student_hailo_30fps/model.onnx")
    s32_int8_sess, s32_int8_in = _sess("experiments/student_hailo_30fps/model_q.onnx")
    s16_fp32_sess, s16_fp32_in = _sess("experiments/student_08_distill_downstream/model.onnx")
    s16_int8_sess, s16_int8_in = _sess("experiments/student_08_distill_downstream/model_q.onnx")
    print("ONNX sessions loaded (NAFNet32 fp32/int8, NAFNet16 fp32/int8)")

    # ---- run inference ----
    size = t_cfg.data.image_size
    n_cols = len(pairs)
    n_rows = 7

    cells = [[None] * n_cols for _ in range(n_rows)]

    for ci, (nir_p, rgb_p) in enumerate(pairs):
        print(f"  [{ci+1}/{n_cols}] {nir_p.name}")
        nir = _load_uint8(nir_p, size)
        rgb = _load_uint8(rgb_p, size)
        cells[0][ci] = nir
        cells[1][ci] = rgb
        cells[2][ci] = _translate(teacher, nir, device)
        cells[3][ci] = _onnx_translate(s32_fp32_sess, s32_fp32_in, nir)
        cells[4][ci] = _onnx_translate(s32_int8_sess, s32_int8_in, nir)
        cells[5][ci] = _onnx_translate(s16_fp32_sess, s16_fp32_in, nir)
        cells[6][ci] = _onnx_translate(s16_int8_sess, s16_int8_in, nir)

    # ---- assemble ----
    th, tw = size, size
    total_w = LABEL_W + n_cols * tw + (n_cols - 1) * PAD
    total_h = n_rows * th + (n_rows - 1) * PAD
    canvas = np.full((total_h, total_w, 3), 30, dtype=np.uint8)

    for ri in range(n_rows):
        y0 = ri * (th + PAD)
        canvas[y0:y0 + th, 0:LABEL_W] = _draw_label(th, LABEL_W, ROW_LABELS[ri])
        for ci in range(n_cols):
            x0 = LABEL_W + ci * (tw + PAD)
            canvas[y0:y0 + th, x0:x0 + tw] = cells[ri][ci]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"saved {out_path}  ({canvas.shape[1]}x{canvas.shape[0]} px)")


if __name__ == "__main__":
    main()
