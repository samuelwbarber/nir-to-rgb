"""Evaluate an ONNX model (fp32 or INT8) on a dataset split.

Usage:
    python scripts/eval_onnx.py --config configs/student_hailo_30fps.yaml \
        --onnx experiments/student_hailo_30fps/model_q.onnx --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.dataset import discover_pairs, split_pairs, PairedNIRRGBDataset
from src.eval.metrics import compute_psnr, compute_ssim, LPIPSWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--onnx", required=True, help="Path to .onnx model (fp32 or INT8)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(
        args.onnx,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    provider = sess.get_providers()[0]
    print(f"ONNX model : {args.onnx}")
    print(f"Provider   : {provider}")

    cfg, _ = load_config(args.config)
    pairs = discover_pairs(cfg.data.nir_dir, cfg.data.rgb_dir, cfg.data.extensions)
    train_p, val_p, test_p = split_pairs(
        pairs, cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio,
        seed=cfg.split.seed, scene_regex=cfg.split.scene_regex,
    )
    chosen = {"train": train_p, "val": val_p, "test": test_p}[args.split]
    ds = PairedNIRRGBDataset(chosen, image_size=cfg.data.image_size, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = LPIPSWrapper(net=cfg.eval.lpips_net).to(device)

    psnr_sum = ssim_sum = lpips_sum = 0.0
    n = 0
    for batch in tqdm(loader, desc=f"eval({args.split})"):
        nir_np = batch["nir"].numpy().astype(np.float32)
        rgb = batch["rgb"].to(device)

        fake_np = sess.run(None, {in_name: nir_np})[0]
        fake = torch.from_numpy(fake_np).to(device)

        bs = rgb.size(0)
        psnr_sum += compute_psnr(fake, rgb) * bs
        ssim_sum += compute_ssim(fake, rgb) * bs
        lpips_sum += lpips_fn(fake, rgb) * bs
        n += bs

    print(f"\n[{args.split}] n={n}  model={Path(args.onnx).name}")
    print(f"  PSNR  = {psnr_sum / n:.3f}")
    print(f"  SSIM  = {ssim_sum / n:.4f}")
    print(f"  LPIPS = {lpips_sum / n:.4f}")


if __name__ == "__main__":
    main()
