"""Generate a 5-column comparison grid: GT | Teacher | Student FP32 | Student INT8 | NIR.

Usage:
    python scripts/compare_teacher_student.py \
        --teacher-config  configs/ablation_08_foundation_ensemble_aggressive.yaml \
        --student-config  configs/student_hailo_30fps.yaml \
        --fp32-onnx       experiments/student_hailo_30fps/model.onnx \
        --int8-onnx       experiments/student_hailo_30fps/model_q.onnx \
        --n-images 5 --split test \
        --out             experiments/student_hailo_30fps/compare_grid.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.dataset import discover_pairs, split_pairs, PairedNIRRGBDataset
from src.models import build_generator
from src.training.checkpoint import load_checkpoint
from src.eval.metrics import compute_psnr, compute_ssim, LPIPSWrapper


def to_01(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(-1, 1) * 0.5 + 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-config", required=True)
    p.add_argument("--student-config", required=True)
    p.add_argument("--fp32-onnx", required=True)
    p.add_argument("--int8-onnx", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-images", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="experiments/student_hailo_30fps/compare_grid.png")
    args = p.parse_args()

    import onnxruntime as ort

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- teacher ----
    tcfg, _ = load_config(args.teacher_config)
    T = build_generator(tcfg, tcfg.data.in_channels, tcfg.data.out_channels).to(device)
    ckpt_path = (ROOT / tcfg.project.output_dir / tcfg.project.experiment
                 / "checkpoints" / "best.pth")
    load_checkpoint(str(ckpt_path), T, map_location=device)
    T.eval()
    print(f"Teacher loaded: {ckpt_path}")

    # ---- student config & data (defines test split) ----
    scfg, _ = load_config(args.student_config)
    pairs = discover_pairs(scfg.data.nir_dir, scfg.data.rgb_dir, scfg.data.extensions)
    _, _, test_pairs = split_pairs(
        pairs, scfg.split.train_ratio, scfg.split.val_ratio, scfg.split.test_ratio,
        seed=scfg.split.seed, scene_regex=scfg.split.scene_regex,
    )
    chosen = {"train": _, "val": _, "test": test_pairs}[args.split]
    full_ds = PairedNIRRGBDataset(test_pairs, image_size=scfg.data.image_size, train=False)

    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(len(full_ds), size=args.n_images, replace=False).tolist())
    ds = Subset(full_ds, idx)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # ---- ONNX sessions ----
    def make_sess(path):
        return ort.InferenceSession(path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    fp32_sess = make_sess(args.fp32_onnx)
    int8_sess = make_sess(args.int8_onnx)
    fp32_in = fp32_sess.get_inputs()[0].name
    int8_in = int8_sess.get_inputs()[0].name

    # ---- metrics accumulators ----
    lpips_fn = LPIPSWrapper(net=scfg.eval.lpips_net).to(device)
    metrics = {k: {"psnr": 0., "ssim": 0., "lpips": 0.} for k in ["teacher", "fp32", "int8"]}

    rows = []
    for batch in loader:
        nir = batch["nir"].to(device)   # [-1,1]
        rgb = batch["rgb"].to(device)   # [-1,1]
        nir_np = batch["nir"].numpy().astype(np.float32)

        with torch.no_grad():
            t_out = T(nir)

        fp32_np = fp32_sess.run(None, {fp32_in: nir_np})[0]
        int8_np = int8_sess.run(None, {int8_in: nir_np})[0]
        fp32_out = torch.from_numpy(fp32_np).to(device)
        int8_out = torch.from_numpy(int8_np).to(device)

        for name, pred in [("teacher", t_out), ("fp32", fp32_out), ("int8", int8_out)]:
            metrics[name]["psnr"]  += compute_psnr(pred, rgb)
            metrics[name]["ssim"]  += compute_ssim(pred, rgb)
            metrics[name]["lpips"] += lpips_fn(pred, rgb)

        # row: GT | Teacher | FP32 | INT8 | NIR
        rows.append(to_01(rgb))
        rows.append(to_01(t_out))
        rows.append(to_01(fp32_out))
        rows.append(to_01(int8_out))
        rows.append(to_01(nir))

    n = args.n_images
    for name, m in metrics.items():
        print(f"{name:8s}  PSNR={m['psnr']/n:.3f}  SSIM={m['ssim']/n:.4f}  LPIPS={m['lpips']/n:.4f}")

    # grid: nrow=5 cols, each group-of-5 is one image row
    grid = vutils.make_grid(torch.cat(rows, dim=0), nrow=5, padding=4, pad_value=1.0)
    import torchvision.transforms.functional as TF
    from PIL import Image
    img = TF.to_pil_image(grid.cpu())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))
    print(f"\nSaved: {out}  ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
