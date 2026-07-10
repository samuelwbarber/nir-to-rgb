"""Run the trained reverse RGB->NIR model on every data/synth-rgb image.

Output: data/synth-nir/<shard>/<filename>.jpg as 1-channel grayscale JPG (matches
the real NIR's effective representation). Resumable — pre-existing outputs are
skipped, so this can be killed and restarted any number of times.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/vol/bitbucket/sb1522/vm-backup/NIR-RGB")
sys.path.insert(0, str(ROOT))
from src.utils.config import load_config
from src.models.nafnet import NAFNet

REVERSE_CFG = ROOT / "configs/rgb_to_nir_v1.yaml"
REVERSE_CKPT = ROOT / "experiments/rgb_to_nir_v1/checkpoints/best.pth"
SRC_DIR = ROOT / "data/synth-rgb"
DST_DIR = ROOT / "data/synth-nir"
BATCH_SIZE = 16
JPG_QUALITY = 92
REPORT_EVERY = 5000


def build_model(device):
    cfg, _ = load_config(REVERSE_CFG)
    g = cfg.model.generator
    G = NAFNet(
        in_channels=cfg.data.in_channels,
        out_channels=cfg.data.out_channels,
        width=g.width,
        enc_blk_nums=tuple(g.enc_blk_nums),
        middle_blk_num=g.middle_blk_num,
        dec_blk_nums=tuple(g.dec_blk_nums),
        dropout=g.dropout,
        global_residual=g.global_residual,
        output_tanh=g.output_tanh,
    )
    state = torch.load(str(REVERSE_CKPT), map_location="cpu", weights_only=False)
    G.load_state_dict(state["G"])
    print(f"loaded reverse model from {REVERSE_CKPT} (epoch={state.get('epoch')})")
    return G.eval().to(device), int(cfg.data.image_size)


def read_rgb(path, image_size):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[:2] != (image_size, image_size):
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0  # (3,H,W) in [-1,1]


def write_nir_gray(tensor_neg11, dst_path):
    # tensor (3,H,W) in [-1,1]. Average the 3 channels (model output is RGB-shaped
    # but the NIR target was always grayscale-tripled, so the 3 channels collapse to
    # the same value on convergence; averaging is robust to small per-channel drift).
    arr = ((tensor_neg11.clamp(-1, 1) + 1) * 127.5).mean(dim=0).cpu().numpy().astype(np.uint8)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), arr, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])


def main():
    if not REVERSE_CKPT.exists():
        raise FileNotFoundError(f"reverse model not yet trained: {REVERSE_CKPT}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, image_size = build_model(device)

    DST_DIR.mkdir(parents=True, exist_ok=True)
    all_paths = sorted(SRC_DIR.rglob("*.jpg"))
    print(f"[generate_synthetic_nir] {len(all_paths)} RGB candidates in {SRC_DIR}")

    todo = []
    n_skipped = 0
    for p in all_paths:
        shard = p.parent.name
        dst = DST_DIR / shard / p.name
        if dst.exists():
            n_skipped += 1
        else:
            todo.append((p, dst))
    print(f"[generate_synthetic_nir] {n_skipped} already done, {len(todo)} to generate")

    if not todo:
        print("[generate_synthetic_nir] nothing to do")
        return

    t0 = time.time()
    batch_paths, batch_imgs = [], []
    n_done = 0
    for src, dst in todo:
        img = read_rgb(src, image_size)
        if img is None:
            continue
        batch_paths.append(dst)
        batch_imgs.append(img)
        if len(batch_imgs) == BATCH_SIZE:
            x = torch.stack(batch_imgs).to(device, non_blocking=True)
            with torch.no_grad():
                out = G(x)
            for dst_path, o in zip(batch_paths, out):
                write_nir_gray(o, dst_path)
            n_done += len(batch_paths)
            batch_paths, batch_imgs = [], []
            if n_done % REPORT_EVERY < BATCH_SIZE:
                rate = n_done / max(1e-6, time.time() - t0)
                eta = (len(todo) - n_done) / max(1, rate)
                print(f"  {n_done}/{len(todo)} ({rate:.1f} img/s, ETA {eta / 60:.0f}min)", flush=True)
    if batch_imgs:
        x = torch.stack(batch_imgs).to(device, non_blocking=True)
        with torch.no_grad():
            out = G(x)
        for dst_path, o in zip(batch_paths, out):
            write_nir_gray(o, dst_path)
        n_done += len(batch_paths)

    print(f"[generate_synthetic_nir] DONE: generated {n_done} synthetic NIRs in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
