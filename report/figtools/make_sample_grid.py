"""Regenerate a NIR | predicted | ground-truth sample grid from a checkpoint.

Reuses the project's own NAFNet and dataset-split code (ground truth) so the
shown pairs are genuine held-out test samples. CPU inference.

Usage:
    py -3 make_sample_grid.py <repo_dir> <ckpt> <nir_dir> <rgb_dir> <out.png> [n_rows]
"""
import sys
import numpy as np
import cv2
import torch

REPO, CKPT, NIR_DIR, RGB_DIR, OUT = sys.argv[1:6]
N_ROWS = int(sys.argv[6]) if len(sys.argv) > 6 else 5
sys.path.insert(0, REPO)

from src.models.nafnet import NAFNet

S = 256
EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# --- dataset helpers, inlined verbatim from src/data/dataset.py (ground truth) ---
import random
from pathlib import Path


def discover_pairs(nir_dir, rgb_dir, extensions):
    nir_dir, rgb_dir = Path(nir_dir), Path(rgb_dir)
    exts = {e.lower() for e in extensions}
    nir_index = {}
    for p in nir_dir.rglob("*"):
        if p.suffix.lower() in exts:
            nir_index[p.stem] = p
    pairs = []
    for p in rgb_dir.rglob("*"):
        if p.suffix.lower() in exts and p.stem in nir_index:
            pairs.append((nir_index[p.stem], p))
    pairs.sort(key=lambda x: x[0].as_posix())
    return pairs


def split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed=42):
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]


def _read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read {path}")
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

model = NAFNet(in_channels=3, out_channels=3, width=64,
               enc_blk_nums=(2, 2, 4, 8), middle_blk_num=12,
               dec_blk_nums=(2, 2, 2, 2), global_residual=False, output_tanh=True)
state = torch.load(CKPT, map_location="cpu")
model.load_state_dict(state["G"])
model.eval()
print("loaded G from", CKPT, "| epoch", state.get("epoch"))

pairs = discover_pairs(NIR_DIR, RGB_DIR, EXTS)
_, _, test = split_pairs(pairs, 0.8, 0.1, 0.1, seed=42)
print(f"{len(pairs)} pairs | test={len(test)}")

# evenly spaced test samples for a representative strip
idxs = [int(round(k * (len(test) - 1) / (N_ROWS - 1))) for k in range(N_ROWS)]


def prep(path):
    img = _read_image(path)                       # RGB uint8
    img = cv2.resize(img, (S, S), interpolation=cv2.INTER_LINEAR)
    return img


rows = []
for i in idxs:
    nir_p, rgb_p = test[i]
    nir = prep(nir_p)
    rgb = prep(rgb_p)
    x = torch.from_numpy(nir.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        y = model(x)
    fake = ((y.clamp(-1, 1)[0] + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
    rows.append(np.concatenate([nir, fake, rgb], axis=1))

grid = np.concatenate(rows, axis=0)

# column header strip
hdr_h = 28
hdr = np.full((hdr_h, grid.shape[1], 3), 255, np.uint8)
for k, label in enumerate(["NIR input", "Predicted RGB", "Ground-truth RGB"]):
    cv2.putText(hdr, label, (k * S + 8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
out = np.concatenate([hdr, grid], axis=0)

cv2.imwrite(OUT, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("wrote", OUT, out.shape)
