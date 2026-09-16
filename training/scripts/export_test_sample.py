"""
Export 100 random test-set image pairs (NIR + RGB) resized to 256x256.
Originals are copied unmodified; resized copies go to a separate folder.
"""
import random
import shutil
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parents[1]
NIR_DIR = BASE / "data" / "nir"
RGB_DIR = BASE / "data" / "rgb"
OUT_DIR = BASE / "data" / "test_sample_256"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
SPLIT_SEED = 42
SAMPLE_SEED = 0
N = 100
SIZE = 256


def discover_pairs(nir_dir, rgb_dir):
    nir_index = {p.stem: p for p in nir_dir.rglob("*") if p.suffix.lower() in EXTENSIONS}
    pairs = [(nir_index[p.stem], p) for p in rgb_dir.rglob("*")
             if p.suffix.lower() in EXTENSIONS and p.stem in nir_index]
    pairs.sort(key=lambda x: x[0].as_posix())
    return pairs


def split_pairs(pairs, seed=42):
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return pairs[:n_train], pairs[n_train:n_train + n_val], pairs[n_train + n_val:]


def resize(src, dst, size):
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(dst), img)


def main():
    print("Discovering pairs...")
    pairs = discover_pairs(NIR_DIR, RGB_DIR)
    _, _, test_pairs = split_pairs(pairs, seed=SPLIT_SEED)
    print(f"Test set: {len(test_pairs)} pairs")

    rng = random.Random(SAMPLE_SEED)
    sample = rng.sample(test_pairs, min(N, len(test_pairs)))

    orig_nir = OUT_DIR / "orig" / "nir"
    orig_rgb = OUT_DIR / "orig" / "rgb"
    resized_nir = OUT_DIR / "resized_256" / "nir"
    resized_rgb = OUT_DIR / "resized_256" / "rgb"
    for d in (orig_nir, orig_rgb, resized_nir, resized_rgb):
        d.mkdir(parents=True, exist_ok=True)

    for i, (nir_path, rgb_path) in enumerate(sample):
        print(f"[{i+1:3d}/{N}] {nir_path.name}")
        shutil.copy2(nir_path, orig_nir / nir_path.name)
        shutil.copy2(rgb_path, orig_rgb / rgb_path.name)
        resize(nir_path, resized_nir / nir_path.name, SIZE)
        resize(rgb_path, resized_rgb / rgb_path.name, SIZE)

    print(f"\nDone. Output: {OUT_DIR}")
    print(f"  orig/nir       — {N} original NIR images")
    print(f"  orig/rgb       — {N} original RGB images")
    print(f"  resized_256/nir — {N} NIR images at 256x256")
    print(f"  resized_256/rgb — {N} RGB images at 256x256")


if __name__ == "__main__":
    main()
