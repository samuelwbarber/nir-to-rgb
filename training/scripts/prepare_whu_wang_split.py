"""
Extract 256x256 patches from WHU-SEN-City using Wang et al. 2019 city-level split.

Train: WHU-SEN-City/train/ (26 cities)  -> train_CITY_NNNNN.png
Test:  WHU-SEN-City/test/  (8 cities)   -> test_CITY_NNNNN.png

SAR processing: float32 big-endian amplitude -> 10*log10 -> global min-max norm -> uint8 x3
Global stats from total_minmax.mat: minmax[0][2]=vv_max_db, minmax[0][3]=vv_min_db
"""
import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import scipy.io
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC  = Path('/vol/bitbucket/sb1522/vm-backup/WHU-SEN-City')

PATCH_SIZE = 256
PATCH_STEP = 128


def parse_envi_hdr(text):
    info = {}
    for line in text.splitlines():
        if '=' in line and not line.strip().startswith(';'):
            k, v = line.split('=', 1)
            info[k.strip().lower()] = v.strip()
    return info


def read_vv_amplitude(zf, hdr_name, img_name):
    hdr = parse_envi_hdr(zf.read(hdr_name).decode('utf-8', errors='replace'))
    samples   = int(hdr['samples'])
    lines     = int(hdr['lines'])
    # data type 4 = float32; byte order 1 = big-endian
    dtype     = '>f4'
    raw       = zf.read(img_name)
    arr       = np.frombuffer(raw, dtype=dtype).reshape(lines, samples)
    return arr.astype(np.float32)          # now native-endian float32


def amp_to_normalised_u8(amp, vv_min_db, vv_max_db):
    db   = 10.0 * np.log10(np.maximum(amp, 1e-10))
    norm = np.clip((db - vv_min_db) / (vv_max_db - vv_min_db), 0.0, 1.0)
    return (norm * 255).astype(np.uint8)


def process_city(zip_path, prefix, vv_min_db, vv_max_db,
                 out_sar, out_rgb, counter_start, dry_run=False):
    city = zip_path.stem
    patches_written = 0

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        # Sentinel-2 optical: contains 'S2' and ends with .png (not _TC_RGB.png)
        rgb_files = [n for n in names
                     if 'S2' in n and n.endswith('.png') and not n.endswith('_TC_RGB.png')]
        if not rgb_files:
            raise RuntimeError(f"No Sentinel-2 RGB found in {zip_path}")
        rgb_name = sorted(rgb_files)[0]

        # VV amplitude
        hdr_files = sorted(n for n in names if n.endswith('Amplitude_VV.hdr'))
        img_files = sorted(n for n in names if n.endswith('Amplitude_VV.img'))
        if not hdr_files or not img_files:
            raise RuntimeError(f"No Amplitude_VV found in {zip_path}")

        # Load optical RGB
        rgb_bytes = zf.read(rgb_name)
        rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to decode RGB in {zip_path}")

        # Load and convert SAR
        amp = read_vv_amplitude(zf, hdr_files[0], img_files[0])

    sar_h, sar_w = amp.shape
    rgb_h, rgb_w = rgb.shape[:2]

    # Co-register: resize RGB to SAR pixel grid (< 1% difference in practice)
    if (rgb_h, rgb_w) != (sar_h, sar_w):
        rgb = cv2.resize(rgb, (sar_w, sar_h), interpolation=cv2.INTER_LINEAR)

    sar_u8  = amp_to_normalised_u8(amp, vv_min_db, vv_max_db)
    sar_rgb = np.stack([sar_u8, sar_u8, sar_u8], axis=2)

    rows = (sar_h - PATCH_SIZE) // PATCH_STEP + 1
    cols = (sar_w - PATCH_SIZE) // PATCH_STEP + 1

    for r in range(rows):
        for c in range(cols):
            idx  = counter_start + patches_written
            fname = f"{prefix}_{city}_{idx:05d}.png"
            r0, c0 = r * PATCH_STEP, c * PATCH_STEP

            if not dry_run:
                sar_p = sar_rgb[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
                rgb_p = rgb    [r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
                cv2.imwrite(str(out_sar / fname), sar_p)
                cv2.imwrite(str(out_rgb / fname), rgb_p)
            patches_written += 1

    return patches_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/whu-sen-city-wang-split',
                    help='Output dataset dir relative to repo root')
    ap.add_argument('--dry-run', action='store_true',
                    help='Count patches without writing files')
    args = ap.parse_args()

    out_root = ROOT / args.out
    out_sar  = out_root / 'sar'
    out_rgb  = out_root / 'rgb'

    if not args.dry_run:
        out_sar.mkdir(parents=True, exist_ok=True)
        out_rgb.mkdir(parents=True, exist_ok=True)

    # Load global normalisation stats
    mat      = scipy.io.loadmat(str(SRC / 'total_minmax.mat'))
    minmax   = mat['minmax'].flatten()
    vv_max_db = float(minmax[2])   # 27.10
    vv_min_db = float(minmax[3])   # 19.72
    print(f"VV dB range: [{vv_min_db:.4f}, {vv_max_db:.4f}]")

    manifest = {
        'vv_min_db': vv_min_db, 'vv_max_db': vv_max_db,
        'patch_size': PATCH_SIZE, 'patch_step': PATCH_STEP,
        'split': 'wang_city_level',
        'train_cities': [], 'test_cities': [],
    }

    train_count = 0
    test_count  = 0

    for split_name, split_dir in [('train', SRC / 'train'), ('test', SRC / 'test')]:
        zips = sorted(split_dir.glob('*.zip'))
        prefix = split_name
        counter = 0
        print(f"\n--- {split_name} cities ({len(zips)}) ---")
        for zp in tqdm(zips, desc=split_name):
            n = process_city(zp, prefix, vv_min_db, vv_max_db,
                             out_sar, out_rgb, counter, dry_run=args.dry_run)
            tqdm.write(f"  {zp.stem}: {n} patches  (running total: {counter + n})")
            manifest[f'{split_name}_cities'].append({'city': zp.stem, 'patches': n})
            counter += n
        if split_name == 'train':
            train_count = counter
        else:
            test_count = counter

    manifest['train_count'] = train_count
    manifest['test_count']  = test_count
    print(f"\nDone: {train_count} train + {test_count} test = {train_count + test_count} total patches")

    if not args.dry_run:
        with open(out_root / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest written to {out_root}/manifest.json")


if __name__ == '__main__':
    main()
