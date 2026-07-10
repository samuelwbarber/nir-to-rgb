"""Extract Places365-Standard 256x256 + flatten into data/synth-rgb via symlinks.

The tar is ~24 GB and ~1.8M images; symlinks (not copies) keep disk usage near
zero on the target side. discover_pairs() uses rglob so the sharded layout is
fine — we shard by md5(filename)[:2] to keep ~18k files per directory.

Idempotent: re-running skips already-extracted tarballs and already-symlinked
files.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PLACES_DIR = Path("/vol/bitbucket/sb1522/places365")
TAR = PLACES_DIR / "train_256_places365standard.tar"
EXTRACT_DIR = PLACES_DIR / "extracted"
# Marker created when extraction finishes — guard against partial extract.
EXTRACT_DONE = EXTRACT_DIR / ".extract_complete"

PROJECT_ROOT = Path("/vol/bitbucket/sb1522/vm-backup/NIR-RGB")
SYNTH_RGB = PROJECT_ROOT / "data/synth-rgb"

REPORT_EVERY = 50_000


def extract():
    if EXTRACT_DONE.exists():
        print(f"[setup_places365] extraction already complete at {EXTRACT_DIR}")
        return
    if not TAR.exists():
        raise FileNotFoundError(f"missing tar: {TAR}")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup_places365] extracting {TAR} -> {EXTRACT_DIR} (this takes ~30-60min)")
    t0 = time.time()
    # tar shell command is much faster than tarfile module for big archives
    subprocess.check_call(["tar", "-xf", str(TAR), "-C", str(EXTRACT_DIR)])
    EXTRACT_DONE.touch()
    print(f"[setup_places365] extraction done in {time.time() - t0:.0f}s")


def link_flat():
    SYNTH_RGB.mkdir(parents=True, exist_ok=True)
    # Pre-create shards
    for i in range(256):
        (SYNTH_RGB / f"{i:02x}").mkdir(exist_ok=True)

    t0 = time.time()
    n_seen = n_linked = n_skipped = 0
    seen_stems = set()
    for img in EXTRACT_DIR.rglob("*.jpg"):
        n_seen += 1
        rel = img.relative_to(EXTRACT_DIR)
        flat = "places_" + "_".join(rel.with_suffix("").parts).lstrip("_") + ".jpg"
        stem = Path(flat).stem
        if stem in seen_stems:
            n_skipped += 1
            continue
        seen_stems.add(stem)
        shard = SYNTH_RGB / hashlib.md5(flat.encode()).hexdigest()[:2]
        link = shard / flat
        if not link.exists():
            link.symlink_to(img.resolve())
            n_linked += 1
        if n_seen % REPORT_EVERY == 0:
            print(f"[setup_places365] {n_seen} scanned / {n_linked} linked / {n_skipped} dup-stem-skipped "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[setup_places365] DONE: scanned={n_seen} linked={n_linked} "
          f"dup_stem_skipped={n_skipped} in {time.time() - t0:.0f}s")
    print(f"[setup_places365] target dir: {SYNTH_RGB}")


def main():
    extract()
    link_flat()


if __name__ == "__main__":
    main()
