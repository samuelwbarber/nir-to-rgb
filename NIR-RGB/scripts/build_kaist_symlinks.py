"""Flatten KAIST HF mirror into nir_dir/rgb_dir layout for discover_pairs.

For each pair under kaist_train/{setXX}/{VYYY,}/{lwir,visible}/I*.jpg,
create symlinks:
  data/kaist-nir/setXX_VYYY_Inum.jpg -> ../../../kaist-hf/kaist_train/setXX/VYYY/lwir/Inum.jpg
  data/kaist-rgb/setXX_VYYY_Inum.jpg -> ../../../kaist-hf/kaist_train/setXX/VYYY/visible/Inum.jpg
"""
from pathlib import Path

KAIST = Path("/vol/bitbucket/sb1522/kaist-hf/kaist_train")
NIR = Path("/vol/bitbucket/sb1522/vm-backup/NIR-RGB/data/kaist-nir")
RGB = Path("/vol/bitbucket/sb1522/vm-backup/NIR-RGB/data/kaist-rgb")

NIR.mkdir(parents=True, exist_ok=True)
RGB.mkdir(parents=True, exist_ok=True)

def video_dirs():
    for set_dir in sorted(KAIST.iterdir()):
        if not set_dir.is_dir():
            continue
        # Two layouts: set/V*/{lwir,visible} or set/{lwir,visible}
        v_dirs = [p for p in set_dir.iterdir() if p.is_dir() and p.name.startswith("V")]
        if v_dirs:
            for vd in sorted(v_dirs):
                yield set_dir.name, vd.name, vd
        elif (set_dir / "lwir").is_dir() and (set_dir / "visible").is_dir():
            yield set_dir.name, "flat", set_dir

n_pairs = 0
n_skipped = 0
for set_name, v_name, base in video_dirs():
    lwir = base / "lwir"
    vis = base / "visible"
    if not (lwir.is_dir() and vis.is_dir()):
        continue
    lwir_stems = {p.stem: p for p in lwir.iterdir() if p.is_file()}
    for vp in sorted(vis.iterdir()):
        if not vp.is_file():
            continue
        lp = lwir_stems.get(vp.stem)
        if lp is None:
            n_skipped += 1
            continue
        name = f"{set_name}_{v_name}_{vp.name}"
        nir_link = NIR / name
        rgb_link = RGB / name
        if not nir_link.is_symlink():
            nir_link.symlink_to(lp)
        if not rgb_link.is_symlink():
            rgb_link.symlink_to(vp)
        n_pairs += 1

print(f"created/verified {n_pairs} pairs, skipped {n_skipped} unmatched")
print(f"nir_dir: {NIR} -> {len(list(NIR.iterdir()))} entries")
print(f"rgb_dir: {RGB} -> {len(list(RGB.iterdir()))} entries")
