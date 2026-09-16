import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, save_config
from src.data.dataset import discover_pairs, split_pairs, PairedNIRRGBDataset, load_mp_cache
from src.training.distill_trainer import DistillTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/student_distill.yaml")
    p.add_argument("--name", type=str, default=None, help="override experiment name")
    args = p.parse_args()

    cfg, raw = load_config(args.config)
    if args.name:
        cfg.project.experiment = args.name
        raw["project"]["experiment"] = args.name

    if not hasattr(cfg, "distill"):
        raise RuntimeError("config is missing a 'distill' section (teacher_checkpoint, teacher).")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        print(f"warning: requested device={cfg.device}, using {device}")

    pairs = discover_pairs(cfg.data.nir_dir, cfg.data.rgb_dir, cfg.data.extensions)
    if not pairs:
        raise RuntimeError(
            f"no pairs found. nir_dir={cfg.data.nir_dir} rgb_dir={cfg.data.rgb_dir}."
        )
    print(f"discovered {len(pairs)} paired samples")

    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        cfg.split.train_ratio,
        cfg.split.val_ratio,
        cfg.split.test_ratio,
        seed=cfg.split.seed,
        scene_regex=cfg.split.scene_regex,
    )
    print(f"split: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    aug = vars(cfg.augment)
    mp_cache_path = getattr(cfg.data, "mp_cache", None)
    mp_cache = load_mp_cache(mp_cache_path)
    if mp_cache_path:
        n_with_crop = sum(1 for v in mp_cache.values() if v.get("crop_bbox"))
        print(f"loaded mp_cache: {len(mp_cache)} entries (crop_bbox={n_with_crop})")
    train_ds = PairedNIRRGBDataset(train_pairs, image_size=cfg.data.image_size, train=True,
                                   aug_cfg=aug, mp_cache=mp_cache)
    val_ds = PairedNIRRGBDataset(val_pairs, image_size=cfg.data.image_size, train=False,
                                 mp_cache=mp_cache)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=cfg.train.num_workers > 0,
    )

    run_dir = Path(cfg.project.output_dir) / cfg.project.experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(raw, run_dir / "config.yaml")

    trainer = DistillTrainer(cfg, train_loader, val_loader, run_dir, device)
    trainer.fit()


if __name__ == "__main__":
    main()
