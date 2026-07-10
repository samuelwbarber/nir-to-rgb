import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, save_config
from src.data.dataset import discover_pairs, split_pairs, PairedNIRRGBDataset, load_mp_cache
from src.training.trainer import Trainer


def _seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/baseline.yaml")
    p.add_argument("--name", type=str, default=None, help="override experiment name")
    args = p.parse_args()

    cfg, raw = load_config(args.config)
    if args.name:
        cfg.project.experiment = args.name
        raw["project"]["experiment"] = args.name

    train_seed = getattr(cfg.train, "seed", None)
    if train_seed is not None:
        train_seed = int(train_seed)
        print(f"training seed: {train_seed}")
        _seed_everything(train_seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        print(f"warning: requested device={cfg.device}, using {device}")

    # Optional: swap roles of nir/rgb (used to train the reverse model RGB->NIR
    # with the same trainer code). The generator's job is always: input "nir" -> output "rgb".
    swap = bool(getattr(cfg.data, "swap_nir_rgb", False))
    nir_dir = cfg.data.rgb_dir if swap else cfg.data.nir_dir
    rgb_dir = cfg.data.nir_dir if swap else cfg.data.rgb_dir
    if swap:
        print(f"[swap_nir_rgb=True] training reverse model: input={nir_dir} target={rgb_dir}")

    pairs = discover_pairs(nir_dir, rgb_dir, cfg.data.extensions)
    if not pairs:
        raise RuntimeError(
            f"no pairs found. nir_dir={nir_dir} rgb_dir={rgb_dir}. "
            f"filenames must match between the two directories."
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

    # Optional: mix in unpaired synthetic pairs (produced by a reverse RGB->NIR
    # model — back-translation augmentation). val/test stay real-only.
    synth_nir_dir = getattr(cfg.data, "synth_nir_dir", None)
    synth_rgb_dir = getattr(cfg.data, "synth_rgb_dir", None)
    real_weight = int(getattr(cfg.data, "real_weight", 1))
    if synth_nir_dir and synth_rgb_dir:
        synth_pairs = discover_pairs(synth_nir_dir, synth_rgb_dir, cfg.data.extensions)
        print(f"discovered {len(synth_pairs)} synthetic pairs (real_weight={real_weight})")
        # Upweight real pairs by duplicating them — simplest WeightedRandomSampler equivalent.
        train_pairs = train_pairs * max(1, real_weight) + synth_pairs
        print(f"train mix after weighting: total={len(train_pairs)} (real_dupes={max(1, real_weight)}x)")

    aug = vars(cfg.augment)
    mp_cache_path = getattr(cfg.data, "mp_cache", None)
    mp_cache = load_mp_cache(mp_cache_path)
    if mp_cache_path:
        n_with_crop = sum(1 for v in mp_cache.values() if v.get("crop_bbox"))
        n_with_region = sum(1 for v in mp_cache.values() if v.get("region_bbox"))
        print(f"loaded mp_cache: {len(mp_cache)} entries "
              f"(crop_bbox={n_with_crop}, region_bbox={n_with_region})")
    train_ds = PairedNIRRGBDataset(train_pairs, image_size=cfg.data.image_size, train=True,
                                   aug_cfg=aug, mp_cache=mp_cache)
    val_ds = PairedNIRRGBDataset(val_pairs, image_size=cfg.data.image_size, train=False,
                                 mp_cache=mp_cache)

    loader_generator = None
    worker_init_fn = None
    if train_seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(train_seed)

        def worker_init_fn(worker_id):
            worker_seed = train_seed + worker_id
            random.seed(worker_seed)
            try:
                import numpy as np
                np.random.seed(worker_seed)
            except Exception:
                pass

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
        generator=loader_generator,
        worker_init_fn=worker_init_fn,
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

    trainer = Trainer(cfg, train_loader, val_loader, run_dir, device)
    trainer.fit()


if __name__ == "__main__":
    main()
