"""Upload all best.pth checkpoints to wandb as model artifacts.

Each experiment gets its own versioned artifact named after the experiment.
Run with WANDB_API_KEY set in environment.

Usage:
    WANDB_API_KEY=<key> python scripts/upload_checkpoints_wandb.py \
        --project nir-rgb [--entity <username>] [--skip-archive]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="nir-rgb")
    p.add_argument("--entity", default=None, help="wandb username/team (auto-detected if omitted)")
    p.add_argument("--skip-archive", action="store_true", help="Skip experiments/_archive/")
    args = p.parse_args()

    import wandb

    # Collect all best.pth paths
    checkpoints = sorted(EXPERIMENTS.glob("**/checkpoints/best.pth"))
    if args.skip_archive:
        checkpoints = [c for c in checkpoints if "_archive" not in c.parts]

    print(f"Found {len(checkpoints)} best.pth checkpoints")
    total_gb = sum(c.stat().st_size for c in checkpoints) / 1e9
    print(f"Total size: {total_gb:.1f} GB\n")

    for ckpt in checkpoints:
        exp_name = ckpt.parts[-3]  # experiments/<exp_name>/checkpoints/best.pth
        size_mb = ckpt.stat().st_size / 1e6

        print(f"Uploading: {exp_name}  ({size_mb:.0f} MB)")
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            job_type="upload",
            name=f"upload-{exp_name}",
            config={"experiment": exp_name, "checkpoint": "best"},
            reinit=True,
        )

        artifact = wandb.Artifact(
            name=exp_name,
            type="model",
            description=f"Best checkpoint for {exp_name}",
            metadata={"experiment": exp_name, "file": "best.pth", "size_mb": round(size_mb)},
        )
        artifact.add_file(str(ckpt), name="best.pth")
        run.log_artifact(artifact)
        run.finish()
        print(f"  done: {exp_name}\n")

    print(f"All {len(checkpoints)} checkpoints uploaded to wandb project '{args.project}'")


if __name__ == "__main__":
    main()
