"""Back up finished-run checkpoints to W&B as artifacts.

One W&B run per local experiment. Uploads best.pth + final.pth (when present)
as a model artifact, attaches config.yaml + train.log as files, parses summary
metrics from the [val] / [best] log lines.

Auth: expects WANDB_API_KEY in the env or a prior `wandb login`.
"""
import argparse
import re
import sys
from pathlib import Path

import wandb
import yaml

ROOT = Path(__file__).resolve().parents[1]


VAL_RE = re.compile(r"\[val\] epoch=(\d+)\s+(.*)")
KV_RE = re.compile(r"(\w+)=([0-9.eE+-]+)")
BEST_RE = re.compile(r"\[best\] epoch (\d+) new best val/(\S+) = ([0-9.eE+-]+)")


def parse_log(log_path: Path):
    """Return (best_epoch, best_metric, best_value, final_val_kv, all_best_kv)."""
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore").replace("\r", "\n")
    final_val_kv = {}
    final_epoch = -1
    best = None  # (epoch, metric_name, value)
    best_val_kv = {}
    val_by_epoch = {}
    for line in text.splitlines():
        m_val = VAL_RE.search(line)
        if m_val:
            ep = int(m_val.group(1))
            kv = {k: float(v) for k, v in KV_RE.findall(m_val.group(2))}
            val_by_epoch[ep] = kv
            if ep >= final_epoch:
                final_epoch = ep
                final_val_kv = kv
        m_best = BEST_RE.search(line)
        if m_best:
            best = (int(m_best.group(1)), m_best.group(2), float(m_best.group(3)))
    if best is not None and best[0] in val_by_epoch:
        best_val_kv = val_by_epoch[best[0]]
    return {
        "final_epoch": final_epoch,
        "final_val": final_val_kv,
        "best_epoch": best[0] if best else None,
        "best_metric": best[1] if best else None,
        "best_value": best[2] if best else None,
        "best_val": best_val_kv,
    }


def file_size_mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024)


def upload_run(entity: str, project: str, exp_dir: Path, extra_files: list[Path]):
    name = exp_dir.name
    cfg_path = exp_dir / "config.yaml"
    log_path = exp_dir / "train.log"
    ck = exp_dir / "checkpoints"

    ckpts = []
    for fname in ("best.pth", "final.pth", "latest.pth"):
        p = ck / fname
        if p.exists():
            ckpts.append(p)

    if not ckpts:
        print(f"[skip] {name}: no checkpoints found in {ck}")
        return

    print(f"\n=== Uploading {name} ===")
    print(f"  exp_dir: {exp_dir}")
    print(f"  checkpoints: {[c.name + f' ({file_size_mb(c):.0f} MB)' for c in ckpts]}")
    print(f"  extra files: {[f.name for f in extra_files]}")

    cfg_dict = {}
    if cfg_path.exists():
        try:
            cfg_dict = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception as e:
            print(f"  [warn] couldn't parse config.yaml: {e}")

    summary = parse_log(log_path) or {}
    print(f"  parsed log: best ep={summary.get('best_epoch')} "
          f"{summary.get('best_metric')}={summary.get('best_value')}; "
          f"final ep={summary.get('final_epoch')}")

    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        job_type="checkpoint-backup",
        config={"exp_dir": str(exp_dir), "config_yaml": cfg_dict},
        reinit=True,
        settings=wandb.Settings(silent="true", _disable_stats=True),
    )
    try:
        # Files: config + truncated log (last 200KB).
        if cfg_path.exists():
            wandb.save(str(cfg_path), base_path=str(exp_dir.parent), policy="now")
        if log_path.exists():
            log_bytes = log_path.read_bytes()
            if len(log_bytes) > 200_000:
                tail = log_bytes[-200_000:]
                short = exp_dir / "train.tail.log"
                short.write_bytes(tail)
                wandb.save(str(short), base_path=str(exp_dir.parent), policy="now")
            else:
                wandb.save(str(log_path), base_path=str(exp_dir.parent), policy="now")

        # Summary metrics
        if summary.get("best_metric") is not None:
            wandb.summary[f"best/{summary['best_metric']}"] = summary["best_value"]
            wandb.summary["best/epoch"] = summary["best_epoch"]
        for k, v in summary.get("best_val", {}).items():
            wandb.summary[f"best_val/{k}"] = v
        for k, v in summary.get("final_val", {}).items():
            wandb.summary[f"final_val/{k}"] = v
        if summary.get("final_epoch", -1) >= 0:
            wandb.summary["final/epoch"] = summary["final_epoch"]

        # Checkpoints as one model artifact
        art = wandb.Artifact(
            name=f"{name}-checkpoints",
            type="model",
            metadata={
                "best_epoch": summary.get("best_epoch"),
                "best_metric": summary.get("best_metric"),
                "best_value": summary.get("best_value"),
                "final_epoch": summary.get("final_epoch"),
            },
        )
        for c in ckpts:
            print(f"  + add {c.name} ({file_size_mb(c):.0f} MB) ...")
            art.add_file(str(c), name=f"checkpoints/{c.name}")
        if cfg_path.exists():
            art.add_file(str(cfg_path), name="config.yaml")
        run.log_artifact(art).wait()
        print(f"  ckpts artifact uploaded.")

        # Optional extra-files artifact (for deployment exports etc.)
        if extra_files:
            xart = wandb.Artifact(
                name=f"{name}-exports",
                type="deployment-export",
                metadata={"source_run": name},
            )
            for f in extra_files:
                if f.exists():
                    print(f"  + extra {f.name} ({file_size_mb(f):.1f} MB) ...")
                    xart.add_file(str(f), name=f.name)
            run.log_artifact(xart).wait()
            print(f"  exports artifact uploaded.")

        print(f"  W&B run URL: {run.url}")
    finally:
        run.finish()


# Each entry: (experiment_dir_name, [extra_files_for_export_artifact])
RUNS = [
    ("ablation_08_nir_l1boost_gan_msssim_bf16_200_seed8065", []),
    ("pix2pix_nir_200", []),
    ("student_08_distill_downstream", [
        "model.onnx", "model_q.onnx",
        "model_litert_fp32.tflite", "model_litert_int8.tflite",
    ]),
    ("student_hailo_30fps", [
        "model.onnx", "model_q.onnx", "model_static.onnx",
        "model_litert_fp32.tflite", "model_litert_int8.tflite",
    ]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="sbarber9876-imperial-college-london")
    p.add_argument("--project", default="nir-rgb-checkpoints")
    p.add_argument("--only", nargs="*", help="restrict to a subset of run names")
    args = p.parse_args()

    print(f"Target: {args.entity}/{args.project}")
    only = set(args.only) if args.only else None
    for run_name, extras in RUNS:
        if only is not None and run_name not in only:
            continue
        exp_dir = ROOT / "experiments" / run_name
        if not exp_dir.is_dir():
            print(f"[skip] {run_name}: directory not found at {exp_dir}")
            continue
        extra_paths = [exp_dir / e for e in extras]
        try:
            upload_run(args.entity, args.project, exp_dir, extra_paths)
        except Exception as e:
            print(f"[FAIL] {run_name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
