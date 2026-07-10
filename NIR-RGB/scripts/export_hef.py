"""Export NAFNet checkpoint to Hailo HEF for deployment on RPi5 AI HAT (Hailo-8L).

INSTALL Hailo DFC first (requires a free Hailo developer account):
    1. Register at https://hailo.ai/developer-zone/
    2. Download hailo_sdk_client-*.whl from the Software Downloads section
    3. pip install /path/to/hailo_sdk_client-*.whl

Export pipeline (run on training machine — GPU not required after training):
    python scripts/export_hef.py --config configs/student_hailo_30fps.yaml

This script:
  1. Loads best.pth and exports a static-batch ONNX (opset 13, batch=1)
  2. Translates ONNX → raw HAR (Hailo Archive)
  3. Calibrates using 512 random val images (INT8 quantization)
  4. Compiles to model.hef targeting Hailo-8L (13 TOPS)

Outputs written to experiments/<experiment>/:
    model_static.onnx   — float32 ONNX with fixed batch=1 (Hailo requirement)
    model_raw.har       — untranslated HAR
    model_optimized.har — INT8-calibrated HAR
    model.hef           — final Hailo Executable Format

Deploy on RPi5 AI HAT:
    Copy model.hef to the Pi, then use HailoRT Python bindings:
        from hailo_platform import HEF, VDevice, FormatType
        hef = HEF("model.hef")
        # see hailo_platform docs for inference loop

Benchmark reference:
    NAFNet width=32, 256x256 on Hailo-8L (13 TOPS INT8): ~50-100 fps expected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export_onnx_static(config_path: str) -> tuple[Path, int]:
    """Load checkpoint and export a fixed-batch-1 ONNX for Hailo DFC."""
    import torch
    from src.utils.config import load_config
    from src.models import build_generator

    cfg, _ = load_config(config_path)
    G = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels)

    run_dir = ROOT / cfg.project.output_dir / cfg.project.experiment
    ckpt = run_dir / "checkpoints" / "best.pth"
    if not ckpt.exists():
        ckpt = run_dir / "checkpoints" / "latest.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint in {run_dir}/checkpoints/")

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    G.load_state_dict(state["G"])
    G.eval()
    print(f"loaded {ckpt}  (epoch={state.get('epoch')})")

    params = sum(p.numel() for p in G.parameters())
    print(f"parameters: {params:,}  ({params/1e6:.2f}M)")

    try:
        from thop import profile as thop_profile
        dummy_thop = torch.zeros(1, cfg.data.in_channels, cfg.data.image_size, cfg.data.image_size)
        flops, _ = thop_profile(G, inputs=(dummy_thop,), verbose=False)
        print(f"FLOPs @ {cfg.data.image_size}x{cfg.data.image_size}: {flops/1e9:.2f} GFLOPs")
    except ImportError:
        pass

    run_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = run_dir / "model_static.onnx"
    size = cfg.data.image_size
    dummy = torch.zeros(1, cfg.data.in_channels, size, size)

    # Static batch=1 — Hailo DFC requires fixed input shapes.
    torch.onnx.export(
        G, dummy, str(onnx_path),
        opset_version=13,
        input_names=["nir"],
        output_names=["rgb"],
        dynamic_axes=None,          # no dynamic axes
        do_constant_folding=True,
    )
    print(f"static ONNX: {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)")
    return onnx_path, size


def load_calibration_images(config_path: str, n: int = 512) -> np.ndarray:
    """Load n random val images as float32 numpy array (N, C, H, W) in [-1, 1]."""
    import torch
    from src.utils.config import load_config
    from src.data.dataset import build_loaders

    cfg, _ = load_config(config_path)
    _, val_loader, _ = build_loaders(cfg)

    images = []
    for batch in val_loader:
        nir = batch["nir"]            # (B, C, H, W)  float32 in [-1, 1]
        for img in nir:
            images.append(img.numpy())
            if len(images) >= n:
                break
        if len(images) >= n:
            break

    calib = np.stack(images[:n], axis=0)
    print(f"calibration set: {calib.shape}  range [{calib.min():.2f}, {calib.max():.2f}]")
    return calib


def compile_hef(onnx_path: Path, calib: np.ndarray, image_size: int) -> Path:
    """Translate ONNX → HAR → optimise → compile → HEF."""
    from hailo_sdk_client import ClientRunner

    run_dir = onnx_path.parent

    # --- Step 1: translate ONNX → raw HAR ---
    runner = ClientRunner(hw_arch="hailo8l")
    runner.translate_onnx_model(
        str(onnx_path),
        "nir2rgb",
        start_node_names=["nir"],
        end_node_names=["rgb"],
        net_input_shapes={"nir": [1, 3, image_size, image_size]},
    )
    raw_har = run_dir / "model_raw.har"
    runner.save_har(str(raw_har))
    print(f"raw HAR: {raw_har}")

    # --- Step 2: calibrate + INT8 quantisation ---
    runner.load_har(str(raw_har))
    runner.optimize(calib)
    opt_har = run_dir / "model_optimized.har"
    runner.save_har(str(opt_har))
    print(f"optimized HAR: {opt_har}")

    # --- Step 3: compile → HEF ---
    runner.load_har(str(opt_har))
    hef_bytes = runner.compile()
    hef_path = run_dir / "model.hef"
    with open(str(hef_path), "wb") as f:
        f.write(hef_bytes)
    size_mb = hef_path.stat().st_size / 1e6
    print(f"HEF: {hef_path}  ({size_mb:.1f} MB)")
    print()
    print("Deploy on RPi5 AI HAT:")
    print(f"  scp {hef_path} pi@<rpi5-ip>:~/nir2rgb/model.hef")
    print("  # then run inference via hailo_platform Python SDK on the Pi")
    return hef_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="YAML config (e.g. configs/student_hailo_30fps.yaml)")
    ap.add_argument("--calib-n", type=int, default=512,
                    help="Number of val images for INT8 calibration (default 512)")
    args = ap.parse_args()

    onnx_path, image_size = export_onnx_static(args.config)

    try:
        import hailo_sdk_client  # noqa: F401
    except ImportError:
        print()
        print("=" * 60)
        print("hailo_sdk_client not installed — ONNX exported, HEF skipped.")
        print("To compile to HEF:")
        print("  1. Register at https://hailo.ai/developer-zone/")
        print("  2. Download hailo_sdk_client-*.whl")
        print("  3. pip install /path/to/hailo_sdk_client-*.whl")
        print("  4. Re-run this script")
        print("=" * 60)
        sys.exit(0)

    print("\nLoading calibration images...")
    calib = load_calibration_images(args.config, n=args.calib_n)

    print("\nCompiling to HEF...")
    compile_hef(onnx_path, calib, image_size)
