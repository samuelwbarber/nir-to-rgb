"""
Proper PyTorch → TFLite export via litert-torch + ai-edge-quantizer.

Why this replaces export_tflite.py (onnx2tf path):
  onnx2tf converts NCHW ONNX graphs to NHWC TFLite, but its layout-conversion pass
  mishandles LayerNorm's ReduceMean outputs: it double-applies the NCHW→NHWC transpose,
  producing broadcast tensors shaped [1,1,H,W] instead of [1,H,W,1].  That causes
  SUB/DIV failures against NHWC activations [1,H,W,C] at XNNPACK delegate prep time.

  litert-torch (Google's official PyTorch→TFLite tool) converts via torch.export +
  MLIR.  All layout handling happens at the PyTorch IR level, so LayerNorm/ReduceMean
  axes are converted correctly.  Zero NCHW broadcast tensors in the output graph.

Outputs (alongside model.onnx in the experiment dir):
  model_litert_fp32.tflite   — float32, XNNPACK fp32 path
  model_litert_int8.tflite   — Conv/Mul/Add quantized int8, LayerNorm stays fp32

Dependencies (server-side):
  pip install litert-torch ai-edge-quantizer

Pi-side:
  pip install tflite-runtime

Usage:
  python scripts/export_litert.py \\
      --config experiments/student_08_distill_downstream/config.yaml \\
      --calib_dir data/test_sample_256/resized_256/nir

  # Both models:
  for cfg in student_08_distill_downstream student_hailo_30fps; do
    python scripts/export_litert.py \\
        --config experiments/$cfg/config.yaml \\
        --calib_dir data/test_sample_256/resized_256/nir
  done
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INPUT_SIZE   = 256
CALIB_LIMIT  = 100
# Quantize Conv + elementwise ops to int8; skip LayerNorm ops (SQRT/DIV/SUB/MEAN)
# which have no int8 TFLite kernel.
INT8_OPS = [
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "MUL",
    "ADD",
    "CONCATENATION",
    "SPLIT",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config_path: Path):
    import torch
    from src.utils.config import load_config
    from src.models import build_generator

    cfg, _ = load_config(str(config_path))
    G = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels)

    run_dir = ROOT / cfg.project.output_dir / cfg.project.experiment
    ckpt = run_dir / "checkpoints" / "best.pth"
    if not ckpt.exists():
        ckpt = run_dir / "checkpoints" / "latest.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint in {run_dir}/checkpoints/")

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    G.load_state_dict(state["G"])
    G.eval()
    print(f"  Loaded {ckpt.name}  (epoch={state.get('epoch')})")
    return G, cfg, run_dir


# ---------------------------------------------------------------------------
# Calibration images (NCHW float32 [-1, 1])
# ---------------------------------------------------------------------------

def load_calib(calib_dir: Path, limit: int = CALIB_LIMIT) -> list:
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in sorted(calib_dir.iterdir()) if p.suffix.lower() in exts][:limit]
    imgs = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        arr = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[np.newaxis]
        imgs.append(arr)
    return imgs


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_fp32(G, run_dir: Path) -> Path:
    import torch
    import litert_torch as lt

    out_path = run_dir / "model_litert_fp32.tflite"
    dummy = (torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE),)
    print("  Converting to TFLite fp32 via litert-torch ...")
    lt.convert(G, dummy).export(str(out_path))
    print(f"  Saved {out_path.name}  ({out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


def export_int8(fp32_path: Path, calib_imgs: list, run_dir: Path) -> Path:
    import ai_edge_quantizer as aeq
    import ai_edge_quantizer.qtyping as qt

    out_path = run_dir / "model_litert_int8.tflite"
    fp32_bytes = fp32_path.read_bytes()

    # Discover the input tensor name from the fp32 model
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_content=fp32_bytes)
    interp.allocate_tensors()
    inp_name_full = interp.get_input_details()[0]["name"]   # e.g. "serving_default_args_0"
    # ai_edge_quantizer expects the key WITHOUT the "serving_default_" prefix
    inp_key = inp_name_full.removeprefix("serving_default_")

    calib_data = {"serving_default": [{inp_key: img} for img in calib_imgs]}

    quantizer = aeq.Quantizer(fp32_bytes)
    for op_name in INT8_OPS:
        op = qt.TFLOperationName[op_name]
        quantizer.add_static_config(".*", op,
                                    activation_num_bits=8,
                                    weight_num_bits=8,
                                    weight_granularity=qt.QuantGranularity.CHANNELWISE)

    print(f"  Calibrating ({len(calib_imgs)} images) ...")
    calib_result = quantizer.calibrate(calib_data)
    result = quantizer.quantize(calib_result)
    out_path.write_bytes(result.quantized_model)
    print(f"  Saved {out_path.name}  ({out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(tflite_path: Path, num_threads: int = 4):
    import tensorflow as tf

    interp = tf.lite.Interpreter(str(tflite_path), num_threads=num_threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    arr = np.random.randn(*inp["shape"]).astype(inp["dtype"])
    interp.set_tensor(inp["index"], arr)
    interp.invoke()

    # Validate no NCHW-style broadcast shapes remain
    bad = [t["name"] for t in interp.get_tensor_details()
           if len(t["shape"]) == 4 and list(t["shape"]) == [1, 1, INPUT_SIZE, INPUT_SIZE]]
    if bad:
        print(f"  WARNING: {len(bad)} NCHW broadcast tensor(s) found — layout may be wrong:")
        for name in bad[:3]:
            print(f"    {name}")
    else:
        print(f"  XNNPACK smoke test: PASSED  (no NCHW broadcast tensors)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Path to experiment config YAML (e.g. experiments/student_08_distill_downstream/config.yaml)")
    ap.add_argument("--calib_dir", required=True,
                    help="Directory of 256×256 NIR calibration images")
    ap.add_argument("--calib_limit", type=int, default=CALIB_LIMIT)
    ap.add_argument("--no_int8", action="store_true", help="Skip int8 export")
    ap.add_argument("--no_smoke", action="store_true", help="Skip smoke test")
    args = ap.parse_args()

    config_path = Path(args.config)
    calib_dir   = Path(args.calib_dir)

    if not config_path.exists():
        ap.error(f"Config not found: {config_path}")
    if not calib_dir.is_dir():
        ap.error(f"Calibration dir not found: {calib_dir}")

    print(f"\nConfig     : {config_path}")
    print(f"Calib dir  : {calib_dir}  ({args.calib_limit} images max)")

    # Load
    print("\n[1/4] Loading model ...")
    G, cfg, run_dir = load_model(config_path)

    # Calibration images
    print(f"\n[2/4] Loading calibration images ...")
    calib_imgs = load_calib(calib_dir, limit=args.calib_limit)
    print(f"  {len(calib_imgs)} images loaded")

    # FP32
    print(f"\n[3/4] Exporting fp32 TFLite ...")
    fp32_path = export_fp32(G, run_dir)
    if not args.no_smoke:
        smoke_test(fp32_path)

    # INT8
    if not args.no_int8:
        print(f"\n[4/4] Exporting int8 TFLite (Conv/Mul/Add quantized, LayerNorm stays fp32) ...")
        int8_path = export_int8(fp32_path, calib_imgs, run_dir)
        if not args.no_smoke:
            smoke_test(int8_path)
    else:
        print("\n[4/4] Skipping int8 (--no_int8)")

    print(f"\nOutputs in {run_dir}/:")
    for p in [fp32_path, run_dir / "model_litert_int8.tflite"]:
        if p.exists():
            print(f"  {p.name}  ({p.stat().st_size/1e6:.1f} MB)")

    print("\nDownload to Pi:")
    print(f"  model_litert_int8.tflite  ← primary target")
    print(f"  model_litert_fp32.tflite  ← fp32 baseline")
    print(f"\nPi deps:   pip install tflite-runtime")
    print(f"Benchmark: python3 benchmark_pi_cpu.py --models_dir .")


if __name__ == "__main__":
    main()
