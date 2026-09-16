"""
Export a float32 ONNX model to static INT8 QDQ quantization using real calibration images.

Fixes the problem with model_q.onnx (dynamic weight-only quantization):
  - Dynamic quant: weights stored as uint8, activations dequantized to fp32 at runtime
                   → no fast int8 kernel, can be SLOWER than fp32
  - Static QDQ:    both weights + activations calibrated and kept int8 end-to-end
                   → hits NEON int8 kernels on ARM, should be faster on Pi

Usage (run on server — no GPU needed, only the exported model.onnx):
    # Single model:
    python scripts/export_static_quant.py \\
        --model experiments/student_hailo_30fps/model.onnx \\
        --calib_dir data/test_sample_256/resized_256/nir

    # Both student models:
    for d in student_hailo_30fps student_08_distill_downstream; do
        python scripts/export_static_quant.py \\
            --model experiments/$d/model.onnx \\
            --calib_dir data/test_sample_256/resized_256/nir
    done

Outputs (alongside the input model.onnx):
    model_sq.onnx     — static int8 QDQ (download this to the Pi)
    model_pre.onnx    — pre-processed intermediate (safe to delete)

Note: the benchmark shown here runs on x86 where QDQ int8 is typically SLOWER than fp32.
      The speedup only appears on ARM (Pi). Use benchmark_pi_cpu.py on the Pi.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Suppress noisy "unsupported type to quantize" warnings for int64 shape tensors
logging.getLogger("root").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INPUT_SIZE = 256
CALIB_IMAGES = 100  # images to use for calibration


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------

class NIRCalibrationReader:
    """Feeds preprocessed NIR images to the ORT calibration engine."""

    def __init__(self, calib_dir: Path, input_name: str, limit: int = CALIB_IMAGES):
        exts = {".jpg", ".jpeg", ".png"}
        paths = [p for p in sorted(calib_dir.iterdir()) if p.suffix.lower() in exts]
        if not paths:
            raise FileNotFoundError(f"No images found in {calib_dir}")
        self.paths = paths[:limit]
        self.input_name = input_name
        self._idx = 0

    def get_next(self) -> dict | None:
        if self._idx >= len(self.paths):
            return None
        p = self.paths[self._idx]
        self._idx += 1
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        arr = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[np.newaxis]
        return {self.input_name: arr}

    def rewind(self):
        self._idx = 0


# ---------------------------------------------------------------------------
# Quick ORT benchmark
# ---------------------------------------------------------------------------

def benchmark(model_path: Path, calib_dir: Path, n_warmup: int = 10, n_runs: int = 50) -> float:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(str(model_path), sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"    {model_path.name:35s}  FAILED to load: {e}")
        return 0.0
    inp = sess.get_inputs()[0].name

    exts = {".jpg", ".jpeg", ".png"}
    imgs = []
    for p in sorted(calib_dir.iterdir()):
        if p.suffix.lower() in exts:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            arr = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[np.newaxis]
            imgs.append(arr)
            if len(imgs) >= 50:
                break

    for i in range(n_warmup):
        sess.run(None, {inp: imgs[i % len(imgs)]})

    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {inp: imgs[i % len(imgs)]})
        times.append(time.perf_counter() - t0)

    lat = np.array(times) * 1000
    fps = 1000.0 / lat.mean()
    print(f"    {model_path.name:35s}  {lat.mean():6.1f} ms  {fps:5.1f} fps")
    return fps


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(model_path: Path, calib_dir: Path, calib_limit: int = CALIB_IMAGES) -> Path:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.preprocess import quant_pre_process
    import onnxruntime as ort

    out_dir = model_path.parent
    pre_path = out_dir / "model_pre.onnx"
    sq_path  = out_dir / "model_sq.onnx"

    # Get input name from the fp32 model
    sess_fp32 = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name
    del sess_fp32

    # Step 1: pre-process (op fusion, node merging — required for clean QDQ insertion)
    print("  Pre-processing model...")
    quant_pre_process(str(model_path), str(pre_path),
                      skip_optimization=False, skip_symbolic_shape=True)
    print(f"  Saved: {pre_path.name}  ({pre_path.stat().st_size/1e6:.1f} MB)")

    n_calib = min(calib_limit, sum(1 for p in calib_dir.iterdir()
                                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}))
    print(f"\n  Calibrating with {n_calib} images from {calib_dir} ...")

    # Exclude DepthToSpace (pixel_shuffle) nodes AND the Conv nodes that feed directly
    # into them. ORT's graph optimizer fuses away the DQ boundary nodes before
    # DepthToSpace, leaving it with int8 inputs where no int8 kernel exists (opset 17).
    # Excluding the feeder Conv nodes prevents a Q node being placed on their outputs,
    # so DepthToSpace receives float32 end-to-end even after optimization.
    import onnx as _onnx
    _m = _onnx.load(str(model_path))
    _tensor_to_node = {out: n.name for n in _m.graph.node for out in n.output}
    nodes_to_exclude = []
    for n in _m.graph.node:
        if n.op_type == "DepthToSpace":
            nodes_to_exclude.append(n.name)
            feeder = _tensor_to_node.get(n.input[0])
            if feeder:
                nodes_to_exclude.append(feeder)
    print(f"  Excluding {len(nodes_to_exclude)} nodes from int8 "
          f"(DepthToSpace + feeder Convs): {nodes_to_exclude}")

    # Step 2: static QDQ — int8 weights + int8 activations, per-channel weights
    quantize_static(
        model_input=str(pre_path),
        model_output=str(sq_path),
        calibration_data_reader=NIRCalibrationReader(calib_dir, input_name, limit=calib_limit),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=nodes_to_exclude,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    print(f"  Saved: {sq_path.name}  ({sq_path.stat().st_size/1e6:.1f} MB)")
    return sq_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to float32 model.onnx")
    ap.add_argument("--calib_dir", required=True,
                    help="Directory of 256x256 NIR calibration images")
    ap.add_argument("--calib_limit", type=int, default=CALIB_IMAGES,
                    help=f"Max calibration images (default: {CALIB_IMAGES})")
    ap.add_argument("--no_benchmark", action="store_true",
                    help="Skip the server-side benchmark after export")
    args = ap.parse_args()

    model_path = Path(args.model)
    calib_dir  = Path(args.calib_dir)

    if not model_path.exists():
        ap.error(f"Model not found: {model_path}")
    if not calib_dir.is_dir():
        ap.error(f"Calibration dir not found: {calib_dir}")

    fp32_mb = sum(p.stat().st_size for p in model_path.parent.glob("model.onnx*")) / 1e6
    print(f"\nExperiment : {model_path.parent.name}")
    print(f"FP32 model : {model_path.name}  ({fp32_mb:.1f} MB)")
    print(f"Calib data : {calib_dir}")

    sq_path = export(model_path, calib_dir, calib_limit=args.calib_limit)

    if not args.no_benchmark:
        print(f"\n  NOTE: QDQ int8 is often SLOWER on x86 — speedup only shows on ARM (Pi).")
        print(f"  {'Model':35s}  {'ms/frame':>8}  {'FPS':>5}")
        print(f"  {'-'*35}  {'-'*8}  {'-'*5}")
        benchmark(model_path, calib_dir)
        benchmark(sq_path, calib_dir)

    print(f"\nDone. Output: {sq_path}")
    print("Download model_sq.onnx to the Pi and run:")
    print("  python3 benchmark_pi_cpu.py --models_dir .")


if __name__ == "__main__":
    main()
