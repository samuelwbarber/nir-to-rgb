#!/usr/bin/env python3
"""
Maximum-throughput CPU benchmark for all distilled models (ONNX + TFLite).
Designed for Raspberry Pi without Hailo hat.

Dependencies:
  pip install onnxruntime numpy opencv-python   # always needed
  pip install tflite-runtime                    # for .tflite models (lightweight)

Usage:
    python3 benchmark_pi_cpu.py
    python3 benchmark_pi_cpu.py --models_dir /path/to/models --images_dir /path/to/nir

Expected layout:
    student_08_distill_downstream/
        model.onnx  model_q.onnx  model_sq.onnx
        model_tflite_fp32.tflite  model_tflite_int8.tflite
    student_hailo_30fps/
        model.onnx  model_q.onnx  model_sq.onnx  model_static.onnx
        model_tflite_fp32.tflite  model_tflite_int8.tflite
    test_sample_256/resized_256/nir/   ← optional; auto-detected
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

WARMUP = 30
RUNS = 200
INPUT_SIZE = 256

STUDENT_DIRS = [
    "student_08_distill_downstream",
    "student_hailo_30fps",
]


# ---------------------------------------------------------------------------
# Image loading — two formats needed
# ---------------------------------------------------------------------------

def _read_img_nhwc(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return (img.astype(np.float32) / 127.5 - 1.0)[np.newaxis]   # (1, H, W, C)


def load_images(images_dir: Path, limit: int = 100):
    """Returns (nchw_list, nhwc_list) for ONNX and TFLite respectively."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts][:limit]
    nchw, nhwc = [], []
    for p in paths:
        arr = _read_img_nhwc(p)
        if arr is None:
            continue
        nhwc.append(arr)
        nchw.append(arr.transpose(0, 3, 1, 2))   # (1, C, H, W)
    return nchw, nhwc


def make_dummy(n: int = 50):
    rng = np.random.default_rng(0)
    nhwc = [rng.uniform(-1, 1, (1, INPUT_SIZE, INPUT_SIZE, 3)).astype(np.float32)
            for _ in range(n)]
    nchw = [x.transpose(0, 3, 1, 2) for x in nhwc]
    return nchw, nhwc


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def find_models(root: Path) -> list[Path]:
    found = []
    for d in STUDENT_DIRS:
        sub = root / d
        if not sub.is_dir():
            sub = root
        for ext in ("*.onnx", "*.tflite"):
            for p in sorted(sub.glob(ext)):
                if p not in found:
                    found.append(p)
        if sub == root:
            break
    seen, unique = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# ONNX benchmark
# ---------------------------------------------------------------------------

def benchmark_onnx(model_path: Path, images_nchw: list, num_threads: int,
                   warmup: int, runs: int) -> dict | None:
    import onnxruntime as ort

    label = f"{model_path.parent.name}/{model_path.name}"
    size_mb = sum(p.stat().st_size for p in model_path.parent.glob(f"{model_path.name}*")) / 1e6

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    print(f"\n  Loading {label}  ({size_mb:.1f} MB) ...", flush=True)
    try:
        sess = ort.InferenceSession(str(model_path), sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return None

    inp_name  = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    print(f"  Input : {inp_name} {inp_shape}")

    for i in range(warmup):
        sess.run(None, {inp_name: images_nchw[i % len(images_nchw)]})

    latencies = []
    for i in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: images_nchw[i % len(images_nchw)]})
        latencies.append(time.perf_counter() - t0)

    lat = np.array(latencies) * 1000
    fps = 1000.0 / lat.mean()
    print(f"  Latency : {lat.mean():.1f} ms  "
          f"(min {lat.min():.1f}  p50 {np.median(lat):.1f}  "
          f"p95 {np.percentile(lat, 95):.1f}  max {lat.max():.1f})")
    print(f"  FPS     : {fps:.2f}")
    return {"label": label, "fps": fps, "lat_mean": lat.mean(),
            "lat_p95": np.percentile(lat, 95), "size_mb": size_mb}


# ---------------------------------------------------------------------------
# TFLite benchmark
# ---------------------------------------------------------------------------

def benchmark_tflite(model_path: Path, images_nhwc: list, num_threads: int,
                     warmup: int, runs: int) -> dict | None:
    try:
        import tflite_runtime.interpreter as _tfl
        Interpreter = _tfl.Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            print(f"\n  SKIP {model_path.name}: install tflite-runtime to benchmark .tflite models")
            print(  "        pip install tflite-runtime")
            return None

    label    = f"{model_path.parent.name}/{model_path.name}"
    size_mb  = model_path.stat().st_size / 1e6

    print(f"\n  Loading {label}  ({size_mb:.1f} MB) ...", flush=True)
    try:
        interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
        interp.allocate_tensors()
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return None

    inp = interp.get_input_details()[0]
    print(f"  Input : {inp['name']} {inp['shape']}  dtype={inp['dtype'].__name__}")

    for i in range(warmup):
        interp.set_tensor(inp["index"], images_nhwc[i % len(images_nhwc)])
        interp.invoke()

    latencies = []
    for i in range(runs):
        interp.set_tensor(inp["index"], images_nhwc[i % len(images_nhwc)])
        t0 = time.perf_counter()
        interp.invoke()
        latencies.append(time.perf_counter() - t0)

    lat = np.array(latencies) * 1000
    fps = 1000.0 / lat.mean()
    print(f"  Latency : {lat.mean():.1f} ms  "
          f"(min {lat.min():.1f}  p50 {np.median(lat):.1f}  "
          f"p95 {np.percentile(lat, 95):.1f}  max {lat.max():.1f})")
    print(f"  FPS     : {fps:.2f}")
    return {"label": label, "fps": fps, "lat_mean": lat.mean(),
            "lat_p95": np.percentile(lat, 95), "size_mb": size_mb}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark all distilled models on CPU (Raspberry Pi)")
    ap.add_argument("--models_dir", default=".")
    ap.add_argument("--images_dir", default=None,
                    help="256×256 NIR images directory; auto-detected if omitted")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()

    warmup, runs = args.warmup, args.runs

    # Images
    if args.images_dir:
        images_dir = Path(args.images_dir)
        if not images_dir.is_dir():
            ap.error(f"--images_dir not found: {images_dir}")
        nchw, nhwc = load_images(images_dir)
        print(f"Loaded {len(nchw)} images from {images_dir}")
    else:
        default = Path(args.models_dir) / "test_sample_256" / "resized_256" / "nir"
        if default.is_dir():
            nchw, nhwc = load_images(default)
            print(f"Loaded {len(nchw)} images from {default}")
        else:
            nchw, nhwc = make_dummy()
            print("No images_dir found — using random noise.")

    if not nchw:
        nchw, nhwc = make_dummy()
        print("Warning: no images loaded, falling back to random noise.")

    # Models
    models_dir  = Path(args.models_dir)
    model_paths = find_models(models_dir)
    if not model_paths:
        ap.error(f"No .onnx or .tflite files found under {models_dir}")

    print(f"\nPlatform : {os.uname().nodename}  ({os.cpu_count()} cores)")
    print(f"Threads  : {args.threads}   Warmup: {warmup}   Runs: {runs}")
    print(f"\nModels found ({len(model_paths)}):")
    for p in model_paths:
        mb = sum(f.stat().st_size for f in p.parent.glob(f"{p.stem}*")) / 1e6
        print(f"  {p.parent.name}/{p.name}  ({mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("BENCHMARKING")
    print("=" * 60)

    results = []
    for mp in model_paths:
        if mp.suffix == ".tflite":
            r = benchmark_tflite(mp, nhwc, args.threads, warmup, runs)
        else:
            r = benchmark_onnx(mp, nchw, args.threads, warmup, runs)
        if r:
            results.append(r)

    if not results:
        print("No models ran successfully.")
        return

    results.sort(key=lambda x: -x["fps"])

    print("\n" + "=" * 60)
    print("SUMMARY  (sorted by FPS, highest first)")
    print("=" * 60)
    print(f"  {'Model':<52}  {'FPS':>7}  {'ms/frame':>9}  {'p95 ms':>8}  {'MB':>6}")
    print(f"  {'-'*52}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*6}")
    for r in results:
        print(f"  {r['label']:<52}  {r['fps']:>7.2f}  "
              f"{r['lat_mean']:>9.1f}  {r['lat_p95']:>8.1f}  {r['size_mb']:>6.1f}")
    print()


if __name__ == "__main__":
    main()
