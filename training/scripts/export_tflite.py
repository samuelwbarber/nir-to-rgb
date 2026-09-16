"""
Export float32 ONNX model to TFLite with full int8 quantization via onnx2tf.

Why TFLite over ONNX static quant:
  ONNX Runtime on ARM (pip install) does not include XNNPACK, so ORT int8 QDQ
  runs at the same speed as fp32. TFLite ships with XNNPACK compiled in and
  explicitly targets ARM int8 NEON — this is where the real speedup comes from.

Conversion pipeline:
  model.onnx
    → onnx2tf (flatbuffer_direct)
    → model_float32.tflite     (baseline)
    → model_tflite_int8.tflite (full int8, float32 I/O, calibrated)

Dependencies (server-side only):
  pip install onnx2tf tensorflow

Pi-side inference only:
  pip install tflite-runtime   (much lighter than full tensorflow)

Usage:
  python scripts/export_tflite.py \\
      --model experiments/student_08_distill_downstream/model.onnx \\
      --calib_dir data/test_sample_256/resized_256/nir

  # Both models:
  for d in student_08_distill_downstream student_hailo_30fps; do
    python scripts/export_tflite.py \\
        --model experiments/$d/model.onnx \\
        --calib_dir data/test_sample_256/resized_256/nir
  done

Outputs (alongside model.onnx):
  model_tflite_fp32.tflite    — float32 baseline (copy of onnx2tf float32 output)
  model_tflite_int8.tflite    — full int8, float32 I/O  ← use this on the Pi
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_SIZE = 256
CALIB_IMAGES = 100

# onnx2tf normalizes input as: model_input = (calib_data_0to1 - mean) / std
# Our model: float/127.5 - 1.0 = (float/255.0 - 0.5) / 0.5
# → calib data in [0,1], mean=0.5, std=0.5 per channel (NHWC broadcast shape)
CALIB_MEAN = np.array([[[[0.5, 0.5, 0.5]]]], dtype=np.float32)  # (1,1,1,3)
CALIB_STD  = np.array([[[[0.5, 0.5, 0.5]]]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Build calibration numpy file  (N, H, W, C)  float32  values in [0, 1]
# ---------------------------------------------------------------------------

def save_calib_npy(calib_dir: Path, out_npy: Path, limit: int = CALIB_IMAGES):
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in sorted(calib_dir.iterdir()) if p.suffix.lower() in exts][:limit]
    if not paths:
        raise FileNotFoundError(f"No images in {calib_dir}")
    imgs = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        imgs.append(img.astype(np.float32) / 255.0)   # NHWC, 0–1
    arr = np.stack(imgs, axis=0)   # (N, H, W, C)
    np.save(str(out_npy), arr)
    print(f"  Saved calibration data: {out_npy.name}  shape={arr.shape}")
    return out_npy


# ---------------------------------------------------------------------------
# onnx2tf conversion
# ---------------------------------------------------------------------------

def convert(onnx_path: Path, saved_model_dir: Path, calib_npy: Path,
            input_op_name: str = "nir"):
    import onnx2tf

    print(f"  Converting {onnx_path.name} → TFLite (fp32 + int8) ...")
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        batch_size=1,
        non_verbose=True,
        output_integer_quantized_tflite=True,
        input_quant_dtype="float32",   # keep I/O as float32 for ease of use
        output_quant_dtype="float32",
        custom_input_op_name_np_data_path=[
            [input_op_name, str(calib_npy), CALIB_MEAN, CALIB_STD],
        ],
    )


def get_input_op_name(tflite_path: Path) -> str:
    """Read input op name from an existing tflite flatbuffer."""
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(str(tflite_path))
        interp.allocate_tensors()
        return interp.get_input_details()[0]["name"]
    except Exception:
        return "nir"   # fallback — matches our model's ONNX input name


# ---------------------------------------------------------------------------
# Post-processing: patch bad NCHW→NHWC transposes on ReduceMean broadcast tensors
#
# Root cause: onnx2tf converts ReduceMean outputs (shape [1,1,H,W]) to NHWC
# correctly in one pass, but then a second NCHW→NHWC transpose pass applies
# [0,2,3,1] to the already-NHWC [1,H,W,1] tensor, producing [1,H,1,W] instead.
# The fix (identical to the user's manual patch): replace those perm constants
# with identity [0,1,2,3] so the second transpose becomes a no-op.
#
# Uses schema_generated.py (emitted by onnx2tf into the saved_model dir) +
# flatbuffers to deserialise, patch, and re-serialise cleanly.
# ---------------------------------------------------------------------------

BAD_PERM  = [0, 2, 3, 1]
GOOD_PERM = [0, 1, 2, 3]


def _load_schema(saved_model_dir: Path):
    """Import ModelT from onnx2tf's schema_generated.py."""
    import importlib.util, sys
    schema_path = saved_model_dir / "schema_generated.py"
    if not schema_path.exists():
        raise FileNotFoundError(f"schema_generated.py not found in {saved_model_dir}")
    spec = importlib.util.spec_from_file_location("tflite_schema", schema_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ModelT


def patch_mean_transposes(tflite_path: Path, saved_model_dir: Path) -> int:
    """
    Deserialise the tflite flatbuffer, find perm tensors on ReduceMean broadcast
    paths that contain [0,2,3,1], replace with [0,1,2,3], re-serialise in place.
    Returns the number of tensors patched.
    """
    import flatbuffers

    ModelT = _load_schema(saved_model_dir)

    raw = bytearray(tflite_path.read_bytes())
    model = ModelT.InitFromPackedBuf(raw, 0)

    n_patched = 0
    for subgraph in (model.subgraphs or []):
        for tensor in (subgraph.tensors or []):
            name = tensor.name.decode() if isinstance(tensor.name, (bytes, bytearray)) else (tensor.name or "")
            # Target: perm tensors added by onnx2tf for NHWC conversion of mean tensors
            if "_to_nhwc_perm" not in name:
                continue
            buf = model.buffers[tensor.buffer]
            if buf.data is None:
                continue
            perm = list(np.frombuffer(bytes(buf.data), dtype=np.int32))
            if perm == BAD_PERM:
                buf.data = np.array(GOOD_PERM, dtype=np.int32).tobytes()
                print(f"    Patched '{name}': {BAD_PERM} → {GOOD_PERM}")
                n_patched += 1

    if n_patched == 0:
        return 0

    # Re-serialise
    builder = flatbuffers.Builder(len(raw) + 1024)
    root = model.Pack(builder)
    builder.Finish(root)
    patched = bytes(builder.Output())

    # TFLite files require the 'TFL3' file identifier at bytes 4-8
    # flatbuffers.Builder.Finish() does not add it; we must re-insert
    import struct
    tfl3 = b'TFL3'
    if patched[4:8] != tfl3:
        # Insert identifier: flatbuffers file = [offset_to_root (4B)] [identifier (4B)] [data]
        root_off = struct.unpack_from('<I', patched, 0)[0]
        patched = patched[:4] + tfl3 + patched[8:]

    tflite_path.write_bytes(patched)
    return n_patched


# ---------------------------------------------------------------------------
# Benchmark (server-side sanity check only — no XNNPACK here)
# ---------------------------------------------------------------------------

def benchmark(tflite_path: Path, calib_dir: Path, n_warmup: int = 10, n_runs: int = 30):
    import time
    try:
        import tflite_runtime.interpreter as _tfl
        Interpreter = _tfl.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_path=str(tflite_path), num_threads=4)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]

    exts = {".jpg", ".jpeg", ".png"}
    imgs = []
    for p in sorted(calib_dir.iterdir()):
        if p.suffix.lower() in exts:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            # float32 [-1, 1] NHWC — matches tflite model I/O
            imgs.append((img.astype(np.float32) / 127.5 - 1.0)[np.newaxis])
            if len(imgs) >= 50:
                break

    for i in range(n_warmup):
        interp.set_tensor(inp["index"], imgs[i % len(imgs)])
        interp.invoke()

    times = []
    for i in range(n_runs):
        interp.set_tensor(inp["index"], imgs[i % len(imgs)])
        t0 = time.perf_counter()
        interp.invoke()
        times.append(time.perf_counter() - t0)

    lat = np.array(times) * 1000
    fps = 1000.0 / lat.mean()
    print(f"  {tflite_path.name:35s}  {lat.mean():6.1f} ms  {fps:5.1f} fps  "
          f"[server/no-XNNPACK — Pi will differ]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to float32 model.onnx")
    ap.add_argument("--calib_dir", required=True, help="Directory of 256x256 NIR images")
    ap.add_argument("--calib_limit", type=int, default=CALIB_IMAGES)
    ap.add_argument("--no_benchmark", action="store_true")
    args = ap.parse_args()

    model_path = Path(args.model)
    calib_dir  = Path(args.calib_dir)
    out_dir    = model_path.parent

    if not model_path.exists():
        ap.error(f"Model not found: {model_path}")
    if not calib_dir.is_dir():
        ap.error(f"Calibration dir not found: {calib_dir}")

    fp32_mb = sum(p.stat().st_size for p in out_dir.glob("model.onnx*")) / 1e6
    print(f"\nExperiment : {out_dir.name}")
    print(f"FP32 ONNX  : {model_path.name}  ({fp32_mb:.1f} MB)")
    print(f"Calib data : {calib_dir}  ({args.calib_limit} images)")

    saved_model_dir  = out_dir / "saved_model"
    calib_npy        = out_dir / "calib_data.npy"
    fp32_tflite_src  = saved_model_dir / "model_float32.tflite"
    int8_tflite_src  = saved_model_dir / "model_integer_quant.tflite"
    fp32_tflite_dst  = out_dir / "model_tflite_fp32.tflite"
    int8_tflite_dst  = out_dir / "model_tflite_int8.tflite"

    # Step 1: calibration numpy
    print("\n[1/3] Building calibration numpy array ...")
    save_calib_npy(calib_dir, calib_npy, limit=args.calib_limit)

    # Step 2: determine input op name from any pre-existing float32 tflite
    input_op = "nir"
    if fp32_tflite_src.exists():
        input_op = get_input_op_name(fp32_tflite_src)
    print(f"  Input op name: {input_op}")

    # Step 3: onnx2tf (produces both float32 and int8 tflite)
    print("\n[2/3] Running onnx2tf (fp32 + int8) ...")
    convert(model_path, saved_model_dir, calib_npy, input_op_name=input_op)

    # Step 4: copy outputs + patch bad mean transposes
    print("\n[3/3] Copying and patching outputs ...")
    if fp32_tflite_src.exists():
        shutil.copy2(fp32_tflite_src, fp32_tflite_dst)
        n = patch_mean_transposes(fp32_tflite_dst, saved_model_dir)
        print(f"  {fp32_tflite_dst.name}  ({fp32_tflite_dst.stat().st_size/1e6:.1f} MB)"
              f"  [{n} perm(s) patched]")
    else:
        print(f"  WARNING: {fp32_tflite_src} not found — fp32 tflite skipped")

    if int8_tflite_src.exists():
        shutil.copy2(int8_tflite_src, int8_tflite_dst)
    else:
        candidates = list(saved_model_dir.glob("*integer*quant*.tflite")) + \
                     list(saved_model_dir.glob("*int8*.tflite"))
        if candidates:
            shutil.copy2(candidates[0], int8_tflite_dst)
        else:
            print(f"  ERROR: could not find int8 tflite in {saved_model_dir}/")
            print(f"  Files present: {[p.name for p in saved_model_dir.glob('*.tflite')]}")

    if int8_tflite_dst.exists():
        n = patch_mean_transposes(int8_tflite_dst, saved_model_dir)
        print(f"  {int8_tflite_dst.name}  ({int8_tflite_dst.stat().st_size/1e6:.1f} MB)"
              f"  [{n} perm(s) patched]")

    if not args.no_benchmark:
        print(f"\nServer-side benchmark (no XNNPACK — Pi numbers will differ):")
        for p in [fp32_tflite_dst, int8_tflite_dst]:
            if p.exists():
                benchmark(p, calib_dir)

    print(f"\nDownload to Pi:")
    print(f"  {int8_tflite_dst.name}   ← primary target (int8 + XNNPACK on Pi)")
    print(f"  {fp32_tflite_dst.name}   ← fp32 baseline for comparison")
    print(f"\nPi deps:  pip install tflite-runtime")
    print(f"Benchmark: python3 benchmark_pi_cpu.py --models_dir .")


if __name__ == "__main__":
    main()
