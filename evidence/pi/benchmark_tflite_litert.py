#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

INPUT_SIZE = 256
STUDENT_DIRS = ["student_08_distill_downstream", "student_hailo_30fps"]


def read_img(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return (img.astype(np.float32) / 127.5 - 1.0)[None]


def load_images(images_dir: Path, limit: int = 100):
    paths = [
        p for p in sorted(images_dir.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ][:limit]
    images = []
    for path in paths:
        img = read_img(path)
        if img is not None:
            images.append(img)
    return images


def find_models(root: Path):
    found = []
    for dirname in STUDENT_DIRS:
        sub = root / dirname
        if not sub.is_dir():
            sub = root
        found.extend(sorted(sub.glob("*.tflite")))
        if sub == root:
            break
    return found


def prepare_input(float_nhwc, input_detail):
    shape = list(input_detail["shape"])
    if len(shape) == 4 and shape[1] == 3:
        float_input = float_nhwc.transpose(0, 3, 1, 2)
    else:
        float_input = float_nhwc

    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return float_input.astype(np.float32, copy=False)

    scale, zero = input_detail.get("quantization", (0.0, 0))
    if not scale:
        scale, zero = input_detail.get("quantization_parameters", {}).get("scales", [0.0])[0], 0
    if not scale:
        raise ValueError(f"Quantized input has no usable scale: {input_detail}")

    q = np.round(float_input / scale + zero)
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def benchmark(model_path: Path, images, threads: int, warmup: int, runs: int, disable_delegates: bool):
    label = f"{model_path.parent.name}/{model_path.name}"
    print(f"\n  Loading {label} ({model_path.stat().st_size / 1e6:.1f} MB) ...", flush=True)

    kwargs = {"model_path": str(model_path), "num_threads": threads}
    if disable_delegates:
        kwargs["experimental_op_resolver_type"] = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    interpreter = Interpreter(**kwargs)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    print(f"  Input : {inp['name']} {inp['shape']} dtype={inp['dtype'].__name__} quant={inp.get('quantization')}")
    prepared = [prepare_input(img, inp) for img in images]

    for i in range(warmup):
        interpreter.set_tensor(inp["index"], prepared[i % len(prepared)])
        interpreter.invoke()

    latencies = []
    for i in range(runs):
        interpreter.set_tensor(inp["index"], prepared[i % len(prepared)])
        start = time.perf_counter()
        interpreter.invoke()
        latencies.append(time.perf_counter() - start)

    lat = np.array(latencies) * 1000.0
    fps = 1000.0 / lat.mean()
    print(
        f"  Latency : {lat.mean():.1f} ms "
        f"(min {lat.min():.1f} p50 {np.median(lat):.1f} "
        f"p95 {np.percentile(lat, 95):.1f} max {lat.max():.1f})"
    )
    print(f"  FPS     : {fps:.2f}")
    return {
        "label": label,
        "fps": fps,
        "lat_mean": lat.mean(),
        "lat_p95": np.percentile(lat, 95),
        "size_mb": model_path.stat().st_size / 1e6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="/home/pi/tflite_bench")
    parser.add_argument("--images_dir", default="/home/pi/bench/test_sample_256/resized_256/nir")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--disable-delegates", action="store_true")
    args = parser.parse_args()

    images = load_images(Path(args.images_dir))
    if not images:
        raise SystemExit(f"No images found in {args.images_dir}")
    models = find_models(Path(args.models_dir))
    if not models:
        raise SystemExit(f"No TFLite models found in {args.models_dir}")

    print(f"Loaded {len(images)} images from {args.images_dir}")
    print(f"Platform : {os.uname().nodename} ({os.cpu_count()} cores)")
    print(f"Threads  : {args.threads} Warmup: {args.warmup} Runs: {args.runs}")
    print("Models:")
    for model in models:
        print(f"  {model.parent.name}/{model.name} ({model.stat().st_size / 1e6:.1f} MB)")

    results = []
    for model in models:
        try:
            results.append(benchmark(model, images, args.threads, args.warmup, args.runs, args.disable_delegates))
        except Exception as exc:
            print(f"\n  FAILED {model.parent.name}/{model.name}: {exc}")

    if results:
        results.sort(key=lambda item: -item["fps"])
        print("\nSUMMARY")
        print(f"{'Model':<68} {'FPS':>7} {'ms/frame':>9} {'p95 ms':>8} {'MB':>6}")
        for result in results:
            print(
                f"{result['label']:<68} {result['fps']:>7.2f} "
                f"{result['lat_mean']:>9.1f} {result['lat_p95']:>8.1f} {result['size_mb']:>6.1f}"
            )


if __name__ == "__main__":
    main()
