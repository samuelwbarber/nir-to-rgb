"""Export NAFNet checkpoint to ONNX + optional INT8 quantization + RPi5 benchmark.

Export mode (run on training machine with GPU):
    python scripts/export_onnx.py --config configs/student_08_distill_downstream.yaml

Benchmark-only mode (run on Raspberry Pi 5 — no GPU needed):
    python scripts/export_onnx.py --onnx experiments/student_08_distill_downstream/model.onnx
    python scripts/export_onnx.py --onnx experiments/student_08_distill_downstream/model_q.onnx

Outputs written to experiments/<experiment>/:
    model.onnx    — float32, opset 17
    model_q.onnx  — INT8 dynamically quantized (~4x smaller, faster on ARM NEON)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export(config_path: str) -> Path:
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
    print(f"loaded {ckpt} (epoch={state.get('epoch')})")

    params = sum(p.numel() for p in G.parameters())
    print(f"parameters: {params:,} ({params/1e6:.2f}M)")

    try:
        from thop import profile as thop_profile
        dummy_thop = torch.zeros(1, cfg.data.in_channels, cfg.data.image_size, cfg.data.image_size)
        flops, _ = thop_profile(G, inputs=(dummy_thop,), verbose=False)
        print(f"FLOPs @ {cfg.data.image_size}x{cfg.data.image_size}: {flops/1e9:.2f} GFLOPs")
    except ImportError:
        print("thop not installed — skipping FLOPs (pip install thop)")

    run_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = run_dir / "model.onnx"
    size = cfg.data.image_size
    dummy = torch.zeros(1, cfg.data.in_channels, size, size)
    torch.onnx.export(
        G, dummy, str(onnx_path),
        opset_version=17,
        input_names=["nir"],
        output_names=["rgb"],
        dynamic_axes={"nir": {0: "batch"}, "rgb": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"exported: {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q_path = run_dir / "model_q.onnx"
        quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QUInt8)
        print(f"quantized: {q_path} ({q_path.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"quantization skipped: {e}")

    return onnx_path


def benchmark(onnx_path: str, image_size: int = 256, n_threads: int = 4,
              n_warmup: int = 5, n_runs: int = 30):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = 1

    try:
        sess = ort.InferenceSession(
            onnx_path, sess_options=opts,
            providers=["XNNPACKExecutionProvider", "CPUExecutionProvider"],
        )
        print("provider: XNNPACKExecutionProvider (ARM NEON optimised)")
    except Exception:
        sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        print("provider: CPUExecutionProvider")

    dummy = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
    in_name = sess.get_inputs()[0].name

    for _ in range(n_warmup):
        sess.run(None, {in_name: dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {in_name: dummy})
        times.append(time.perf_counter() - t0)

    t_ms = np.array(times) * 1000
    fps = 1000.0 / np.mean(t_ms)
    print(f"  {Path(onnx_path).name} @ {image_size}x{image_size}  threads={n_threads}")
    print(f"  {np.mean(t_ms):.1f} ± {np.std(t_ms):.1f} ms  →  {fps:.1f} fps")
    print(f"  min={t_ms.min():.1f}  p50={np.percentile(t_ms,50):.1f}  p95={np.percentile(t_ms,95):.1f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config (export mode — needs GPU + trained checkpoint)")
    ap.add_argument("--onnx", help="Existing .onnx path (benchmark-only, no GPU needed)")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (4 = all RPi5 cores)")
    args = ap.parse_args()

    if args.config:
        onnx_path = export(args.config)
        print("\n--- float32 benchmark ---")
        benchmark(str(onnx_path), image_size=args.image_size, n_threads=args.threads)
        q_path = Path(str(onnx_path).replace(".onnx", "_q.onnx"))
        if q_path.exists():
            print("\n--- INT8 quantized benchmark ---")
            benchmark(str(q_path), image_size=args.image_size, n_threads=args.threads)
    elif args.onnx:
        benchmark(args.onnx, image_size=args.image_size, n_threads=args.threads)
    else:
        ap.error("provide --config (export) or --onnx (benchmark only)")
