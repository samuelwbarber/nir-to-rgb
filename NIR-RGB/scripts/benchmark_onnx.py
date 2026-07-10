"""Quick CPU latency baseline for an exported ONNX model.

Run anywhere; useful as a sanity-check before/after RKNN conversion.
The on-device NPU runtime will be much faster than this.
"""

import argparse
import time

import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    dummy = np.random.uniform(-1, 1, (1, args.channels, args.size, args.size)).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(None, {inp: dummy})

    t0 = time.perf_counter()
    for _ in range(args.iters):
        sess.run(None, {inp: dummy})
    elapsed = time.perf_counter() - t0

    ms = (elapsed / args.iters) * 1000
    fps = args.iters / elapsed
    print(f"size={args.size}x{args.size} provider=CPU")
    print(f"  per-iter: {ms:.2f} ms")
    print(f"  fps:      {fps:.1f}")


if __name__ == "__main__":
    main()
