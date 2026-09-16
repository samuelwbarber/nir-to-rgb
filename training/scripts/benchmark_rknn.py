"""On-device latency benchmark for an RKNN model.

Run this on the Radxa Rock 5C with rknn-toolkit-lite2 installed:
    pip install rknn-toolkit-lite2

This measures real NPU inference latency (no host PC; no toolkit2).
"""

import argparse
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rknn", type=str, required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--core", type=str, default="auto",
                   choices=["auto", "0", "1", "2", "0_1", "0_1_2"])
    args = p.parse_args()

    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("rknn-toolkit-lite2 not installed. run this on the Rock 5C.")
        sys.exit(1)

    rknn = RKNNLite()
    if rknn.load_rknn(args.rknn) != 0:
        print("load_rknn failed"); sys.exit(1)

    core_map = {
        "auto": RKNNLite.NPU_CORE_AUTO,
        "0": RKNNLite.NPU_CORE_0,
        "1": RKNNLite.NPU_CORE_1,
        "2": RKNNLite.NPU_CORE_2,
        "0_1": RKNNLite.NPU_CORE_0_1,
        "0_1_2": RKNNLite.NPU_CORE_0_1_2,
    }
    if rknn.init_runtime(core_mask=core_map[args.core]) != 0:
        print("init_runtime failed"); sys.exit(1)

    dummy = np.random.randint(0, 256, (args.size, args.size, args.channels), dtype=np.uint8)

    for _ in range(args.warmup):
        rknn.inference(inputs=[dummy])

    t0 = time.perf_counter()
    for _ in range(args.iters):
        rknn.inference(inputs=[dummy])
    elapsed = time.perf_counter() - t0

    ms = (elapsed / args.iters) * 1000
    fps = args.iters / elapsed
    print(f"size={args.size}x{args.size} core={args.core}")
    print(f"  per-iter: {ms:.2f} ms")
    print(f"  fps:      {fps:.1f}")

    rknn.release()


if __name__ == "__main__":
    main()
