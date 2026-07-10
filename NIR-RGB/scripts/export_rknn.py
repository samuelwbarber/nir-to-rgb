"""ONNX -> RKNN conversion for the Radxa Rock 5C (RK3588S NPU).

Run this on a machine with rknn-toolkit2 installed (Linux x86_64; not on the
Rock device itself). Install:
    pip install rknn-toolkit2  # see Rockchip docs for the right wheel

The calibration set should be 200-1000 representative NIR images covering
all your scenes/lighting/subjects. INT8 quality depends heavily on this.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--target", type=str, default="rk3588")
    p.add_argument("--calib-dir", type=str, required=True,
                   help="directory of NIR images for INT8 calibration")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--quant", type=str, default="int8", choices=["int8", "fp16", "none"])
    p.add_argument("--max-calib", type=int, default=500)
    args = p.parse_args()

    try:
        from rknn.api import RKNN
    except ImportError:
        print("rknn-toolkit2 is not installed. install it on a Linux x86_64 host.")
        sys.exit(1)

    calib_dir = Path(args.calib_dir)
    calib_imgs = []
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        calib_imgs.extend(sorted(calib_dir.rglob(f"*{ext}")))
    calib_imgs = calib_imgs[: args.max_calib]
    if not calib_imgs:
        print(f"no calibration images found in {calib_dir}")
        sys.exit(1)
    print(f"using {len(calib_imgs)} calibration images")

    dataset_txt = Path(args.out).with_suffix(".calib.txt")
    dataset_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_txt, "w") as f:
        for p_ in calib_imgs:
            f.write(str(p_.resolve()) + "\n")

    rknn = RKNN(verbose=False)
    rknn.config(
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[127.5, 127.5, 127.5]],
        target_platform=args.target,
        quantized_dtype="w8a8" if args.quant == "int8" else "w16a16",
    )
    print(f"loading {args.onnx}")
    if rknn.load_onnx(model=args.onnx) != 0:
        print("load_onnx failed"); sys.exit(1)

    do_quant = args.quant == "int8"
    print(f"building (quant={args.quant}, target={args.target})")
    if rknn.build(do_quantization=do_quant, dataset=str(dataset_txt) if do_quant else None) != 0:
        print("build failed"); sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rknn.export_rknn(str(out_path)) != 0:
        print("export_rknn failed"); sys.exit(1)
    print(f"exported RKNN to {out_path}")
    rknn.release()


if __name__ == "__main__":
    main()
