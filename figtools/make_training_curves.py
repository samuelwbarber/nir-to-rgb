"""Plot teacher training curves from a TensorBoard event file.

Pure-Python TFRecord + protobuf scalar reader (no tensorboard/tbparse needed).
Reads val/psnr, val/ssim, val/lpips (per-epoch) and train/g_loss (per-step)
written by src/training/trainer.py, and renders a 2x2 figure.

Usage:
    py -3 make_training_curves.py <tb_event_file> <out.pdf>
"""
import struct
import sys
from collections import defaultdict


def _read_varint(buf, i):
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, i


def _parse_fields(buf):
    """Yield (field_number, wire_type, value_bytes_or_int) for a protobuf message."""
    i = 0
    n = len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        fnum, wtype = tag >> 3, tag & 7
        if wtype == 0:          # varint
            val, i = _read_varint(buf, i)
            yield fnum, wtype, val
        elif wtype == 1:        # 64-bit
            yield fnum, wtype, buf[i:i + 8]; i += 8
        elif wtype == 2:        # length-delimited
            ln, i = _read_varint(buf, i)
            yield fnum, wtype, buf[i:i + ln]; i += ln
        elif wtype == 5:        # 32-bit
            yield fnum, wtype, buf[i:i + 4]; i += 4
        else:
            raise ValueError(f"bad wire type {wtype}")


def parse_event_file(path):
    """Return {tag: [(step, value), ...]} for scalar summaries."""
    scalars = defaultdict(list)
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    n = len(data)
    while i + 12 <= n:
        (length,) = struct.unpack_from("<Q", data, i); i += 8
        i += 4                                  # skip length CRC
        if i + length + 4 > n:
            break
        event = data[i:i + length]; i += length
        i += 4                                  # skip data CRC
        step = 0
        summary = None
        for fnum, wt, val in _parse_fields(event):
            if fnum == 2 and wt == 0:
                step = val
            elif fnum == 5 and wt == 2:
                summary = val
        if summary is None:
            continue
        for fnum, wt, val in _parse_fields(summary):     # Summary.value (field 1, repeated)
            if fnum != 1 or wt != 2:
                continue
            tag, simple = None, None
            for vf, vwt, vv in _parse_fields(val):       # Summary.Value
                if vf == 1 and vwt == 2:
                    tag = vv.decode("utf-8", "replace")
                elif vf == 2 and vwt == 5:
                    (simple,) = struct.unpack("<f", vv)
            if tag is not None and simple is not None:
                scalars[tag].append((step, simple))
    return scalars


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tb_path, out_path = sys.argv[1], sys.argv[2]
    sc = parse_event_file(tb_path)
    print("tags found:", {k: len(v) for k, v in sc.items()})

    def xy(tag):
        pts = sorted(sc.get(tag, []))
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, axs = plt.subplots(2, 2, figsize=(9, 6))

    gx, gy = xy("train/g_loss")
    axs[0, 0].plot(gx, gy, lw=0.8, color="#444")
    axs[0, 0].set_title("Generator loss"); axs[0, 0].set_xlabel("step"); axs[0, 0].set_ylabel("$\\mathcal{L}_G$")

    px, py = xy("val/psnr")
    axs[0, 1].plot(px, py, marker="o", ms=3, color="#1f77b4")
    axs[0, 1].set_title("Validation PSNR"); axs[0, 1].set_xlabel("epoch"); axs[0, 1].set_ylabel("dB")

    lx, ly = xy("val/lpips")
    axs[1, 0].plot(lx, ly, marker="o", ms=3, color="#d62728")
    axs[1, 0].set_title("Validation LPIPS"); axs[1, 0].set_xlabel("epoch"); axs[1, 0].set_ylabel("LPIPS ($\\downarrow$)")

    # Panel (1,1): foundation-model feature cosine similarities if present, else SSIM.
    cos_tags = [("val/dinov2_cos", "DINOv2"), ("val/clip_cos", "CLIP"), ("val/resnet50_cos", "ResNet-50")]
    if any(t in sc for t, _ in cos_tags):
        for tag, lab in cos_tags:
            if tag in sc:
                cx, cy = xy(tag)
                axs[1, 1].plot(cx, cy, marker="o", ms=3, label=lab)
        axs[1, 1].set_title("Validation feature cosine sim."); axs[1, 1].set_xlabel("epoch")
        axs[1, 1].set_ylabel("cosine sim. ($\\uparrow$)"); axs[1, 1].legend(fontsize=7)
    else:
        sx, sy = xy("val/ssim")
        axs[1, 1].plot(sx, sy, marker="o", ms=3, color="#2ca02c")
        axs[1, 1].set_title("Validation SSIM"); axs[1, 1].set_xlabel("epoch"); axs[1, 1].set_ylabel("SSIM")

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
