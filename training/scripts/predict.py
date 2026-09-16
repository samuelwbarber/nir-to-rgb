#!/usr/bin/env python3
"""
predict.py — Run NIR → RGB inference with a trained NAFNet checkpoint.

Self-contained: no imports from other project files required.
Only external dependencies: torch, torchvision (optional), opencv-python, numpy, matplotlib (optional).

Usage:
    python predict.py --input <nir_image> [--checkpoint <path.pth>] [--output <out.png>] [--cpu]

Examples:
    python predict.py --input my_nir.png
    python predict.py --input my_nir.jpg \
        --checkpoint /home/sbarber9876/NIR-RGB/experiments/phase2_teacher_nafnet64/checkpoints/epoch_0119.pth \
        --output predicted_rgb.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Default checkpoint
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = (
    "/home/sbarber9876/NIR-RGB/experiments"
    "/phase2_teacher_nafnet64/checkpoints/epoch_0119.pth"
)

# NAFNet hyper-parameters — match teacher_nafnet.yaml
NAFNET_KWARGS = dict(
    in_channels=3,
    out_channels=3,
    width=64,
    enc_blk_nums=(2, 2, 4, 8),
    middle_blk_num=12,
    dec_blk_nums=(2, 2, 2, 2),
    dropout=0.0,
    global_residual=False,
    output_tanh=True,
)

IMAGE_SIZE = 256  # default resize target


# ===========================================================================
# NAFNet — inlined from src/models/nafnet.py
# Reference: https://github.com/megvii-research/NAFNet
# ===========================================================================

class LayerNorm2d(nn.Module):
    """Per-position channel-wise normalisation, as used in NAFNet."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, dropout=0.0):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw, dw, 3, 1, 1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw // 2, dw // 2, 1, 1, 0),
        )
        self.conv3 = nn.Conv2d(dw // 2, c, 1, 1, 0)

        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1, 1, 0)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1, 1, 0)

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=32,
        enc_blk_nums=(2, 2, 4, 8),
        middle_blk_num=12,
        dec_blk_nums=(2, 2, 2, 2),
        dropout=0.0,
        global_residual=False,
        output_tanh=True,
    ):
        super().__init__()
        assert len(enc_blk_nums) == len(dec_blk_nums)

        self.intro = nn.Conv2d(in_channels, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, out_channels, 3, 1, 1)
        self.global_residual = global_residual and (in_channels == out_channels)
        self.output_tanh = output_tanh

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, 2 * ch, 2, 2))
            ch = 2 * ch

        self.middle_blks = nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(middle_blk_num)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, 1, 0, bias=False),
                nn.PixelShuffle(2),
            ))
            ch = ch // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(n)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def _check_image_size(self, x):
        _, _, h, w = x.size()
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return x, h, w

    def forward(self, inp):
        x, H, W = self._check_image_size(inp)
        residual = x
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for up, dec, skip in zip(self.ups, self.decoders, skips[::-1]):
            x = up(x)
            x = x + skip
            x = dec(x)
        x = self.ending(x)
        if self.global_residual:
            x = x + residual
        if self.output_tanh:
            x = torch.tanh(x)
        return x[:, :, :H, :W]


# ===========================================================================
# Image helpers
# ===========================================================================

def read_nir(path: str) -> np.ndarray:
    """Read an NIR image and return a uint8 RGB numpy array (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    # 16-bit → 8-bit
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    # Grayscale → 3-channel
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Drop alpha if present
    if img.shape[2] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(img_rgb: np.ndarray, size: int) -> torch.Tensor:
    """Resize to (size × size) and normalise to [-1, 1]."""
    resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized.astype(np.float32) / 127.5 - 1.0)
    return t.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 3, H, W)


def postprocess(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output (1, 3, H, W) in [-1,1] → uint8 RGB (H, W, 3)."""
    arr = tensor.squeeze(0).detach().clamp(-1, 1).cpu().numpy()
    arr = ((arr + 1.0) * 127.5).astype(np.uint8)
    return arr.transpose(1, 2, 0)


# ===========================================================================
# Model loading
# ===========================================================================

def load_generator(checkpoint_path: str, device: torch.device) -> NAFNet:
    """Build NAFNet and load weights from a checkpoint."""
    G = NAFNet(**NAFNET_KWARGS).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    # Full training checkpoint stores weights under key "G"
    G.load_state_dict(state["G"] if "G" in state else state)
    G.eval()
    print(f"[predict] Loaded generator from: {checkpoint_path}")
    return G


# ===========================================================================
# Display / save
# ===========================================================================

def show_result(nir_rgb: np.ndarray, pred_rgb: np.ndarray, save_path: str | None = None):
    """Show NIR input and predicted RGB side-by-side. Save if path given."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(10, 5), facecolor="#1a1a2e")
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05)

        for col, (title, img) in enumerate(
            zip(["NIR Input", "Predicted RGB"], [nir_rgb, pred_rgb])
        ):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img)
            ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
            ax.axis("off")

        fig.suptitle("NIR → RGB Prediction", color="#a0a0ff", fontsize=16,
                     fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150,
                        facecolor=fig.get_facecolor())
            print(f"[predict] Saved output to: {save_path}")

        plt.show()

    except ImportError:
        print("[predict] matplotlib not found — falling back to OpenCV save.")
        combined = np.concatenate([nir_rgb, pred_rgb], axis=1)
        out = save_path or "predicted_rgb.png"
        cv2.imwrite(str(out), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"[predict] Saved output to: {out}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="NIR → RGB inference (self-contained, no project imports)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      "-i", required=True,
                   help="Path to the input NIR image.")
    p.add_argument("--checkpoint", "-c", default=DEFAULT_CHECKPOINT,
                   help="Path to the model checkpoint (.pth).")
    p.add_argument("--output",     "-o", default=None,
                   help="Optional path to save the predicted RGB image.")
    p.add_argument("--size",       type=int, default=IMAGE_SIZE,
                   help="Resize input to this square size before inference.")
    p.add_argument("--cpu",        action="store_true",
                   help="Force CPU even if CUDA is available.")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[predict] Device: {device}")

    G = load_generator(args.checkpoint, device)

    nir_rgb = read_nir(args.input)
    nir_display = cv2.resize(nir_rgb, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
    input_tensor = preprocess(nir_rgb, size=args.size).to(device)
    print(f"[predict] Input resized to {args.size}×{args.size}")

    with torch.no_grad():
        output_tensor = G(input_tensor)

    pred_rgb = postprocess(output_tensor)
    print(f"[predict] Done. Output shape: {pred_rgb.shape}")

    show_result(nir_display, pred_rgb, save_path=args.output)


if __name__ == "__main__":
    main()
