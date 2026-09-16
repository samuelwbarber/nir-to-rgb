import math

import torch
import torch.nn as nn
from pytorch_msssim import ssim as _ssim


@torch.no_grad()
def compute_psnr(pred, target):
    pred01 = (pred.clamp(-1, 1) + 1) * 0.5
    target01 = (target.clamp(-1, 1) + 1) * 0.5
    mse = ((pred01 - target01) ** 2).mean()
    if mse.item() == 0:
        return 100.0
    return float(20.0 * math.log10(1.0) - 10.0 * math.log10(mse.item()))


@torch.no_grad()
def compute_ssim(pred, target):
    pred01 = (pred.clamp(-1, 1) + 1) * 0.5
    target01 = (target.clamp(-1, 1) + 1) * 0.5
    return float(_ssim(pred01, target01, data_range=1.0, size_average=True))


class LPIPSWrapper(nn.Module):
    def __init__(self, net="alex"):
        super().__init__()
        import lpips
        self.metric = lpips.LPIPS(net=net, verbose=False)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, pred, target):
        return float(self.metric(pred, target).mean())
