"""NAFNet — Simple Baselines for Image Restoration (Megvii, 2022).

Reference: https://github.com/megvii-research/NAFNet

Differences from the reference:
  - Final Tanh to keep output in [-1, 1] (matches our normalization contract).
  - Global input->output residual disabled by default (NIR vs RGB are different
    modalities; the residual would force the delta to encode all colour info).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Per-position channel-wise normalization, as used in NAFNet."""

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
