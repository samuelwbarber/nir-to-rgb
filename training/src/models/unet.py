import torch
import torch.nn as nn


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """8-level U-Net for 256x256 inputs (Pix2Pix-style)."""

    def __init__(self, in_channels=3, out_channels=3, base_channels=64, use_dropout=True):
        super().__init__()
        c = base_channels
        self.d1 = _Down(in_channels, c, normalize=False)
        self.d2 = _Down(c, c * 2)
        self.d3 = _Down(c * 2, c * 4)
        self.d4 = _Down(c * 4, c * 8)
        self.d5 = _Down(c * 8, c * 8)
        self.d6 = _Down(c * 8, c * 8)
        self.d7 = _Down(c * 8, c * 8)
        self.d8 = _Down(c * 8, c * 8, normalize=False)

        self.u1 = _Up(c * 8, c * 8, dropout=use_dropout)
        self.u2 = _Up(c * 16, c * 8, dropout=use_dropout)
        self.u3 = _Up(c * 16, c * 8, dropout=use_dropout)
        self.u4 = _Up(c * 16, c * 8)
        self.u5 = _Up(c * 16, c * 4)
        self.u6 = _Up(c * 8, c * 2)
        self.u7 = _Up(c * 4, c)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(c * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)
        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        return self.final(u7)
