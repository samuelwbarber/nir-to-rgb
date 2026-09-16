import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN — operates on concatenated (input, target) pair."""

    def __init__(self, in_channels=6, base_channels=64, n_layers=3):
        super().__init__()
        c = base_channels
        layers = [
            nn.Conv2d(in_channels, c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        prev = c
        for i in range(1, n_layers):
            cur = min(c * (2 ** i), 512)
            layers += [
                nn.Conv2d(prev, cur, 4, 2, 1, bias=False),
                nn.BatchNorm2d(cur),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            prev = cur
        cur = min(c * (2 ** n_layers), 512)
        layers += [
            nn.Conv2d(prev, cur, 4, 1, 1, bias=False),
            nn.BatchNorm2d(cur),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cur, 1, 4, 1, 1),
        ]
        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, src, tgt):
        return self.model(torch.cat([src, tgt], dim=1))
