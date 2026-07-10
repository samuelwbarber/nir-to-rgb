from .unet import UNetGenerator
from .nafnet import NAFNet
from .discriminator import PatchDiscriminator


def build_generator(cfg, in_channels, out_channels):
    g = cfg.model.generator
    if g.type == "unet":
        return UNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=g.base_channels,
            use_dropout=g.use_dropout,
        )
    if g.type == "nafnet":
        return NAFNet(
            in_channels=in_channels,
            out_channels=out_channels,
            width=g.width,
            enc_blk_nums=tuple(g.enc_blk_nums),
            middle_blk_num=g.middle_blk_num,
            dec_blk_nums=tuple(g.dec_blk_nums),
            dropout=getattr(g, "dropout", 0.0),
            global_residual=getattr(g, "global_residual", False),
            output_tanh=getattr(g, "output_tanh", True),
        )
    raise ValueError(f"unknown generator type: {g.type}")


def build_discriminator(cfg, in_channels, out_channels):
    d = cfg.model.discriminator
    if d.type == "patchgan":
        return PatchDiscriminator(
            in_channels=in_channels + out_channels,
            base_channels=d.base_channels,
            n_layers=d.n_layers,
        )
    raise ValueError(f"unknown discriminator type: {d.type}")
