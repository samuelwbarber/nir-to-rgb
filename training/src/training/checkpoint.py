from pathlib import Path

import numpy as np
import torch
import cv2


def save_checkpoint(path, G, D, opt_g, opt_d, epoch, G_ema=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }
    if G_ema is not None:
        payload["G_ema"] = G_ema.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def _shapes_match(state_dict, module):
    msd = module.state_dict()
    for k, v in state_dict.items():
        if k not in msd:
            return False
        if msd[k].shape != v.shape:
            return False
    return True


def load_checkpoint(path, G, D=None, opt_g=None, opt_d=None, G_ema=None,
                    map_location="cpu", weights_only=False, strict_d=True):
    """Load a checkpoint.

    weights_only=True: load G (and G_ema if present), try D if shapes match,
    skip optimizers, return -1 for epoch so the caller can reset the counter.
    """
    state = torch.load(str(path), map_location=map_location)
    G.load_state_dict(state["G"])
    if G_ema is not None:
        if "G_ema" in state:
            G_ema.load_state_dict(state["G_ema"])
        else:
            G_ema.load_state_dict(state["G"])
    if D is not None and "D" in state:
        if weights_only and not _shapes_match(state["D"], D):
            print("[load_checkpoint] D shapes don't match — starting D fresh")
        else:
            D.load_state_dict(state["D"], strict=strict_d)
    if weights_only:
        return -1
    if opt_g is not None and "opt_g" in state:
        opt_g.load_state_dict(state["opt_g"])
    if opt_d is not None and "opt_d" in state:
        opt_d.load_state_dict(state["opt_d"])
    return int(state.get("epoch", -1))


def _denorm_to_uint8(x):
    x = (x.detach().clamp(-1, 1) + 1) * 127.5
    return x.cpu().numpy().astype(np.uint8)


def save_grid(nir, fake, rgb, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = nir.size(0)
    rows = []
    nir_np = _denorm_to_uint8(nir)
    fake_np = _denorm_to_uint8(fake)
    rgb_np = _denorm_to_uint8(rgb)
    for i in range(n):
        triple = np.concatenate([nir_np[i], fake_np[i], rgb_np[i]], axis=2)
        rows.append(triple)
    grid = np.concatenate(rows, axis=1)
    grid = grid.transpose(1, 2, 0)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), grid_bgr)
