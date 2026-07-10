"""Evaluate ablation_08 only — same metrics as eval_all.py."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
import lpips
from pytorch_msssim import ssim as ssim_metric
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.dataset import PairedNIRRGBDataset, discover_pairs, split_pairs
from src.models.nafnet import NAFNet

CKPT = PROJECT_ROOT / "experiments/ablation_08_foundation_ensemble_aggressive/checkpoints/best.pth"
CFG_PATH = PROJECT_ROOT / "configs/ablation_08_foundation_ensemble_aggressive.yaml"
BASE_CFG_PATH = PROJECT_ROOT / "configs/ablation_01_baseline.yaml"

DINOV2_HF = "facebook/dinov2-small"
CLIP_HF = "openai/clip-vit-base-patch16"
ADE_HF = "nvidia/segformer-b0-finetuned-ade-512-512"
DEPTH_HF = "LiheYoung/depth-anything-small-hf"

DINOV2_IN = 224; CLIP_IN = 224; ADE_IN = 512; DEPTH_IN = 224
MASK_HW = (256, 256); DEPTH_HW = (256, 256)
ADE_NUM_CLASSES = 150; CMMD_SIGMA = 10.0; BATCH_SIZE = 4; NUM_WORKERS = 4

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def to_unit(x): return (x.clamp(-1, 1) + 1) * 0.5

def prep(x, size, mean, std, device):
    x = to_unit(x)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    m = mean.to(device).view(1,3,1,1); s = std.to(device).view(1,3,1,1)
    return (x - m) / s

def dinov2_embed(model, x, device):
    with torch.no_grad():
        return model(pixel_values=prep(x, DINOV2_IN, IMAGENET_MEAN, IMAGENET_STD, device)).pooler_output

def clip_embed(model, x, device):
    with torch.no_grad():
        vis = model.vision_model(pixel_values=prep(x, CLIP_IN, CLIP_MEAN, CLIP_STD, device))
        return model.visual_projection(vis.pooler_output)

def ade_mask(model, x, device, out_hw):
    with torch.no_grad():
        out = model(pixel_values=prep(x, ADE_IN, IMAGENET_MEAN, IMAGENET_STD, device))
    return F.interpolate(out.logits, size=out_hw, mode="bilinear", align_corners=False).argmax(dim=1)

def depth_map(model, x, device, out_hw):
    with torch.no_grad():
        d = model(pixel_values=prep(x, DEPTH_IN, IMAGENET_MEAN, IMAGENET_STD, device)).predicted_depth
    if d.dim() == 3: d = d.unsqueeze(1)
    return F.interpolate(d, size=out_hw, mode="bilinear", align_corners=False).squeeze(1)

def normalize_per_image(d, eps=1e-8):
    B = d.size(0); flat = d.reshape(B, -1)
    mn = flat.min(dim=1, keepdim=True).values; mx = flat.max(dim=1, keepdim=True).values
    return ((flat - mn) / (mx - mn + eps)).reshape_as(d)

def psnr_per_image(p, t):
    return -10.0 * torch.log10(((p - t)**2).mean(dim=(1,2,3)).clamp(min=1e-12))

def per_image_miou(pred, gt, nc):
    ious = []
    for c in range(nc):
        p = (pred == c); g = (gt == c)
        if not (p.any() or g.any()): continue
        ious.append(((p & g).sum().float() / (p | g).sum().float()).item())
    return sum(ious) / len(ious) if ious else 1.0

def cmmd_rbf(x, y, sigma=CMMD_SIGMA, chunk=512):
    x = x.float(); y = y.float()
    tsq = 2.0 * sigma * sigma
    def sk(a, b):
        s = 0.0
        for i in range(0, a.size(0), chunk):
            d2 = (a[i:i+chunk].unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)
            s += torch.exp(-d2 / tsq).sum().item()
        return s
    n = x.size(0); m = y.size(0)
    return sk(x,x)/(n*n) + sk(y,y)/(m*m) - 2*sk(x,y)/(n*m)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    from transformers import AutoModel, CLIPModel, SegformerForSemanticSegmentation, AutoModelForDepthEstimation
    import pyiqa

    print("loading foundation models...")
    dinov2 = AutoModel.from_pretrained(DINOV2_HF).eval().to(device)
    clip   = CLIPModel.from_pretrained(CLIP_HF).eval().to(device)
    ade    = SegformerForSemanticSegmentation.from_pretrained(ADE_HF).eval().to(device)
    depth  = AutoModelForDepthEstimation.from_pretrained(DEPTH_HF).eval().to(device)
    musiq  = pyiqa.create_metric("musiq", device=device)
    lpips_fn = lpips.LPIPS(net="alex").eval().to(device)
    for m in [dinov2, clip, ade, depth, lpips_fn]:
        for p in m.parameters(): p.requires_grad_(False)

    base_cfg = yaml.safe_load(open(BASE_CFG_PATH))
    data = base_cfg["data"]; split = base_cfg["split"]
    pairs = discover_pairs(PROJECT_ROOT / data["nir_dir"], PROJECT_ROOT / data["rgb_dir"], data["extensions"])
    _, _, test = split_pairs(pairs, split["train_ratio"], split["val_ratio"], split["test_ratio"], seed=split["seed"])
    ds = PairedNIRRGBDataset(test, image_size=data["image_size"], train=False, aug_cfg=None, mp_cache=None)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    n_test = len(ds)
    print(f"test set: {n_test} pairs")

    # Cache GT
    print("caching GT...")
    gt_dinov2, gt_clip, gt_mask, gt_depth = [], [], [], []
    for i, batch in enumerate(loader):
        rgb = batch["rgb"].to(device)
        gt_dinov2.append(dinov2_embed(dinov2, rgb, device).cpu())
        gt_clip.append(clip_embed(clip, rgb, device).cpu())
        gt_mask.append(ade_mask(ade, rgb, device, MASK_HW).to(torch.uint8).cpu())
        gt_depth.append(normalize_per_image(depth_map(depth, rgb, device, DEPTH_HW)).cpu())
        if (i+1) % 25 == 0 or (i+1) == len(loader):
            print(f"  GT {min((i+1)*BATCH_SIZE, n_test)}/{n_test}", flush=True)
    gt_dinov2 = torch.cat(gt_dinov2)
    gt_clip   = torch.cat(gt_clip)
    gt_mask   = torch.cat(gt_mask)
    gt_depth  = torch.cat(gt_depth)

    # Load ablation_08
    cfg = yaml.safe_load(open(CFG_PATH))
    g = cfg["model"]["generator"]
    G = NAFNet(in_channels=cfg["data"]["in_channels"], out_channels=cfg["data"]["out_channels"],
               width=g["width"], enc_blk_nums=tuple(g["enc_blk_nums"]),
               middle_blk_num=g["middle_blk_num"], dec_blk_nums=tuple(g["dec_blk_nums"]),
               dropout=g.get("dropout", 0.0), global_residual=g.get("global_residual", False),
               output_tanh=g.get("output_tanh", True))
    state = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    G.load_state_dict(state["G"])
    G = G.eval().to(device)
    epoch = state.get("epoch", -1)
    print(f"loaded ablation_08 epoch={epoch}")

    psnrs, ssims, lpipss = [], [], []
    dn_coses, cl_coses, mious, dmaes, mscores = [], [], [], [], []
    clip_embeds = []
    idx = 0
    for i, batch in enumerate(loader):
        nir = batch["nir"].to(device); rgb = batch["rgb"].to(device)
        with torch.no_grad(): pred = G(nir)
        B = nir.size(0)
        pu, ru = to_unit(pred), to_unit(rgb)
        psnrs.extend(psnr_per_image(pu, ru).cpu().tolist())
        ssims.extend(ssim_metric(pu, ru, data_range=1.0, size_average=False).cpu().tolist())
        with torch.no_grad():
            lpipss.extend(lpips_fn(pred.clamp(-1,1), rgb.clamp(-1,1)).flatten().cpu().tolist())
        p_dn = dinov2_embed(dinov2, pred, device)
        p_cl = clip_embed(clip, pred, device)
        p_ma = ade_mask(ade, pred, device, MASK_HW)
        p_de = normalize_per_image(depth_map(depth, pred, device, DEPTH_HW))
        with torch.no_grad(): p_mq = musiq(pu).flatten()
        g_dn = gt_dinov2[idx:idx+B].to(device)
        g_cl = gt_clip[idx:idx+B].to(device)
        g_de = gt_depth[idx:idx+B].to(device)
        dn_coses.extend(F.cosine_similarity(p_dn, g_dn, dim=-1).cpu().tolist())
        cl_coses.extend(F.cosine_similarity(p_cl, g_cl, dim=-1).cpu().tolist())
        for b in range(B):
            mious.append(per_image_miou(p_ma[b].long(), gt_mask[idx+b].to(device).long(), ADE_NUM_CLASSES))
        dmaes.extend((p_de - g_de).abs().mean(dim=(1,2)).cpu().tolist())
        mscores.extend(p_mq.cpu().tolist())
        clip_embeds.append(p_cl.cpu())
        idx += B
        if (i+1) % 25 == 0 or (i+1) == len(loader):
            print(f"  {min((i+1)*BATCH_SIZE, n_test)}/{n_test}", flush=True)

    clip_embeds = torch.cat(clip_embeds)
    cmmd_val = cmmd_rbf(clip_embeds, gt_clip)

    def m(v): return torch.tensor(v).mean().item()
    print(f"\n=== ablation_08 results (epoch {epoch}, n={n_test}) ===")
    print(f"PSNR      {m(psnrs):.4f}")
    print(f"SSIM      {m(ssims):.4f}")
    print(f"LPIPS     {m(lpipss):.4f}")
    print(f"DINOv2    {m(dn_coses):.4f}")
    print(f"CLIP cos  {m(cl_coses):.4f}")
    print(f"ADE mIoU  {m(mious):.4f}")
    print(f"Depth MAE {m(dmaes):.4f}")
    print(f"MUSIQ     {m(mscores):.4f}")
    print(f"CMMD      {cmmd_val:.6f}")

if __name__ == "__main__":
    main()
