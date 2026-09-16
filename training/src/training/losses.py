import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_msssim import MS_SSIM


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_indices=(3, 8, 15, 22)):
        super().__init__()
        try:
            weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
            vgg = torchvision.models.vgg16(weights=weights).features
        except Exception:
            vgg = torchvision.models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList()
        last = 0
        for idx in layer_indices:
            self.blocks.append(vgg[last:idx + 1])
            last = idx + 1
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x):
        x = (x + 1) * 0.5
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        x_p = self._prep(pred)
        x_t = self._prep(target)
        loss = pred.new_zeros(())
        for block in self.blocks:
            x_p = block(x_p)
            x_t = block(x_t)
            loss = loss + F.l1_loss(x_p, x_t)
        return loss


class GANLoss(nn.Module):
    def __init__(self, gan_type="lsgan"):
        super().__init__()
        if gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"unknown gan_type: {gan_type}")

    def forward(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


class MSSSIMLoss(nn.Module):
    def __init__(self, data_range=2.0):
        super().__init__()
        self.metric = MS_SSIM(data_range=data_range, size_average=True, channel=3)

    def forward(self, pred, target):
        return 1.0 - self.metric(pred + 1.0, target + 1.0)


class FeatureMapLoss(nn.Module):
    """Diverse pretrained-encoder feature loss for task-agnostic NIR→RGB alignment.

    Encoders span foundation-model families (CNN classification, self-supervised
    ViT, language-aligned ViT) so the generator is pulled toward outputs that
    match real RGB statistics under *any* downstream backbone, not one in
    particular. Each extractor contributes plain L1 on raw feature maps + L1 on
    the pooled global embedding (no L2-normalization — see history for why that
    silently kills the gradient scale).

    Per-extractor weights are absolute contributions; the trainer applies an
    additional 0→1 warmup ramp on top.
    """

    def __init__(self, extractors_cfg=None, input_size=224, **legacy):
        super().__init__()
        # legacy kwargs (`models=...`) preserved for any caller that still passes them.
        if extractors_cfg is None and "models" in legacy:
            extractors_cfg = [{"name": m, "weight": 1.0} for m in legacy["models"]]
        if not extractors_cfg:
            raise ValueError("FeatureMapLoss requires `extractors_cfg` or legacy `models=`")
        self.input_size = int(input_size)
        self.extractors = nn.ModuleDict()
        self.weights = {}
        self.last_per_extractor = {}
        self.last_pred_pooled = {}
        self.last_target_pooled = {}
        for cfg in extractors_cfg:
            name = cfg.name if hasattr(cfg, "name") else cfg["name"]
            weight = float(cfg.weight if hasattr(cfg, "weight") else cfg.get("weight", 1.0))
            if name == "resnet50":
                enc = _ResNet50Encoder()
            elif name == "dinov2_vits14":
                enc = _DINOv2Encoder("vit_small_patch14_dinov2", self.input_size)
            elif name == "clip_vitb32":
                enc = _CLIPEncoder("vit_base_patch32_clip_224", 224)
            elif name == "segformer_ade_b0":
                enc = _SegFormerADEEncoder()
            elif name == "depthanything_small":
                enc = _DepthAnythingSmallEncoder()
            else:
                raise ValueError(f"unknown feature_map extractor: {name}")
            self.extractors[name] = enc
            self.weights[name] = weight
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x, size):
        x = (x + 1) * 0.5
        if x.shape[-1] != size or x.shape[-2] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    @staticmethod
    def _feature_l1(pred, target):
        return F.l1_loss(pred.float(), target.float())

    def forward(self, pred, target):
        per_extractor = {}
        total = pred.new_zeros(())
        for name, enc in self.extractors.items():
            size = getattr(enc, "input_size", self.input_size)
            pred_in = self._prep(pred, size)
            with torch.no_grad():
                target_in = self._prep(target, size)

            pred_out = enc(pred_in)
            with torch.no_grad():
                target_out = enc(target_in)

            ext_loss = pred.new_zeros(())
            n = 0
            for pf, tf in zip(pred_out["dense"], target_out["dense"]):
                ext_loss = ext_loss + self._feature_l1(pf, tf)
                n += 1
            ext_loss = ext_loss + self._feature_l1(pred_out["pooled"], target_out["pooled"])
            n += 1
            ext_loss = ext_loss / max(1, n)
            weighted = self.weights[name] * ext_loss
            per_extractor[name] = weighted
            total = total + weighted

        # Cache float values for trainer logging without holding the graph.
        self.last_per_extractor = {k: float(v.detach()) for k, v in per_extractor.items()}
        return total

    @torch.no_grad()
    def pooled_embeddings(self, x):
        """Return {extractor_name: pooled_embed (B, D)} — for val-time cosine logging."""
        out = {}
        for name, enc in self.extractors.items():
            size = getattr(enc, "input_size", self.input_size)
            out[name] = enc(self._prep(x, size))["pooled"]
        return out


class _ResNet50Encoder(nn.Module):
    """ImageNet-supervised CNN. 5 dense stage outputs + GAP embedding."""

    input_size = 224

    def __init__(self):
        super().__init__()
        try:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            model = torchvision.models.resnet50(weights=weights)
        except Exception:
            model = torchvision.models.resnet50(pretrained=True)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.eval()

    def forward(self, x):
        s = self.stem(x)
        l1 = self.layer1(self.maxpool(s))
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        pooled = F.adaptive_avg_pool2d(l4, 1).flatten(1)
        return {"dense": [s, l1, l2, l3, l4], "pooled": pooled}


class _ViTTimmEncoder(nn.Module):
    """Generic timm ViT encoder: returns reshaped patch tokens (dense) + CLS (pooled)."""

    def __init__(self, model_name, input_size):
        super().__init__()
        import timm
        try:
            self.model = timm.create_model(model_name, pretrained=True, img_size=input_size, num_classes=0)
        except TypeError:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.input_size = int(input_size)
        # patch_size may be tuple (ph, pw); we assume square.
        ps = getattr(self.model, "patch_embed", None)
        patch = getattr(ps, "patch_size", None) if ps is not None else None
        if isinstance(patch, (tuple, list)):
            patch = patch[0]
        self.patch_size = int(patch) if patch else None
        self.num_prefix = int(getattr(self.model, "num_prefix_tokens", 1))
        self.eval()

    def forward(self, x):
        feats = self.model.forward_features(x)
        # feats can be (B, N, D) for ViTs or (B, D, H, W) for hybrid backbones.
        if feats.ndim == 3:
            cls_tok = feats[:, 0]  # use first prefix token as global embedding
            patch_tokens = feats[:, self.num_prefix:]
            B, N, D = patch_tokens.shape
            side = int(round(N ** 0.5))
            dense = patch_tokens.transpose(1, 2).reshape(B, D, side, side).contiguous()
            return {"dense": [dense], "pooled": cls_tok}
        else:
            pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            return {"dense": [feats], "pooled": pooled}


class _DINOv2Encoder(_ViTTimmEncoder):
    pass


class _CLIPEncoder(_ViTTimmEncoder):
    pass


class _SegFormerADEEncoder(nn.Module):
    """HF SegFormer-B0 finetuned on ADE20K. Used in eval_lineage for per-class Dice.

    Returns the 4 encoder hidden-state stages (channels [32, 64, 160, 256]) as
    dense features plus GAP of the last stage as the pooled embedding. Training
    against these features pulls the generator toward outputs whose ADE-class
    layout matches the GT — directly proxying the downstream Dice metric.
    """

    input_size = 224

    def __init__(self):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        self.backbone = model.segformer
        self.eval()

    def forward(self, x):
        out = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
        stages = list(out.hidden_states)  # 4 stage feature maps
        pooled = F.adaptive_avg_pool2d(stages[-1], 1).flatten(1)
        return {"dense": stages, "pooled": pooled}


class _DepthAnythingSmallEncoder(nn.Module):
    """HF Depth-Anything-Small monocular depth estimator (DINOv2-S backbone + DPT head).

    Returns the predicted depth map as a single dense feature plus its global
    mean as the pooled embedding. Training against this pulls the generator
    toward outputs whose monocular depth structure matches the GT — directly
    proxying the downstream Depth MAE metric.
    """

    input_size = 224

    def __init__(self):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        self.model = AutoModelForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-small-hf"
        )
        self.eval()

    def forward(self, x):
        out = self.model(pixel_values=x)
        depth = out.predicted_depth  # (B, H, W)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # (B, 1, H, W)
        pooled = F.adaptive_avg_pool2d(depth, 1).flatten(1)
        return {"dense": [depth], "pooled": pooled}
