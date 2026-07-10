import copy
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models import build_generator, build_discriminator
from ..eval.metrics import compute_psnr, compute_ssim, LPIPSWrapper
from .losses import VGGPerceptualLoss, GANLoss, MSSSIMLoss, FeatureMapLoss
from .checkpoint import save_checkpoint, load_checkpoint, save_grid


def _parse_amp(amp_cfg):
    """Return (enabled, dtype, use_scaler)."""
    if amp_cfg is True or amp_cfg == "fp16":
        return True, torch.float16, True
    if amp_cfg == "bf16":
        return True, torch.bfloat16, False
    return False, torch.float32, False


class Trainer:
    def __init__(self, cfg, train_loader, val_loader, run_dir, device):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = Path(run_dir)
        self.device = device

        self.G = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels).to(device)
        self.D = build_discriminator(cfg, cfg.data.in_channels, cfg.data.out_channels).to(device)

        # Knowledge-distillation teacher (optional) — frozen, never updated.
        distill_cfg = getattr(cfg, "distill", None)
        if distill_cfg is not None:
            teacher_ckpt = Path(distill_cfg.teacher_checkpoint)
            if not teacher_ckpt.exists():
                raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")
            from ..models.nafnet import NAFNet
            tc = distill_cfg.teacher
            self.teacher = NAFNet(
                in_channels=cfg.data.in_channels,
                out_channels=cfg.data.out_channels,
                width=tc.width,
                enc_blk_nums=tuple(tc.enc_blk_nums),
                middle_blk_num=tc.middle_blk_num,
                dec_blk_nums=tuple(tc.dec_blk_nums),
                dropout=getattr(tc, "dropout", 0.0),
                global_residual=getattr(tc, "global_residual", False),
                output_tanh=getattr(tc, "output_tanh", True),
            ).to(device)
            state = torch.load(str(teacher_ckpt), map_location="cpu", weights_only=False)
            self.teacher.load_state_dict(state["G"])
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.kd_l1_weight = float(getattr(cfg.loss, "kd_l1_weight", 0.0))
            self.kd_feature_weight = float(getattr(cfg.loss, "kd_feature_weight", 0.0))
            print(f"[distill] teacher loaded from {teacher_ckpt} (epoch={state.get('epoch')}, kd_l1_weight={self.kd_l1_weight}, kd_feature_weight={self.kd_feature_weight})")
        else:
            self.teacher = None
            self.kd_l1_weight = 0.0
            self.kd_feature_weight = 0.0

        self.perceptual = VGGPerceptualLoss(tuple(cfg.loss.perceptual_layers)).to(device)
        self.gan_loss = GANLoss(cfg.loss.gan_type).to(device)
        self.msssim_loss = MSSSIMLoss().to(device)
        self.feature_map_loss = None
        self.feature_map_warmup_epochs = max(0, int(getattr(cfg.loss, "feature_map_warmup_epochs", 0)))
        extractors_cfg = getattr(cfg.loss, "feature_map_extractors", None)
        if extractors_cfg:
            input_size = int(getattr(cfg.loss, "feature_map_input_size", 224))
            self.feature_map_loss = FeatureMapLoss(extractors_cfg=extractors_cfg, input_size=input_size).to(device)
            self.feature_map_weight_target = 1.0
        else:
            self.feature_map_weight_target = 0.0

        self.opt_g = torch.optim.Adam(
            self.G.parameters(),
            lr=cfg.optim.generator.lr,
            betas=tuple(cfg.optim.generator.betas),
        )
        self.opt_d = torch.optim.Adam(
            self.D.parameters(),
            lr=cfg.optim.discriminator.lr,
            betas=tuple(cfg.optim.discriminator.betas),
        )

        self.lr_sched_g = None
        lr_sched_cfg = getattr(cfg.optim, "lr_schedule", None)
        if lr_sched_cfg is not None and getattr(lr_sched_cfg, "type", None) == "cosine":
            total_ep = int(cfg.train.epochs)
            warmup_ep = int(getattr(lr_sched_cfg, "warmup_epochs", 0))
            base_lr = float(cfg.optim.generator.lr)
            min_lr = float(getattr(lr_sched_cfg, "min_lr", 0.0))
            def _cosine_lr_lambda(ep):
                if ep < warmup_ep:
                    return float(ep + 1) / max(1, warmup_ep)
                prog = (ep - warmup_ep) / max(1, total_ep - warmup_ep)
                prog = min(1.0, max(0.0, prog))
                cos_val = 0.5 * (1.0 + math.cos(math.pi * prog))
                return (min_lr + (base_lr - min_lr) * cos_val) / base_lr
            self.lr_sched_g = torch.optim.lr_scheduler.LambdaLR(self.opt_g, lr_lambda=_cosine_lr_lambda)

        amp_enabled, amp_dtype, use_scaler = _parse_amp(getattr(cfg.train, "amp", False))
        self.amp = amp_enabled and device.startswith("cuda")
        self.amp_dtype = amp_dtype
        self.use_scaler = use_scaler and self.amp
        self.scaler_g = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        self.scaler_d = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        self.grad_clip = float(getattr(cfg.train, "grad_clip", 0.0))
        self.grad_accum = max(1, int(getattr(cfg.train, "grad_accum", 1)))
        self._epoch_nonfinite = 0

        ema_decay = float(getattr(cfg.train, "ema_decay", 0.0))
        if ema_decay > 0.0:
            self.ema_decay = ema_decay
            self.G_ema = copy.deepcopy(self.G).to(device)
            for p in self.G_ema.parameters():
                p.requires_grad_(False)
        else:
            self.ema_decay = 0.0
            self.G_ema = None

        gan_warmup = int(getattr(cfg.loss, "gan_warmup_epochs", 0))
        self.gan_warmup_epochs = max(0, gan_warmup)
        self.gan_weight_target = float(cfg.loss.gan_weight)

        self.lpips = LPIPSWrapper(net=cfg.eval.lpips_net).to(device) if "lpips" in cfg.eval.metrics else None

        self.writer = SummaryWriter(str(self.run_dir / "tb"))
        self.global_step = 0
        self.start_epoch = 0
        self.best_psnr = float("-inf")
        self.best_metric_name = str(getattr(cfg.train, "best_metric", "val/psnr"))
        self.best_metric_mode = str(getattr(cfg.train, "best_metric_mode", "max")).lower()
        if self.best_metric_mode not in ("max", "min"):
            raise ValueError(f"best_metric_mode must be 'max' or 'min', got {self.best_metric_mode!r}")
        self.best_metric_value = float("-inf") if self.best_metric_mode == "max" else float("inf")

        if cfg.train.resume:
            resume_path = cfg.train.resume
            if resume_path == "latest":
                resume_path = self.run_dir / "checkpoints" / "latest.pth"
            weights_only = getattr(cfg.train, "resume_mode", "full") == "weights_only"
            if Path(resume_path).exists():
                print(f"Resuming from {resume_path} (weights_only={weights_only})")
                ep = load_checkpoint(
                    resume_path, self.G, self.D, self.opt_g, self.opt_d,
                    G_ema=self.G_ema, weights_only=weights_only,
                )
                self.start_epoch = 0 if weights_only else ep + 1
            elif cfg.train.resume != "latest":
                raise FileNotFoundError(f"Resume checkpoint {resume_path} not found.")

        self.fixed_val_batch = self._collect_fixed_batch()
        # accumulation state — gradients persist across micro-batches
        self._accum_step = 0

    def _collect_fixed_batch(self):
        if self.val_loader is None:
            return None
        n = self.cfg.train.sample_count
        nirs, rgbs = [], []
        for batch in self.val_loader:
            nirs.append(batch["nir"])
            rgbs.append(batch["rgb"])
            if sum(t.shape[0] for t in nirs) >= n:
                break
        if not nirs:
            return None
        nir = torch.cat(nirs, 0)[:n].to(self.device)
        rgb = torch.cat(rgbs, 0)[:n].to(self.device)
        return nir, rgb

    @staticmethod
    def _nir_grad_weight(nir, w_min, win):
        """Per-pixel L1 weight from local NIR variability.

        Where the input NIR has no local variation (uniform paper, blank wall),
        the model has no signal to predict chroma — penalising it with full L1
        forces the conditional mean and washes out outputs. Down-weight those
        pixels to ``w_min``. High-variance NIR regions (edges, texture) keep
        weight 1.0. Per-image normalised so absolute NIR scale doesn't matter.
        """
        gray = nir.mean(dim=1, keepdim=True)
        pad = win // 2
        avg = F.avg_pool2d(gray, win, stride=1, padding=pad)
        sq = F.avg_pool2d(gray * gray, win, stride=1, padding=pad)
        std = (sq - avg * avg).clamp(min=0).sqrt()
        std_max = std.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return w_min + (1.0 - w_min) * (std / std_max)

    @torch.no_grad()
    def _ema_update(self):
        if self.G_ema is None:
            return
        d = self.ema_decay
        for ep, p in zip(self.G_ema.parameters(), self.G.parameters()):
            ep.data.mul_(d).add_(p.data, alpha=1.0 - d)
        # Also EMA buffers (BatchNorm-style running stats — NAFNet uses LN so this
        # is usually a no-op, but harmless and correct if buffers exist).
        for eb, b in zip(self.G_ema.buffers(), self.G.buffers()):
            eb.data.copy_(b.data)

    def _current_gan_weight(self, epoch):
        if self.gan_warmup_epochs <= 0:
            return self.gan_weight_target
        if epoch >= self.gan_warmup_epochs:
            return self.gan_weight_target
        # linear ramp 0 → target over `gan_warmup_epochs` epochs
        return self.gan_weight_target * (epoch / self.gan_warmup_epochs)

    def _current_feature_map_weight(self, epoch):
        if self.feature_map_loss is None:
            return 0.0
        if self.feature_map_warmup_epochs <= 0:
            return self.feature_map_weight_target
        if epoch >= self.feature_map_warmup_epochs:
            return self.feature_map_weight_target
        return self.feature_map_weight_target * (epoch / self.feature_map_warmup_epochs)

    def train_step(self, nir, rgb, region_mask=None, gan_w=None, feature_map_w=None):
        amp_ctx = torch.amp.autocast("cuda", enabled=self.amp, dtype=self.amp_dtype)
        gan_w = self.gan_weight_target if gan_w is None else float(gan_w)
        feature_map_w = self.feature_map_weight_target if feature_map_w is None else float(feature_map_w)
        accum = self.grad_accum

        # ---- D step ----
        train_d = gan_w > 0.0
        with amp_ctx:
            fake = self.G(nir)
            if train_d:
                d_real = self.D(nir, rgb)
                d_fake = self.D(nir, fake.detach())
                d_loss = 0.5 * (self.gan_loss(d_real, True) + self.gan_loss(d_fake, False))
            else:
                d_loss = torch.zeros((), device=nir.device)
        d_finite = bool(torch.isfinite(d_loss).item())
        if train_d and d_finite:
            if self._accum_step == 0:
                self.opt_d.zero_grad(set_to_none=True)
            self.scaler_d.scale(d_loss / accum).backward()
            if self._accum_step + 1 == accum:
                if self.grad_clip > 0:
                    self.scaler_d.unscale_(self.opt_d)
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)
                self.scaler_d.step(self.opt_d)
                self.scaler_d.update()

        # ---- G step ----
        # Freeze D so g_loss.backward() doesn't accumulate gradients into D.grad —
        # without this, grad_accum>1 corrupts D's update direction (the leaked
        # adversarial gradient from G points opposite to D's own d_loss gradient).
        for p in self.D.parameters():
            p.requires_grad_(False)
        try:
            boost = float(getattr(self.cfg.loss, "region_l1_boost", 1.0))
            nir_w_min = float(getattr(self.cfg.loss, "nir_grad_l1_min", 1.0))
            nir_w_win = int(getattr(self.cfg.loss, "nir_grad_l1_win", 7))
            with amp_ctx:
                if train_d:
                    d_fake_for_g = self.D(nir, fake)
                    g_gan = self.gan_loss(d_fake_for_g, True)
                else:
                    g_gan = torch.zeros((), device=nir.device)
                w_map = None
                if nir_w_min < 1.0:
                    w_map = self._nir_grad_weight(nir, nir_w_min, nir_w_win)
                if boost != 1.0 and region_mask is not None:
                    region_factor = 1.0 + (boost - 1.0) * region_mask
                    w_map = region_factor if w_map is None else w_map * region_factor
                if w_map is not None:
                    g_l1 = ((fake - rgb).abs() * w_map).mean() / w_map.mean()
                else:
                    g_l1 = F.l1_loss(fake, rgb)
                if self.teacher is not None:
                    with torch.no_grad():
                        teacher_out = self.teacher(nir)
                    g_kd_l1 = F.l1_loss(fake, teacher_out)
                else:
                    g_kd_l1 = torch.zeros((), device=nir.device)
                g_perc = self.perceptual(fake, rgb)
                g_msssim = self.msssim_loss(fake, rgb)
                if self.feature_map_loss is not None and feature_map_w > 0.0:
                    g_feature_map = self.feature_map_loss(fake, rgb)
                else:
                    g_feature_map = torch.zeros((), device=nir.device)
                if (
                    self.teacher is not None
                    and self.feature_map_loss is not None
                    and self.kd_feature_weight > 0.0
                    and feature_map_w > 0.0
                ):
                    g_kd_feature = self.feature_map_loss(fake, teacher_out)
                else:
                    g_kd_feature = torch.zeros((), device=nir.device)
                g_loss = (
                    self.cfg.loss.l1_weight * g_l1
                    + self.kd_l1_weight * g_kd_l1
                    + self.cfg.loss.perceptual_weight * g_perc
                    + gan_w * g_gan
                    + self.cfg.loss.msssim_weight * g_msssim
                    + feature_map_w * g_feature_map
                    + self.kd_feature_weight * feature_map_w * g_kd_feature
                )
            g_finite = bool(torch.isfinite(g_loss).item())
            if g_finite:
                if self._accum_step == 0:
                    self.opt_g.zero_grad(set_to_none=True)
                self.scaler_g.scale(g_loss / accum).backward()
                stepped_g = False
                if self._accum_step + 1 == accum:
                    if self.grad_clip > 0:
                        self.scaler_g.unscale_(self.opt_g)
                        torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.grad_clip)
                    self.scaler_g.step(self.opt_g)
                    self.scaler_g.update()
                    stepped_g = True
                if stepped_g:
                    self._ema_update()
        finally:
            for p in self.D.parameters():
                p.requires_grad_(True)

        self._accum_step = (self._accum_step + 1) % accum

        out = {
            "d_loss": float(d_loss.detach()),
            "g_loss": float(g_loss.detach()),
            "g_l1": float(g_l1.detach()),
            "g_kd_l1": float(g_kd_l1.detach()),
            "g_perc": float(g_perc.detach()),
            "g_gan": float(g_gan.detach()),
            "g_msssim": float(g_msssim.detach()),
            "g_feature_map": float(g_feature_map.detach()),
            "g_kd_feature": float(g_kd_feature.detach()),
            "gan_w": gan_w,
            "feature_map_w": feature_map_w,
            "nonfinite": (not d_finite) or (not g_finite),
        }
        # Per-extractor weighted contributions (already include the per-extractor `weight`,
        # so multiplying by `feature_map_w` gives each one's exact share of g_loss).
        if self.feature_map_loss is not None:
            for k, v in getattr(self.feature_map_loss, "last_per_extractor", {}).items():
                out[f"g_fm_{k}"] = float(v) * feature_map_w
        # Weighted contributions of the other loss terms — makes balance visible in TB.
        out["wg_l1"] = self.cfg.loss.l1_weight * out["g_l1"]
        out["wg_kd_l1"] = self.kd_l1_weight * out["g_kd_l1"]
        out["wg_perc"] = self.cfg.loss.perceptual_weight * out["g_perc"]
        out["wg_gan"] = gan_w * out["g_gan"]
        out["wg_msssim"] = self.cfg.loss.msssim_weight * out["g_msssim"]
        out["wg_feature_map"] = feature_map_w * out["g_feature_map"]
        out["wg_kd_feature"] = self.kd_feature_weight * feature_map_w * out["g_kd_feature"]
        return out

    @torch.no_grad()
    def _save_random_val_sample(self, out_path):
        """Run one random val sample through eval_G and dump a NIR/fake/RGB grid."""
        if self.val_loader is None:
            return
        ds = self.val_loader.dataset
        if len(ds) == 0:
            return
        eval_G = self.G_ema if self.G_ema is not None else self.G
        was_training = eval_G.training
        eval_G.eval()
        idx = random.randrange(len(ds))
        sample = ds[idx]
        nir1 = sample["nir"].unsqueeze(0).to(self.device)
        rgb1 = sample["rgb"].unsqueeze(0).to(self.device)
        amp_ctx = torch.amp.autocast("cuda", enabled=self.amp, dtype=self.amp_dtype)
        with amp_ctx:
            fake1 = eval_G(nir1)
        save_grid(nir1, fake1.float(), rgb1, out_path)
        if was_training:
            eval_G.train()

    def train_epoch(self, epoch):
        self.G.train()
        self.D.train()
        if self.feature_map_loss is not None:
            self.feature_map_loss.eval()
        gan_w = self._current_gan_weight(epoch)
        feature_map_w = self._current_feature_map_weight(epoch)
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        ema_loss = None
        nonfinite_count = 0
        self._accum_step = 0  # reset accumulation at epoch boundary
        sample_interval = int(getattr(self.cfg.train, "sample_interval_batches", 0))
        for batch in pbar:
            nir = batch["nir"].to(self.device, non_blocking=True)
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            region_mask = batch.get("region_mask")
            if region_mask is not None:
                region_mask = region_mask.to(self.device, non_blocking=True)
            metrics = self.train_step(nir, rgb, region_mask=region_mask, gan_w=gan_w, feature_map_w=feature_map_w)
            self.global_step += 1
            if metrics["nonfinite"]:
                nonfinite_count += 1
                if nonfinite_count == 1:
                    print(f"[warn] non-finite loss at step {self.global_step} (epoch {epoch}); skipping optimizer update")
            else:
                ema_loss = metrics["g_loss"] if ema_loss is None else 0.95 * ema_loss + 0.05 * metrics["g_loss"]
            if self.global_step % self.cfg.train.log_interval == 0 and not metrics["nonfinite"]:
                for k, v in metrics.items():
                    if k == "nonfinite":
                        continue
                    self.writer.add_scalar(f"train/{k}", v, self.global_step)
            if sample_interval > 0 and self.global_step % sample_interval == 0:
                out = self.run_dir / "in_train_samples" / f"step_{self.global_step:08d}.png"
                self._save_random_val_sample(out)
            ema_str = f"{ema_loss:.3f}" if ema_loss is not None else "nan"
            pbar.set_postfix({"g_loss": ema_str, "d_loss": f"{metrics['d_loss']:.3f}", "skip": nonfinite_count})
        if nonfinite_count > 0:
            print(f"[warn] epoch {epoch}: {nonfinite_count} non-finite batches skipped")
        self._epoch_nonfinite = nonfinite_count

    @torch.no_grad()
    def validate(self, epoch):
        if self.val_loader is None:
            return None
        # Evaluate on EMA weights if available — these are what we'll deploy.
        eval_G = self.G_ema if self.G_ema is not None else self.G
        eval_G.eval()
        prefix = "val_ema" if self.G_ema is not None else "val"
        ssim_sum = psnr_sum = lpips_sum = 0.0
        fm_loss_sum = 0.0
        fm_per_ext_sum = {k: 0.0 for k in (self.feature_map_loss.extractors.keys() if self.feature_map_loss is not None else [])}
        cos_sum = {k: 0.0 for k in fm_per_ext_sum}
        n = 0
        for batch in self.val_loader:
            nir = batch["nir"].to(self.device, non_blocking=True)
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            fake = eval_G(nir)
            B = nir.size(0)
            psnr_sum += compute_psnr(fake, rgb) * B
            ssim_sum += compute_ssim(fake, rgb) * B
            if self.lpips is not None:
                lpips_sum += self.lpips(fake, rgb) * B
            if self.feature_map_loss is not None:
                fm_total = float(self.feature_map_loss(fake, rgb).detach())
                fm_loss_sum += fm_total * B
                for k, v in self.feature_map_loss.last_per_extractor.items():
                    fm_per_ext_sum[k] += v * B
                pred_embs = self.feature_map_loss.pooled_embeddings(fake)
                tgt_embs = self.feature_map_loss.pooled_embeddings(rgb)
                for k, pe in pred_embs.items():
                    cs = F.cosine_similarity(pe.float(), tgt_embs[k].float(), dim=-1).mean()
                    cos_sum[k] += float(cs) * B
            n += B
        if n == 0:
            return None

        metrics = {
            "val/psnr": psnr_sum / n,
            "val/ssim": ssim_sum / n,
        }
        if self.lpips is not None:
            metrics["val/lpips"] = lpips_sum / n
        if self.feature_map_loss is not None:
            metrics["val/feature_map_loss"] = fm_loss_sum / n
            for k in fm_per_ext_sum:
                metrics[f"val/feature_map/{k}"] = fm_per_ext_sum[k] / n
            for k, s in cos_sum.items():
                metrics[f"val/{k}_cos"] = s / n
            # Aliases the user spec'd
            if "clip_vitb32" in cos_sum:
                metrics["val/clip_cos"] = cos_sum["clip_vitb32"] / n
            if "dinov2_vits14" in cos_sum:
                metrics["val/dinov2_cos"] = cos_sum["dinov2_vits14"] / n
        for tag, v in metrics.items():
            self.writer.add_scalar(tag, v, epoch)

        msg = f"[{prefix}] epoch={epoch} psnr={metrics['val/psnr']:.3f} ssim={metrics['val/ssim']:.4f}"
        if "val/lpips" in metrics:
            msg += f" lpips={metrics['val/lpips']:.4f}"
        if "val/feature_map_loss" in metrics:
            msg += f" fm_loss={metrics['val/feature_map_loss']:.4f}"
            if "val/clip_cos" in metrics:
                msg += f" clip_cos={metrics['val/clip_cos']:.4f}"
            if "val/dinov2_cos" in metrics:
                msg += f" dinov2_cos={metrics['val/dinov2_cos']:.4f}"
        print(msg)

        if self.fixed_val_batch is not None:
            nir, rgb = self.fixed_val_batch
            fake = eval_G(nir)
            save_grid(nir, fake, rgb, self.run_dir / "samples" / f"epoch_{epoch:04d}.png")

        self._save_random_val_sample(self.run_dir / "random_samples" / f"epoch_{epoch:04d}.png")
        return metrics

    def fit(self):
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            t0 = time.time()
            print(f"Starting epoch {epoch}...")
            self.train_epoch(epoch)
            if self.lr_sched_g is not None:
                self.lr_sched_g.step()
            print(f"Finished epoch {epoch} training loop. Starting validation...")
            val_metrics = None
            if (epoch + 1) % self.cfg.train.val_interval_epochs == 0:
                val_metrics = self.validate(epoch)
            if self._epoch_nonfinite > 0:
                print(f"[warn] epoch {epoch} had {self._epoch_nonfinite} non-finite batches; skipping checkpoint saves to preserve recoverability")
            else:
                print(f"Finished epoch {epoch} validation. Saving checkpoints...")
                if (epoch + 1) % self.cfg.train.save_interval_epochs == 0:
                    ckpt = self.run_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                    save_checkpoint(ckpt, self.G, self.D, self.opt_g, self.opt_d, epoch, G_ema=self.G_ema)
                    keep_last = int(getattr(self.cfg.train, "keep_last_checkpoints", 0))
                    if keep_last > 0:
                        numbered = sorted((self.run_dir / "checkpoints").glob("epoch_*.pth"))
                        for old in numbered[:-keep_last]:
                            old.unlink()
                save_checkpoint(self.run_dir / "checkpoints" / "latest.pth", self.G, self.D, self.opt_g, self.opt_d, epoch, G_ema=self.G_ema)
                if val_metrics is not None:
                    tag = self.best_metric_name
                    if tag not in val_metrics:
                        print(f"[warn] best_metric={tag!r} not in val metrics {list(val_metrics)}; falling back to val/psnr")
                        tag = "val/psnr"
                    v = val_metrics[tag]
                    improved = v > self.best_metric_value if self.best_metric_mode == "max" else v < self.best_metric_value
                    if improved:
                        self.best_metric_value = v
                        extra = {f"val_{k.replace('val/', '')}": float(val_metrics[k]) for k in val_metrics}
                        extra["best_metric_name"] = tag
                        extra["best_metric_mode"] = self.best_metric_mode
                        save_checkpoint(self.run_dir / "checkpoints" / "best.pth", self.G, self.D, self.opt_g, self.opt_d, epoch, G_ema=self.G_ema, extra=extra)
                        print(f"[best] epoch {epoch} new best {tag} = {v:.4f} ({self.best_metric_mode}) — saved best.pth")
            print(f"epoch {epoch} took {time.time() - t0:.1f}s")
        print("Training finished. Saving final checkpoint...")
        save_checkpoint(self.run_dir / "checkpoints" / "final.pth", self.G, self.D, self.opt_g, self.opt_d, self.cfg.train.epochs - 1, G_ema=self.G_ema)
