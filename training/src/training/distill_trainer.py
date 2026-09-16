import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models import build_generator
from ..eval.metrics import compute_psnr, compute_ssim, LPIPSWrapper
from .losses import VGGPerceptualLoss, MSSSIMLoss
from .checkpoint import load_checkpoint, save_grid


def _save_student_ckpt(path, student, opt, epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "G": student.state_dict(), "opt_g": opt.state_dict()},
        str(path),
    )


def _load_student_ckpt(path, student, opt=None, map_location="cpu"):
    state = torch.load(str(path), map_location=map_location)
    student.load_state_dict(state["G"])
    if opt is not None and "opt_g" in state:
        opt.load_state_dict(state["opt_g"])
    return int(state.get("epoch", -1))


class DistillTrainer:
    """Distill a frozen teacher generator into a smaller student.

    Loss = w_kd_pix  * L1(student, teacher.detach())     # match teacher
         + w_gt_l1   * L1(student, rgb_gt)               # don't drift from GT
         + w_perc    * VGG(student, rgb_gt)              # perceptual anchor on GT
         + w_msssim  * (1 - MS-SSIM(student, rgb_gt))    # structural
         + w_kd_perc * VGG(student, teacher.detach())    # optional perceptual KD

    No discriminator on the student — the teacher's output already encodes the
    GAN's contribution, and adding another adversarial loop on a small student
    usually destabilises distillation.
    """

    def __init__(self, cfg, train_loader, val_loader, run_dir, device):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = Path(run_dir)
        self.device = device

        # ---- teacher ----
        teacher_cfg = SimpleNamespace(model=SimpleNamespace(generator=cfg.distill.teacher))
        self.teacher = build_generator(teacher_cfg, cfg.data.in_channels, cfg.data.out_channels).to(device)
        load_checkpoint(cfg.distill.teacher_checkpoint, self.teacher, D=None, map_location=device)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # ---- student ----
        self.student = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels).to(device)

        self.perceptual = VGGPerceptualLoss(tuple(cfg.loss.perceptual_layers)).to(device)
        self.msssim_loss = MSSSIMLoss().to(device)

        self.opt = torch.optim.Adam(
            self.student.parameters(),
            lr=cfg.optim.generator.lr,
            betas=tuple(cfg.optim.generator.betas),
        )

        self.amp = bool(cfg.train.amp) and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.grad_clip = float(getattr(cfg.train, "grad_clip", 0.0))
        self._epoch_nonfinite = 0

        self.lpips = LPIPSWrapper(net=cfg.eval.lpips_net).to(device) if "lpips" in cfg.eval.metrics else None

        self.writer = SummaryWriter(str(self.run_dir / "tb"))
        self.global_step = 0
        self.start_epoch = 0

        if cfg.train.resume:
            resume_path = cfg.train.resume
            if resume_path == "latest":
                resume_path = self.run_dir / "checkpoints" / "latest.pth"
            if Path(resume_path).exists():
                print(f"Resuming student from {resume_path}")
                self.start_epoch = _load_student_ckpt(resume_path, self.student, self.opt) + 1
            elif cfg.train.resume != "latest":
                raise FileNotFoundError(f"Resume checkpoint {resume_path} not found.")

        self.fixed_val_batch = self._collect_fixed_batch()

        n_params = sum(p.numel() for p in self.student.parameters())
        print(f"student params: {n_params/1e6:.2f}M")

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

    def train_step(self, nir, rgb):
        amp_ctx = torch.amp.autocast("cuda", enabled=self.amp)

        with torch.no_grad():
            teacher_out = self.teacher(nir)

        with amp_ctx:
            student_out = self.student(nir)

            kd_pix = F.l1_loss(student_out, teacher_out)
            gt_l1 = F.l1_loss(student_out, rgb)
            perc = self.perceptual(student_out, rgb)
            msssim = self.msssim_loss(student_out, rgb)

            w_kd_perc = float(getattr(self.cfg.loss, "kd_perceptual_weight", 0.0))
            kd_perc = self.perceptual(student_out, teacher_out) if w_kd_perc > 0 else student_out.new_zeros(())

            loss = (
                self.cfg.loss.kd_pixel_weight * kd_pix
                + self.cfg.loss.gt_l1_weight * gt_l1
                + self.cfg.loss.perceptual_weight * perc
                + self.cfg.loss.msssim_weight * msssim
                + w_kd_perc * kd_perc
            )

        finite = bool(torch.isfinite(loss).item())
        if finite:
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

        return {
            "loss": float(loss.detach()),
            "kd_pix": float(kd_pix.detach()),
            "gt_l1": float(gt_l1.detach()),
            "perc": float(perc.detach()),
            "msssim": float(msssim.detach()),
            "kd_perc": float(kd_perc.detach()) if w_kd_perc > 0 else 0.0,
            "nonfinite": not finite,
        }

    def train_epoch(self, epoch):
        self.student.train()
        self.teacher.eval()
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        ema_loss = None
        nonfinite_count = 0
        for batch in pbar:
            nir = batch["nir"].to(self.device, non_blocking=True)
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            metrics = self.train_step(nir, rgb)
            self.global_step += 1
            if metrics["nonfinite"]:
                nonfinite_count += 1
                if nonfinite_count == 1:
                    print(f"[warn] non-finite loss at step {self.global_step} (epoch {epoch}); skipping optimizer update")
            else:
                ema_loss = metrics["loss"] if ema_loss is None else 0.95 * ema_loss + 0.05 * metrics["loss"]
            if self.global_step % self.cfg.train.log_interval == 0 and not metrics["nonfinite"]:
                for k, v in metrics.items():
                    if k == "nonfinite":
                        continue
                    self.writer.add_scalar(f"train/{k}", v, self.global_step)
            ema_str = f"{ema_loss:.3f}" if ema_loss is not None else "nan"
            pbar.set_postfix({"loss": ema_str, "skip": nonfinite_count})
        if nonfinite_count > 0:
            print(f"[warn] epoch {epoch}: {nonfinite_count} non-finite batches skipped")
        self._epoch_nonfinite = nonfinite_count

    @torch.no_grad()
    def validate(self, epoch):
        if self.val_loader is None:
            return
        self.student.eval()
        ssim_sum = psnr_sum = lpips_sum = 0.0
        ssim_t_sum = psnr_t_sum = 0.0
        n = 0
        for batch in self.val_loader:
            nir = batch["nir"].to(self.device, non_blocking=True)
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            student_out = self.student(nir)
            teacher_out = self.teacher(nir)
            psnr_sum += compute_psnr(student_out, rgb) * nir.size(0)
            ssim_sum += compute_ssim(student_out, rgb) * nir.size(0)
            psnr_t_sum += compute_psnr(student_out, teacher_out) * nir.size(0)
            ssim_t_sum += compute_ssim(student_out, teacher_out) * nir.size(0)
            if self.lpips is not None:
                lpips_sum += self.lpips(student_out, rgb) * nir.size(0)
            n += nir.size(0)
        if n == 0:
            return
        psnr = psnr_sum / n
        ssim = ssim_sum / n
        psnr_t = psnr_t_sum / n
        ssim_t = ssim_t_sum / n
        self.writer.add_scalar("val/psnr_gt", psnr, epoch)
        self.writer.add_scalar("val/ssim_gt", ssim, epoch)
        self.writer.add_scalar("val/psnr_vs_teacher", psnr_t, epoch)
        self.writer.add_scalar("val/ssim_vs_teacher", ssim_t, epoch)
        msg = f"[val] epoch={epoch} psnr_gt={psnr:.3f} ssim_gt={ssim:.4f} psnr_t={psnr_t:.3f} ssim_t={ssim_t:.4f}"
        if self.lpips is not None:
            lpips_v = lpips_sum / n
            self.writer.add_scalar("val/lpips_gt", lpips_v, epoch)
            msg += f" lpips_gt={lpips_v:.4f}"
        print(msg)

        if self.fixed_val_batch is not None:
            nir, rgb = self.fixed_val_batch
            student_out = self.student(nir)
            save_grid(nir, student_out, rgb, self.run_dir / "samples" / f"epoch_{epoch:04d}.png")

        ds = self.val_loader.dataset
        if len(ds) > 0:
            idx = random.randrange(len(ds))
            sample = ds[idx]
            nir1 = sample["nir"].unsqueeze(0).to(self.device)
            rgb1 = sample["rgb"].unsqueeze(0).to(self.device)
            student_out1 = self.student(nir1)
            save_grid(nir1, student_out1, rgb1,
                      self.run_dir / "random_samples" / f"epoch_{epoch:04d}.png")

    def fit(self):
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            t0 = time.time()
            print(f"Starting epoch {epoch}...")
            self.train_epoch(epoch)
            print(f"Finished epoch {epoch} training loop. Starting validation...")
            if (epoch + 1) % self.cfg.train.val_interval_epochs == 0:
                self.validate(epoch)
            if self._epoch_nonfinite > 0:
                print(f"[warn] epoch {epoch} had {self._epoch_nonfinite} non-finite batches; skipping checkpoint saves")
            else:
                print(f"Finished epoch {epoch} validation. Saving checkpoints...")
                if (epoch + 1) % self.cfg.train.save_interval_epochs == 0:
                    ckpt = self.run_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
                    _save_student_ckpt(ckpt, self.student, self.opt, epoch)
                    keep_last = int(getattr(self.cfg.train, "keep_last_checkpoints", 0))
                    if keep_last > 0:
                        numbered = sorted((self.run_dir / "checkpoints").glob("epoch_*.pth"))
                        for old in numbered[:-keep_last]:
                            old.unlink()
                _save_student_ckpt(self.run_dir / "checkpoints" / "latest.pth", self.student, self.opt, epoch)
            print(f"epoch {epoch} took {time.time() - t0:.1f}s")
        print("Training finished. Saving final checkpoint...")
        _save_student_ckpt(self.run_dir / "checkpoints" / "final.pth", self.student, self.opt, self.cfg.train.epochs - 1)
