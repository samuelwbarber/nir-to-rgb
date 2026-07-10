<div align="center">

# Edge-Deployable NIR→RGB Translation as a Pre-Processor for Pre-Trained Vision Models

**MEng Final-Year Project · Imperial College London · 2026 · First Class Honours**

Samuel Barber · Supervisor: Prof. Cong Ling

[![Report](https://img.shields.io/badge/report-main.pdf-b31b1b)](main.pdf)
[![Paper](https://img.shields.io/badge/paper-WACV%20draft-1f6feb)](paper/wacv2027_draft.pdf)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](LICENSE.md)
![Python](https://img.shields.io/badge/python-3.11-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-teacher%2Fstudent-ee4c2c)
![Edge](https://img.shields.io/badge/deploy-Raspberry%20Pi%205%20%2B%20Hailo--8L-c51a4a)

</div>

---

## TL;DR

Modern vision models are trained almost entirely on visible-band **RGB** imagery and
degrade badly on **near-infrared (NIR)** input. Retraining every downstream model on NIR
is impractical. This project instead builds a **translator that sits between the NIR sensor
and an unmodified, frozen RGB model** — and judges it by a stricter criterion than image
quality: *how much of each frozen model's native-RGB behaviour the translation recovers.*

A 116 M-parameter NAFNet **teacher**, trained with feature-matching losses against frozen
foundation models, is distilled into a **1.7 M-parameter student (67.3× smaller)** and
exported as a deterministic **2.3 MB mixed-INT8 LiteRT** artefact that runs in real time on
a Raspberry Pi 5.

## Headline results

| | Result |
|---|---|
| **Dataset** | 12,536 simultaneous, homography-aligned NIR/RGB pairs, captured across London on a custom shared-aperture beam-splitter rig |
| **Teacher** | 116 M-param NAFNet, feature-matching losses vs. frozen **DINOv2 + CLIP + ResNet-50** (in place of a single perceptual term) |
| **Student** | 1.7 M params — a **67.3× reduction** — exported to a **2.3 MB mixed-INT8 LiteRT** artefact |
| **Downstream gap closure** | **Positive on all 5 primary tasks** spanning detection, segmentation, and depth |
| **Latency** | **5.05 FPS** on Raspberry Pi 5 CPU · **27.2 FPS** on Hailo-8L NPU (target: ≥ 5 FPS) |
| **Key finding** | A Pix2Pix baseline matching the student on **PSNR/SSIM recovers almost none** of the downstream behaviour — feature-space supervision and behaviour-centred evaluation, *not* pixel fidelity, are what make a translation pre-processor useful |

## The capture rig

Supervising a translator needs NIR and RGB of the **same scene at the same instant**. A
custom, hand-built rig splits one optical path with a beam-splitter cube onto two
Raspberry Pi cameras — one behind an 850 nm long-pass filter (NIR), one unfiltered (RGB) —
all inside a 3D-printed case. Carried around London, it captured **12,536** simultaneous
pairs.

<table>
<tr>
<td width="50%"><img src="imgs/device-labeled.png" alt="Annotated internals of the capture rig" width="100%"></td>
<td width="50%"><img src="imgs/rig-deployed.jpg" alt="The rig deployed on a backpack during a capture walk" width="100%"></td>
</tr>
<tr>
<td align="center"><em>Internals: beam-splitter cube, 850 nm NIR camera, RGB camera, 3D-printed case.</em></td>
<td align="center"><em>Deployed in the field — capturing paired data on the move around London.</em></td>
</tr>
</table>

After capture, pairs are blur-filtered and **SIFT/RANSAC homography-aligned** so NIR and RGB
share a pixel grid:

<div align="center">
<img src="NIR-RGB/docs/figures/dataset_pixel_pairs.png" alt="Aligned NIR/RGB training pairs, left half NIR and right half RGB" width="90%">
</div>

## Repository layout

```
final-report/
├── main.tex, chapters/, references.bib   ← the thesis (LaTeX source)
├── main.pdf                              ← compiled report (authoritative)
├── ic_eee_thesis.cls                     ← Imperial EEE thesis class
│
├── NIR-RGB/                              ← the codebase
│   ├── src/         library: data · models (NAFNet/UNet/discriminator) · training · eval
│   ├── scripts/     entry points: train · eval · export · benchmark
│   ├── configs/     one YAML per experiment (teacher, students, ablations)
│   ├── experiments/ per-run outputs + evaluation campaigns
│   ├── eval_*.py    cross-model downstream-eval drivers
│   └── run-*.sh     launch wrappers for each training/eval run
│
├── paper/                               ← condensed WACV-style paper draft
├── Presentation_Ready/                  ← viva slides + live Pi demo video
├── figtools/                            ← figure-generation scripts
├── imgs/                                ← report figures
├── pi_evidence/                         ← on-device deployment evidence
├── hallucinator-bin/                    ← packaged demo CLI
└── *_demo.html                          ← interactive loss / PSNR / SSIM explainers
```

## The idea in one diagram

```
        NIR sensor          ┌───────────────────────┐        Frozen, unmodified
   (850 nm long-pass)  ──▶  │  NIR→RGB translator   │  ──▶   RGB vision model
                            │  (1.7 M student, INT8)│        (YOLO / seg / depth /
                            └───────────────────────┘         CLIP / DINOv2)
   No colour, wrong           Trained to preserve            Works as if it were
   material appearance        *downstream behaviour*         looking at real RGB
```

## Results

**Translation quality** — raw NIR → translated RGB → ground-truth RGB (per-pair PSNR shown):

<div align="center">
<img src="imgs/grid_22.png" alt="Raw NIR, l1boost translation, and ground-truth RGB across three street scenes" width="80%">
</div>

**What actually matters** — how much of each *frozen* RGB model's native behaviour the
translation recovers. Blue (translated RGB) beats grey (raw NIR) on detection (YOLO,
Mask R-CNN), segmentation (DeepLab), depth (MiDaS), and embeddings (ResNet-50):

<div align="center">
<img src="NIR-RGB/docs/figures/downstream_gap_closure.png" alt="Downstream gap closure: translated RGB vs raw NIR across eight frozen models" width="85%">
</div>

## Method, briefly

1. **Data.** A Raspberry-Pi-based beam-splitter rig captures NIR and RGB through a shared
   aperture at the same instant, then homography-aligns each pair — 12,536 pairs total.
2. **Teacher.** A 116 M NAFNet is trained with **feature-matching losses** against frozen
   DINOv2, CLIP, and ResNet-50, so the translation is optimised for what downstream models
   *see*, not for pixel PSNR.
3. **Distillation.** The teacher is distilled into a 1.7 M student and quantised to a
   deterministic mixed-INT8 LiteRT export for the edge.
4. **Deployment.** The student runs at 5.05 FPS on the Pi 5 CPU and 27.2 FPS on the
   Hailo-8L NPU, meeting the real-time target.
5. **Evaluation.** Every model is scored by **gap closure** — the fraction of each frozen
   model's native-RGB performance recovered on translated NIR — across detection,
   segmentation, and depth.

## Reproducing the code

```bash
cd NIR-RGB
python -m venv .venv && source .venv/bin/activate
# install a CUDA-matched torch first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install -r requirements-eval.txt      # optional downstream-eval deps
```

Key entry points (see [`NIR-RGB/README.md`](NIR-RGB/README.md) for full detail):

- **Report teacher** — `configs/ablation_15b_l1boost_split42_s2.yaml`
- **Deployed student** — `configs/student_l1b_c_featkd.yaml`
- **Headline eval** — `experiments/eval_full_v4_canonical/` (canonical split),
  `eval_full_v5_int8/` (quantised student), `robustness_v1/`

> **Note.** Trained checkpoints and exported artefacts (ONNX/LiteRT/HEF) and the raw image
> dataset are intentionally **not** in this repo — they live in Weights & Biases and are
> available on request. The compiled `main.pdf` is the authoritative description of the
> method and results.

## Read the work

- 📄 **[main.pdf](main.pdf)** — the full thesis
- 📝 **[paper/wacv2027_draft.pdf](paper/wacv2027_draft.pdf)** — condensed paper version
- 🎞️ **[Presentation_Ready/](Presentation_Ready/)** — viva slides and a live on-Pi demo video

## License

The written report, figures, and this documentation are licensed under
[**CC BY-NC 4.0**](LICENSE.md). Please attribute and do not use commercially.
