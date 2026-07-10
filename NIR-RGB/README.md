# NIR → RGB translation

Trains an image-to-image translator that converts **NIR** (near-infrared,
3-channel, 850 nm long-pass) images into **RGB-looking** images, so that
off-the-shelf RGB-trained models (detection, segmentation, depth, CLIP/DINOv2
embeddings) work on NIR input without retraining. A small **student** model is
distilled from a NAFNet teacher and quantised for real-time edge deployment
(Raspberry Pi 5 CPU and Pi AI HAT+ / Hailo-8L NPU).

This repository accompanies the MEng final-year project report
*Edge-Deployable NIR-to-RGB Translation as a Pre-Processor for Pre-Trained
Vision Models* (Samuel Barber, Imperial College London, 2026). The report is
the authoritative description of the method and results.

Key entry points:

- Report teacher: `configs/ablation_15b_l1boost_split42_s2.yaml`
- Selected deployment student: `configs/student_l1b_c_featkd.yaml`
  (see `experiments/student_l1b_c_featkd/README.md`)
- Headline evaluation outputs: `experiments/eval_full_v4_canonical/`
  (canonical split), `eval_full_v5_int8/` (quantised student),
  `eval_full_v6_seed8065/` (ablation split), `robustness_v1/`

Checkpoints and exported artefacts (ONNX/LiteRT/HEF) are gitignored; they are
stored in Weights & Biases and available to the supervisor and examiners on
request.

## Repository layout

```
NIR-RGB/
├── src/             ← library code: data/, models/, training/, eval/, utils/
├── scripts/         ← entry points (train, eval, export, benchmark)
├── configs/         ← one YAML per experiment (+ _archive/ superseded)
├── experiments/     ← per-run outputs and evaluation campaigns
├── docs/            ← interim progress report + figures
├── eval_*.py        ← cross-model evaluation drivers
└── run-*.sh         ← launch wrappers (nohup-friendly) for each run
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
# install CUDA-matched torch first (https://pytorch.org/get-started/locally/)
pip install -r requirements.txt
pip install -r requirements-eval.txt   # optional downstream-eval deps
```

The pinned environment used for the report's numbers is
`experiments/_repro/pip_freeze.txt`. LiteRT export uses Google's
`litert-torch`; Hailo compilation uses the Hailo Dataflow Compiler 3.33.1
(vendor Docker image).

## Dataset layout

Paired images matched by **filename stem** (extension may differ):

```
data/
  nir/  ...               ← aligned 256×256 NIR frames
  rgb/  ...               ← matching RGB frames
  kaist-nir/ kaist-rgb/   ← KAIST thermal/RGB (cross-modality)
```

The 80/10/10 train/val/test split is derived deterministically from seed 42.
The capture rig, alignment pipeline and privacy measures are described in the
report; the dataset itself is not published.

## Usage

```bash
# sanity-check a config's data loads
python scripts/check_dataset.py --config configs/ablation_15b_l1boost_split42_s2.yaml

# train the report teacher
python scripts/train.py --config configs/ablation_15b_l1boost_split42_s2.yaml

# distil the selected student from it
python scripts/train.py --config configs/student_l1b_c_featkd.yaml

# evaluate one checkpoint (PSNR/SSIM/LPIPS)
python scripts/eval.py --config <cfg> --checkpoint <ckpt> --split test

# downstream gap-closure harness
python scripts/eval_downstream.py --config <cfg> --checkpoint <ckpt> --models all

# verify the harness itself (10 oracle/edge-case fixtures)
python scripts/harness_unit_tests.py
```

Outputs land in `experiments/<experiment>/`: `tb/` (TensorBoard),
`checkpoints/{best,latest}.pth`, `samples/epoch_XXXX.png`
(NIR | predicted | ground-truth). The `run-*.sh` wrappers launch each run
under `nohup` and tee to `experiments/<exp>/train.log`.

## Edge deployment

```bash
# export the student to LiteRT (the deployed artefact) and ONNX
python scripts/export_litert.py --config configs/student_l1b_c_featkd.yaml
python scripts/export_onnx.py   --config configs/student_l1b_c_featkd.yaml

# Raspberry Pi 5 CPU benchmark (on the Pi)
python3 scripts/benchmark_pi_cpu.py --models_dir experiments/student_l1b_c_featkd

# compile and benchmark for the Hailo-8L NPU (Pi AI HAT+)
python scripts/export_hef.py --config configs/student_l1b_c_featkd.yaml
hailortcli run student_small_nir_to_rgb.hef --measure-latency
```

Quantisation is post-training (100 calibration frames from the training
split); PixelShuffle and LayerNorm reductions stay FP32, so the artefacts are
mixed-INT8.

## Notes

- **`amp: false` everywhere** — fp16 AMP caused NaN collapse on multiple runs;
  the frozen feature extractors are also more stable in fp32.
- **`best.pth` is selected by `val/clip_cos`**, not val loss — pixel loss does
  not track downstream usefulness.
- `experiments/_archive/` and `logs/` hold superseded runs/logs and are
  gitignored.
