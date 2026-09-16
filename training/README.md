# Training an NIR-to-RGB translator

This directory contains the teacher/student models, training recipes, evaluation
code and edge-export tools used in the [final report](../report/main.pdf).
Run the commands below from **`training/`**. Dataset and experiment paths in the
YAML files are relative to that directory.

## Setup

The recorded experiments used Linux and CUDA. Create a Python environment, install
a CUDA-compatible PyTorch/torchvision pair, then install the dependencies:

```bash
cd training
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-eval.txt  # optional downstream evaluators
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For Windows data loading,
set `train.num_workers: 0`; the recorded training setup targets Linux.

The original environment snapshot is
[`experiments/_repro/pip_freeze.txt`](experiments/_repro/pip_freeze.txt).
It records the research machine's full environment, including optional tools;
it is not a minimal installation manifest. Frozen feature encoders download
pretrained weights on first use. Training the full teacher and feature ensemble
requires a CUDA GPU with sufficient memory; reduce the batch size if needed.

## Prepare paired data

Supply spatially aligned NIR/RGB images with matching filename stems:

```text
data/
├── nir/scene_0001.png
└── rgb/scene_0001.png
```

The loaders discover image pairs recursively. The selected configs use three
input/output channels and resize to 256 × 256. Set `data.nir_dir` and
`data.rgb_dir` in a copied config for your own dataset. Keep scene-related frames
together when creating a new evaluation split; `split.scene_regex` supports
grouping filenames. The report's canonical split is 80/10/10 with seed 42;
other experiment configs intentionally use different seeds.

The original 12,536-pair dataset and trained weights are not distributed here.
`data/mp_cache_no_crop.json` is historical metadata for the original dataset,
used by an early ablation. For new data, generate your own cache with
`scripts/precompute_mediapipe_boxes.py`, or set `data.mp_cache: null` as in the
selected teacher and student configs.

```bash
python scripts/check_dataset.py --config configs/ablation_15b_l1boost_split42_s2.yaml
```

## Train the teacher and student

The headline report teacher is the seed-42 split / training-seed-2 run:

```bash
python scripts/train.py --config configs/ablation_15b_l1boost_split42_s2.yaml
```

The deployed student's preserved config references a different parent: the
seed-8065 L1-boost teacher. To follow that recorded distillation lineage:

```bash
python scripts/train.py --config configs/ablation_08_nir_l1boost_gan_msssim_bf16_200_seed8065.yaml
python scripts/train.py --config configs/student_l1b_c_featkd.yaml
```

Alternatively, copy the student config and set `distill.teacher_checkpoint` to
the `best.pth` from your own teacher. Keep `distill.teacher` consistent with that
checkpoint's architecture. This is a new experiment, rather than reproduction
of the released student's exact weights.

The current `scripts/train.py` supports the selected student's output and feature
distillation losses. `scripts/train_distill.py` is the earlier, separate
distillation implementation, retained for the older student recipes.

Outputs go to `experiments/<project.experiment>/`: checkpoints, TensorBoard logs
and sample images. `train.resume: latest` resumes an existing run or starts fresh
if none exists. Use a new `project.experiment` (or `--name`) to start a separate
run; make sure the student's checkpoint path follows the teacher's run name.
For the selected configs, `best.pth` is selected by validation CLIP cosine and
`amp: false` disables AMP. Historical configs retain their original settings.

Other recipes include Pix2Pix baselines, teacher/student ablations, reverse
RGB-to-NIR training, and KAIST/WHU cross-modality experiments. Some older recipes
require warm-start checkpoints or extra datasets. Inspect their `train.resume`,
`data` and `distill` fields before launching them.

## Evaluate

```bash
python scripts/eval.py \
  --config configs/student_l1b_c_featkd.yaml \
  --checkpoint experiments/student_l1b_c_featkd/checkpoints/best.pth --split test

python scripts/eval_downstream.py \
  --config configs/student_l1b_c_featkd.yaml \
  --checkpoint experiments/student_l1b_c_featkd/checkpoints/best.pth --models all

python scripts/harness_unit_tests.py
```

The root `eval_*.py` scripts preserve the multi-model research campaigns; inspect
their model registries and dataset settings before running them. The parameterised
entry points in `scripts/` are the starting point for a new model.

Recorded results remain in [`experiments/`](experiments/):

| Directory | Contents |
| --- | --- |
| `eval_full_v4_canonical/` | Canonical evaluation, per-image results and bootstrap CIs |
| `eval_full_v5_ablations/` | Teacher ablations |
| `eval_full_v5_int8/` | Quantised student evaluation |
| `eval_full_v6_seed8065/` | Evaluation on the ablation split |
| `robustness_v1/` | Controlled input perturbations |
| `dataset_integrity_v1/` | Original dataset integrity measurements |

Earlier evaluation CSVs and run configs provide experiment provenance. New run
outputs, checkpoints, datasets and caches are ignored by Git.

## Export and benchmark

Export reads `checkpoints/best.pth` from the experiment named in the config.
For LiteRT, use a separate compatible export environment with `litert-torch`
and `ai-edge-quantizer`; recorded versions are in the environment snapshot.
Prepare a calibration folder using **only your training split**; the exporter
does not select a split for you.

```bash
python scripts/export_litert.py \
  --config configs/student_l1b_c_featkd.yaml \
  --calib_dir /path/to/training-only/nir-calibration --calib_limit 100

python scripts/export_onnx.py --config configs/student_l1b_c_featkd.yaml
```

LiteRT export writes `model_litert_fp32.tflite` and `model_litert_int8.tflite`
under the experiment directory. The INT8 profile retains floating-point
operations, so it is mixed precision. `export_tflite.py` is an older ONNX-to-TFLite
route with known layout issues; use `export_litert.py` for this architecture.

Copy the exports to a Raspberry Pi with the relevant runtime installed, then run:

```bash
python scripts/benchmark_pi_cpu.py --models_dir /path/to/exported-models
```

The recorded [Pi benchmark script and measurements](../evidence/pi/) accompany
the thesis. Hailo compilation additionally needs the vendor's Dataflow Compiler;
the retained `export_hef.py` is a research helper, and hardware compilation and
performance must be validated on the target device. RKNN and other export helpers
are also retained for related deployment experiments.
