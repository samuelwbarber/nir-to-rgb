# Final Report Evidence Workbook

Use this file as the single checklist for collecting and recording the evidence
needed to finish the report. Fill every blank with a measured fact, an artifact
path, or `N/A - reason`. Do not write a result into the report unless its raw
artifact and provenance are recorded here.

## Rules

- [ ] I have not estimated, copied from an old run, or invented any value.
- [ ] Every headline table uses the same canonical test manifest.
- [ ] Every model is evaluated with the checkpoint named in the model register.
- [ ] All test-set results are from inference-only runs; the test set was not
      used for training, checkpoint selection, calibration, or method selection.
- [ ] INT8 calibration images come from the training split only.
- [ ] Every aggregate result retains per-image or per-group raw results.
- [ ] Higher-is-better and lower-is-better metrics are handled correctly.
- [ ] Native-RGB model outputs are described as a behavioural reference or
      pseudo-ground truth, not human-labelled task ground truth.
- [ ] Any failed, missing, excluded, or incomplete experiment is stated openly.
- [ ] The abstract is updated only after all headline results are frozen.

---

# 1. Protocol Freeze

Complete this section before collecting final results. The current repository
contains incompatible protocols: seed 8065 versus seed 42, image-level versus
grouped splits, full-test versus first-100/first-200 subsets, and multiple
teacher checkpoints.

## 1.1 Canonical Dataset Protocol

| Field | Fill in |
|---|---|
| Dataset version/name | `________` |
| Dataset root | `________` |
| Total discovered matched pairs | `________` |
| Manifest creation date | `________` |
| Manifest-generation command | `________` |
| Pairing key | `________` |
| Stored image size/channels | `________` |
| Split method: image-level or grouped | `________` |
| Group/scene identifier or regex | `________` |
| Split seed | `________` |
| Train count | `________` |
| Validation count | `________` |
| Test count | `________` |
| Train manifest path | `________` |
| Validation manifest path | `________` |
| Test manifest path | `________` |
| Test manifest SHA-256 | `________` |
| Proof no group crosses a split | `________` |

Checklist:

- [ ] The split described above matches Chapters 4, 5, and 6.
- [ ] If grouped splitting is claimed, `scene_regex: null` is not used.
- [ ] All final evaluators consume the saved manifest rather than independently
      reshuffling file paths.
- [ ] The manifest records image stem, NIR path, RGB path, group ID, split, and
      content hashes.
- [ ] Test images are absent from INT8 calibration data.

Decision box:

> Final split decision and justification:
>
> `________________________________________________________________________`
>
> `________________________________________________________________________`

## 1.2 Canonical Evaluation Conditions

| Field | Fill in |
|---|---|
| Input resolution | `________` |
| Test sample count | `________` |
| Full test set or fixed subset | `________` |
| Subset selection seed, if applicable | `________` |
| Subset manifest path | `________` |
| Image preprocessing | `________` |
| Colour/channel convention | `________` |
| Tensor value range | `________` |
| Crop/resize policy | `________` |
| Bootstrap unit: image or capture group | `________` |
| Bootstrap resamples | `________` |
| Bootstrap seed | `________` |
| Confidence level | `________` |
| Evaluation code commit | `________` |

- [ ] The same samples and order are used for raw NIR, translated RGB, native
      RGB, every model variant, FP32, and INT8.
- [ ] The qualitative examples are selected by a declared rule.
- [ ] The final protocol does not silently use `first 100` or `first 200`.

## 1.3 Final Evaluator Panel

Choose one panel and make Chapters 1, 3, 5, 6, and 7 agree.

| Task | Frozen model and exact weights | Version/source | Primary agreement metric | Trained in loss? | Keep? |
|---|---|---|---|---|---|
| Detection 1 | `________` | `________` | `________` | `Yes / No` | `[ ]` |
| Detection 2 | `________` | `________` | `________` | `Yes / No` | `[ ]` |
| Segmentation 1 | `________` | `________` | `________` | `Yes / No` | `[ ]` |
| Segmentation 2 | `________` | `________` | `________` | `Yes / No` | `[ ]` |
| Depth | `________` | `________` | `________` | `Yes / No` | `[ ]` |
| Embedding 1 | `________` | `________` | cosine similarity | `Yes / No` | `[ ]` |
| Embedding 2 | `________` | `________` | cosine similarity | `Yes / No` | `[ ]` |
| Embedding 3 | `________` | `________` | cosine similarity | `Yes / No` | `[ ]` |
| Optional classifier | `________` | `________` | `________` | `Yes / No` | `[ ]` |

Important decisions:

- [ ] At least one principal evaluator is independent of the training loss.
- [ ] DINOv2/CLIP results are identified as partially non-independent if those
      same backbones supplied training losses or checkpoint selection.
- [ ] The number of success-criterion tasks is stated consistently.
- [ ] The report does not say five tasks if the final panel defines a different
      number.

## 1.4 Gap-Closure Definition

For higher-is-better metrics:

`gap_closure = (translated - raw_nir) / (native_rgb - raw_nir)`

For lower-is-better error metrics:

`gap_closure = (raw_nir - translated) / (raw_nir - native_rgb)`

Fill in:

| Item | Fill in |
|---|---|
| Denominator near-zero threshold | `________` |
| Behaviour when denominator is near zero | `________` |
| Behaviour when native RGB is worse than raw NIR | `________` |
| Whether values above 1 are clipped | `Yes / No` |
| Whether negative values are retained | `Yes / No` |
| Unit tests path | `________` |

- [ ] Identity translated = raw NIR returns 0.
- [ ] Identity translated = native RGB returns 1.
- [ ] Worse-than-NIR returns a negative value.
- [ ] Better-than-RGB can return greater than 1 and is explained.
- [ ] Undefined denominators are reported as undefined, not zero.

---

# 2. Evidence Register

Create one row for every final artifact. Add rows as needed.

| ID | Evidence | Status | Raw artifact path | Command/log path | Report destination |
|---|---|---|---|---|---|
| E01 | Canonical split manifests | `[ ]` | `________` | `________` | Ch. 4-6 |
| E02 | Dataset integrity audit | `[ ]` | `________` | `________` | Ch. 5 |
| E03 | Alignment statistics | `[ ]` | `________` | `________` | Ch. 5 |
| E04 | Manual alignment audit | `[ ]` | `________` | `________` | Ch. 5 |
| E05 | Privacy audit | `[ ]` | `________` | `________` | Ch. 4/App. A |
| E06 | Teacher training provenance | `[ ]` | `________` | `________` | Ch. 4 |
| E07 | Student training provenance | `[ ]` | `________` | `________` | Ch. 4 |
| E08 | Image-quality table | `[ ]` | `________` | `________` | Ch. 6 |
| E09 | Embedding metrics | `[ ]` | `________` | `________` | Ch. 6 |
| E10 | Detection agreement | `[ ]` | `________` | `________` | Ch. 6 |
| E11 | Segmentation agreement | `[ ]` | `________` | `________` | Ch. 6 |
| E12 | Depth agreement | `[ ]` | `________` | `________` | Ch. 6 |
| E13 | Gap-closure summary | `[ ]` | `________` | `________` | Ch. 6/7 |
| E14 | Teacher ablation | `[ ]` | `________` | `________` | Ch. 6 |
| E15 | Student ablation | `[ ]` | `________` | `________` | Ch. 6 |
| E16 | Export equivalence | `[ ]` | `________` | `________` | Ch. 5/6 |
| E17 | FP32 versus INT8 quality | `[ ]` | `________` | `________` | Ch. 6 |
| E18 | Pi runtime benchmark | `[ ]` | `________` | `________` | Ch. 6 |
| E19 | Sustained thermal/memory test | `[ ]` | `________` | `________` | Ch. 5/6 |
| E20 | Robustness sweep | `[ ]` | `________` | `________` | Ch. 5/7 |
| E21 | Qualitative typical cases | `[ ]` | `________` | `________` | Ch. 4/6 |
| E22 | Qualitative failure cases | `[ ]` | `________` | `________` | Ch. 6/7 |
| E23 | Reproduction environment | `[ ]` | `________` | `________` | App. B |
| E24 | Repository/commit/artifact hashes | `[ ]` | `________` | `________` | App. B |
| E25 | Ethical/legal/safety evidence | `[ ]` | `________` | `________` | App. A |
| E26 | Optional thermal/SAR extension | `[ ]` | `________` | `________` | Ch. 6/7 |

---

# 3. Model and Artifact Register

Do not evaluate a model until this row is complete.

| Role | Run name | Config | Checkpoint/export | Epoch | Selection metric/value | SHA-256 | Params |
|---|---|---|---|---:|---|---|---:|
| Pix2Pix baseline | `________` | `________` | `________` | `___` | `________` | `________` | `________` |
| NAFNet baseline | `________` | `________` | `________` | `___` | `________` | `________` | `________` |
| Final teacher | `________` | `________` | `________` | `___` | `________` | `________` | `________` |
| Student baseline KD | `________` | `________` | `________` | `___` | `________` | `________` | `________` |
| Student downstream-heads | `________` | `________` | `________` | `___` | `________` | `________` | `________` |
| Student feature KD | `________` | `________` | `________` | `___` | `________` | `________` | `________` |
| Selected FP32 student | `________` | `________` | `________` | `___` | `________` | `________` | `________` |
| Selected INT8 student | `________` | `________` | `________` | `N/A` | `calibration: ________` | `________` | `________` |

Verified architecture facts currently obtainable from the repository:

| Model config | Expected measured parameters | Confirmed in final run? |
|---|---:|---|
| NAFNet-64 `[2,2,4,8]/12/[2,2,2,2]` | 115,982,915 | `[ ]` |
| NAFNet-16 `[1,1,1,2]/2/[1,1,1,1]` | 1,723,315 | `[ ]` |
| Pix2Pix U-Net base width 64 | 54,414,531 | `[ ]` |

Parameter-measurement command:

```powershell
@'
from src.utils.config import load_config
from src.models import build_generator
cfg, _ = load_config("CONFIG_PATH")
m = build_generator(cfg, cfg.data.in_channels, cfg.data.out_channels)
print(sum(p.numel() for p in m.parameters()))
'@ | python -
```

---

# 4. Dataset Evidence

## 4.1 Dataset Counts and Integrity

| Measurement | Result |
|---|---:|
| NIR files discovered | `________` |
| RGB files discovered | `________` |
| Matched stems | `________` |
| NIR-only stems | `________` |
| RGB-only stems | `________` |
| Unreadable NIR images | `________` |
| Unreadable RGB images | `________` |
| Wrong dimensions | `________` |
| Wrong channel count | `________` |
| Duplicate paths | `________` |
| Duplicate image hashes | `________` |
| Final accepted pairs | `________` |

| Evidence detail | Fill in |
|---|---|
| Integrity script/command | `________` |
| Raw CSV/JSON path | `________` |
| Run date | `________` |
| Dataset manifest hash | `________` |
| Rejected-file list path | `________` |

- [ ] Every rejected item has a reason.
- [ ] Reported total `12,536` is confirmed or corrected everywhere.
- [ ] Train/validation/test totals sum to accepted pairs.

## 4.2 Capture Campaign Evidence

| Fact | Evidence/value |
|---|---|
| Total capture duration | `________` |
| Number of capture sessions | `________` |
| Capture date range | `________` |
| Approximate route/area coverage | `________` |
| Capture interval | `________` |
| Raw pair count before filtering | `________` |
| Pairs rejected for blur | `________` |
| Pairs rejected for alignment | `________` |
| Pairs rejected for other reasons | `________` |
| Final retention rate | `________` |
| Hardware bill/list path | `________` |
| Rig photo/CAD evidence path | `________` |

## 4.3 Sharpness Filter Evidence

| Item | Fill in |
|---|---|
| Sharpness metric | `________` |
| Threshold | `________` |
| Number below threshold | `________` |
| Number above threshold | `________` |
| Threshold-selection sample size | `________` |
| Distribution figure path | `________` |
| Example accepted images path | `________` |
| Example rejected images path | `________` |

> Why this threshold is defensible:
>
> `________________________________________________________________________`

## 4.4 Alignment Statistics

Collect statistics over all retained pairs, not only one illustrative image.

| Statistic | Mean | Median | P5 | P95 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Accepted SIFT matches | `___` | `___` | `___` | `___` | `___` | `___` |
| RANSAC inliers | `___` | `___` | `___` | `___` | `___` | `___` |
| Inlier ratio | `___` | `___` | `___` | `___` | `___` | `___` |
| Reprojection error, px | `___` | `___` | `___` | `___` | `___` | `___` |
| Valid-overlap fraction | `___` | `___` | `___` | `___` | `___` | `___` |

| Acceptance rule | Fill in |
|---|---|
| Minimum matches | `________` |
| Minimum inliers | `________` |
| Minimum inlier ratio | `________` |
| Maximum reprojection error | `________` |
| Degenerate-homography checks | `________` |
| Rejected alignment count | `________` |
| Statistics CSV path | `________` |
| Histogram/CDF figure path | `________` |

## 4.5 Manual Alignment Audit

| Item | Fill in |
|---|---|
| Random seed | `________` |
| Audit sample size | `________` |
| Reviewer(s) | `________` |
| Audit display method | `edge overlay / split view / other: ________` |
| Pass definition | `________` |
| Pass count | `________` |
| Minor-error count | `________` |
| Fail count | `________` |
| Most common failure | `________` |
| Audit sheet path | `________` |
| Example failure figure path | `________` |

Manual audit table:

| Stem | Pass/minor/fail | Error type | Approx. offset | Notes |
|---|---|---|---:|---|
| `________` | `________` | `________` | `___ px` | `________` |
| `________` | `________` | `________` | `___ px` | `________` |
| `________` | `________` | `________` | `___ px` | `________` |

Conclusion for success criterion S1:

- [ ] PASS
- [ ] PARTIAL
- [ ] FAIL

Reason:

`____________________________________________________________________________`

## 4.6 Privacy and Anonymisation Audit

| Item | Fill in |
|---|---|
| Face detector/version | `________` |
| Number-plate detector/version | `________` |
| Detection confidence thresholds | `________` |
| Blur method/kernel | `________` |
| Audit sample size | `________` |
| Images with visible unblurred faces | `________` |
| Images with visible unblurred plates | `________` |
| False-positive over-blurs observed | `________` |
| Raw-frame retention/deletion policy | `________` |
| Audit artifact path | `________` |

- [ ] The report does not claim “no identifiable data remained” unless the
      manual audit supports that exact statement.
- [ ] Limitations of automatic anonymisation are acknowledged.

---

# 5. Training Evidence

## 5.1 Final Teacher Provenance

| Field | Fill in |
|---|---|
| Final run name | `________` |
| Config path | `________` |
| Config SHA-256 | `________` |
| Checkpoint path | `________` |
| Checkpoint SHA-256 | `________` |
| Git commit | `________` |
| Dataset manifest/hash | `________` |
| Split seed/grouping | `________` |
| Training seed | `________` |
| Start/end date and time | `________` |
| Wall-clock duration | `________` |
| GPU model/count | `________` |
| GPU memory | `________` |
| Cloud/workstation type | `________` |
| Total epochs completed | `________` |
| Best epoch | `________` |
| Checkpoint-selection metric | `________` |
| Best selection value | `________` |
| Training log path | `________` |
| TensorBoard/W&B artifact path | `________` |

Final/best validation values:

| Metric | Best-checkpoint value | Final-epoch value |
|---|---:|---:|
| Generator loss | `________` | `________` |
| PSNR, dB | `________` | `________` |
| SSIM | `________` | `________` |
| LPIPS | `________` | `________` |
| DINOv2 cosine | `________` | `________` |
| CLIP cosine | `________` | `________` |
| ResNet-50 cosine | `________` | `________` |
| Discriminator loss | `________` | `________` |

Loss weights:

| Term | Final weight | Warm-up |
|---|---:|---|
| L1 | `________` | `________` |
| MS-SSIM | `________` | `________` |
| GAN | `________` | `________` |
| DINOv2 | `________` | `________` |
| CLIP | `________` | `________` |
| ResNet-50 | `________` | `________` |

- [ ] The final run is genuinely seed 42 if the report calls it seed 42.
- [ ] The training-curve figure is generated from this run, not an earlier run.
- [ ] The caption names the exact run and checkpoint-selection rule.
- [ ] Any interrupted/resumed training is recorded.
- [ ] Claims of stable convergence are supported by curves.

## 5.2 Training-Seed Evidence

If multiple training seeds exist:

| Seed | Best epoch | PSNR | LPIPS | CLIP cos | Independent downstream metric | Checkpoint |
|---:|---:|---:|---:|---:|---:|---|
| `___` | `___` | `___` | `___` | `___` | `___` | `________` |
| `___` | `___` | `___` | `___` | `___` | `___` | `________` |
| `___` | `___` | `___` | `___` | `___` | `___` | `________` |

Across-seed mean and standard deviation:

| Metric | Mean | SD |
|---|---:|---:|
| PSNR | `________` | `________` |
| LPIPS | `________` | `________` |
| CLIP cosine | `________` | `________` |
| Independent downstream metric | `________` | `________` |

If only one seed:

> Only one final teacher seed was trained because:
>
> `________________________________________________________________________`

## 5.3 Student Training Provenance

Complete once per student variant.

| Field | Baseline KD | Downstream heads | Feature KD |
|---|---|---|---|
| Run name | `________` | `________` | `________` |
| Config path | `________` | `________` | `________` |
| Checkpoint path | `________` | `________` | `________` |
| SHA-256 | `________` | `________` | `________` |
| Teacher checkpoint | `________` | `________` | `________` |
| Epoch selected | `________` | `________` | `________` |
| Selection metric/value | `________` | `________` | `________` |
| Training seed | `________` | `________` | `________` |
| Wall-clock/GPU | `________` | `________` | `________` |
| Teacher-output L1 weight | `________` | `________` | `________` |
| Ground-truth L1 weight | `________` | `________` | `________` |
| DINOv2 weight | `________` | `________` | `________` |
| CLIP weight | `________` | `________` | `________` |
| Other feature weights | `________` | `________` | `________` |
| Feature-KD weight | `________` | `________` | `________` |

---

# 6. Verification Tests T1-T8

## T1: Translation Interface

| Check | Expected | Observed | Evidence |
|---|---|---|---|
| Every test input produces output | all pass | `________` | `________` |
| Output channels | 3 | `________` | `________` |
| Output size | same as input | `________` | `________` |
| Finite values | 100% | `________` | `________` |
| Tensor range before conversion | `[-1,1]` | `________` | `________` |
| 8-bit output range | `[0,255]` | `________` | `________` |

Result: `[ ] PASS  [ ] FAIL`

## T2: Dataset and Split Integrity

| Check | Result | Evidence |
|---|---|---|
| Matching stems | `________` | `________` |
| Readable files | `________` | `________` |
| No group overlap | `________` | `________` |
| Alignment acceptance | `________` | `________` |
| Manifest unchanged | `________` | `________` |

Result: `[ ] PASS  [ ] FAIL`

## T3: Determinism

Run at least 10 repeated inferences on a fixed set.

| Runtime/model | Images | Repeats | Max absolute tensor difference | 8-bit mismatched pixels | Result |
|---|---:|---:|---:|---:|---|
| PyTorch FP32 | `___` | `___` | `________` | `________` | `________` |
| ONNX FP32 | `___` | `___` | `________` | `________` | `________` |
| LiteRT FP32 | `___` | `___` | `________` | `________` | `________` |
| LiteRT INT8 | `___` | `___` | `________` | `________` | `________` |

Result: `[ ] PASS  [ ] FAIL`

## T4: Drop-In Interface

| Check | Result | Evidence |
|---|---|---|
| Native sensor frame accepted | `________` | `________` |
| Preprocessing documented | `________` | `________` |
| Standard RGB output produced | `________` | `________` |
| Downstream model needs no retraining | `________` | `________` |
| End-to-end demo command works | `________` | `________` |

Result: `[ ] PASS  [ ] FAIL`

## T5: Downstream Utility

| Success condition | Result |
|---|---|
| Number of defined primary tasks | `________` |
| Tasks with positive gap closure | `________` |
| Tasks below raw-NIR baseline | `________` |
| Required number for pass | `________` |
| Overall S2 result | `PASS / PARTIAL / FAIL` |

Evidence: `________`

## T6: Throughput

| Item | Result |
|---|---|
| Target | `>= 5 FPS` / `<= 200 ms` |
| Mean end-to-end latency | `________ ms` |
| Mean end-to-end FPS | `________` |
| P95 latency | `________ ms` |
| Sustained 10-minute FPS | `________` |
| Overall S3 result | `PASS / FAIL` |

Evidence: `________`

## T7: Edge Footprint

| Item | Result |
|---|---:|
| Model artifact size | `________ MB` |
| Peak RSS after load | `________ MB` |
| Peak RSS during inference | `________ MB` |
| Swap used | `________ MB` |
| OOM events | `________` |
| Meets T6 simultaneously | `Yes / No` |

Result: `[ ] PASS  [ ] FAIL`

## T8: Export and Quantisation Equivalence

| Comparison | Mean abs. error | Max abs. error | PSNR delta | LPIPS delta | Downstream delta |
|---|---:|---:|---:|---:|---:|
| PyTorch vs ONNX FP32 | `___` | `___` | `___` | `___` | `___` |
| PyTorch vs LiteRT FP32 | `___` | `___` | `___` | `___` | `___` |
| Student FP32 vs INT8 | `___` | `___` | `___` | `___` | `___` |

Result: `[ ] PASS  [ ] FAIL`

## Harness Unit Tests

| Fixture | Expected | Observed | Pass? |
|---|---|---|---|
| NIR identity | gap closure 0 | `________` | `[ ]` |
| RGB identity | gap closure 1 | `________` | `[ ]` |
| Negative control | negative closure | `________` | `[ ]` |
| Zero denominator | undefined, no crash | `________` | `[ ]` |
| Empty reference detections | declared behaviour | `________` | `[ ]` |
| Empty translated detections | declared behaviour | `________` | `[ ]` |
| Duplicate candidate match | one-to-one match | `________` | `[ ]` |
| IoU threshold boundary | declared behaviour | `________` | `[ ]` |
| Reordered images | same aggregate | `________` | `[ ]` |
| Lower-is-better metric | correct sign | `________` | `[ ]` |

| Field | Fill in |
|---|---|
| Test command | `________` |
| Passing count | `________` |
| Failing count | `________` |
| Test log path | `________` |

---

# 7. Final Results Tables

## 7.1 Image Quality and Size

All rows must use the same test manifest and preprocessing.

| Model | PSNR dB mean [95% CI] | SSIM mean [95% CI] | LPIPS mean [95% CI] | Params | Artifact MB |
|---|---:|---:|---:|---:|---:|
| Raw NIR | `________` | `________` | `________` | N/A | N/A |
| Pix2Pix baseline | `________` | `________` | `________` | `________` | `________` |
| NAFNet baseline | `________` | `________` | `________` | `________` | `________` |
| Final teacher | `________` | `________` | `________` | `________` | `________` |
| Student FP32 | `________` | `________` | `________` | `________` | `________` |
| Student INT8 | `________` | `________` | `________` | `________` | `________` |

| Provenance | Fill in |
|---|---|
| Raw per-image CSV | `________` |
| Summary CSV | `________` |
| Command | `________` |
| Test n | `________` |
| Manifest hash | `________` |

## 7.2 Foundation and Independent Embedding Agreement

| Evaluator | Used in training? | Raw NIR mean [CI] | Teacher mean [CI] | Student FP32 | Student INT8 |
|---|---|---:|---:|---:|---:|
| DINOv2-S | `Yes / No` | `________` | `________` | `________` | `________` |
| DINOv2-L | `Yes / No` | `________` | `________` | `________` | `________` |
| CLIP-B/16 | `Yes / No` | `________` | `________` | `________` | `________` |
| SigLIP2 | `Yes / No` | `________` | `________` | `________` | `________` |
| Independent model: `______` | `No` | `________` | `________` | `________` | `________` |

Distribution metric:

| Metric | Raw NIR | Teacher | Student FP32 | Student INT8 |
|---|---:|---:|---:|---:|
| `CMMD / FID / chosen metric: ________` | `___` | `___` | `___` | `___` |

- [ ] The report uses the correct metric name. Existing code computes CMMD in
      CLIP space; it should not be relabelled CLIP-FID unless FID is actually run.

## 7.3 Detection Agreement

The current legacy harness computes matched IoU/recall/class agreement against
native-RGB detections. It does not automatically produce COCO mAP. Fill only the
metrics actually implemented, or implement and verify pseudo-label mAP first.

| Detector | Metric | Raw NIR | Teacher | Student FP32 | Student INT8 | Native RGB reference |
|---|---|---:|---:|---:|---:|---:|
| `________` | `________` | `___` | `___` | `___` | `___` | `___` |
| `________` | `________` | `___` | `___` | `___` | `___` | `___` |
| `________` | `________` | `___` | `___` | `___` | `___` | `___` |
| `________` | `________` | `___` | `___` | `___` | `___` | `___` |

Per-detector gap closure:

| Detector | Primary metric | Teacher closure [CI] | Student FP32 | Student INT8 | Negative cases? |
|---|---|---:|---:|---:|---|
| `________` | `________` | `________` | `________` | `________` | `________` |
| `________` | `________` | `________` | `________` | `________` | `________` |

## 7.4 ADE20K Segmentation Agreement

Use “Dice agreement with native-RGB prediction,” not absolute segmentation
accuracy.

| Class | Effective n | Raw NIR Dice [CI] | Teacher Dice [CI] | Student FP32 | Student INT8 |
|---|---:|---:|---:|---:|---:|
| Tree | `___` | `________` | `________` | `________` | `________` |
| Person | `___` | `________` | `________` | `________` | `________` |
| Building | `___` | `________` | `________` | `________` | `________` |
| Road | `___` | `________` | `________` | `________` | `________` |
| Mean | `___` | `________` | `________` | `________` | `________` |

## 7.5 Cityscapes Segmentation Agreement

| Class | Effective n | Raw NIR Dice [CI] | Teacher Dice [CI] | Student FP32 | Student INT8 |
|---|---:|---:|---:|---:|---:|
| Road | `___` | `________` | `________` | `________` | `________` |
| Sidewalk | `___` | `________` | `________` | `________` | `________` |
| Building | `___` | `________` | `________` | `________` | `________` |
| Person | `___` | `________` | `________` | `________` | `________` |
| Vegetation | `___` | `________` | `________` | `________` | `________` |
| Car | `___` | `________` | `________` | `________` | `________` |
| Mean | `___` | `________` | `________` | `________` | `________` |

Segmentation definitions:

| Item | Fill in |
|---|---|
| Minimum reference-mask fraction | `________` |
| Dice formula | `________` |
| Mean weighting: class/image/unweighted | `________` |
| Undefined-class handling | `________` |
| Per-image raw CSV | `________` |

- [ ] Report text no longer equates Dice with intersection-over-union.

## 7.6 Depth Agreement

Do not report metres unless using a metric-depth model with a justified scale.
The existing `eval_all.py` computes MAE after per-image min-max normalisation.
The legacy MiDaS harness computes rank correlation and normalised relative L1.

| Model | Metric | Units | Raw NIR | Teacher | Student FP32 | Student INT8 |
|---|---|---|---:|---:|---:|---:|
| `________` | Spearman rank correlation | unitless | `___` | `___` | `___` | `___` |
| `________` | Normalised relative L1/MAE | unitless | `___` | `___` | `___` | `___` |
| Optional metric-depth model | `________` | `m / N/A` | `___` | `___` | `___` | `___` |

Depth sanity conclusion:

- [ ] Translation preserves or improves depth agreement.
- [ ] Translation harms depth agreement.

Evidence/interpretation:

`____________________________________________________________________________`

## 7.7 Headline Gap-Closure Summary

| Task | Model | Primary metric | Raw NIR | Translated | RGB reference | Gap closure [CI] | Pass? |
|---|---|---|---:|---:|---:|---:|---|
| Detection | `______` | `______` | `___` | `___` | `___` | `___` | `[ ]` |
| Detection | `______` | `______` | `___` | `___` | `___` | `___` | `[ ]` |
| Segmentation | `______` | `______` | `___` | `___` | `___` | `___` | `[ ]` |
| Segmentation | `______` | `______` | `___` | `___` | `___` | `___` | `[ ]` |
| Depth | `______` | `______` | `___` | `___` | `___` | `___` | `[ ]` |
| Independent embedding | `______` | `______` | `___` | `___` | `___` | `___` | `[ ]` |

| Overall statement | Fill in |
|---|---|
| Tasks with positive closure | `________ / ________` |
| Tasks below raw NIR | `________` |
| Median gap closure | `________` |
| Mean gap closure, if defensible | `________` |
| S2 decision | `PASS / PARTIAL / FAIL` |

---

# 8. Ablation Evidence

## 8.1 Teacher Ablation

Evaluate every row under the canonical final protocol.

| Configuration | Exact change | PSNR | LPIPS | Independent embedding | Detection closure | Segmentation closure | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| Raw NIR | no translation | `___` | `___` | `___` | `___` | `___` | N/A |
| Pix2Pix | architecture baseline | `___` | `___` | `___` | `___` | `___` | `___` |
| NAFNet baseline | `________` | `___` | `___` | `___` | `___` | `___` | `___` |
| Region L1 | `________` | `___` | `___` | `___` | `___` | `___` | `___` |
| NIR-gradient L1 | `________` | `___` | `___` | `___` | `___` | `___` | `___` |
| Feature-map variant | `________` | `___` | `___` | `___` | `___` | `___` | `___` |
| Final ensemble | `________` | `___` | `___` | `___` | `___` | `___` | `___` |
| Native RGB | reference | `N/A` | `N/A` | `1.0` | `1.0` | `1.0` | N/A |

Questions the evidence must answer:

| Question | Evidence-based answer |
|---|---|
| Did NAFNet outperform Pix2Pix? | `________` |
| Did feature losses improve independent downstream behaviour? | `________` |
| What pixel-quality tradeoff occurred? | `________` |
| Did any change hurt person/skin classes? | `________` |
| Is the final ensemble best on the stated primary metric? | `________` |
| Are differences larger than uncertainty intervals? | `________` |

## 8.2 Student Distillation Ablation

| Student variant | PSNR | LPIPS | CLIP cos | Independent embedding | Seg. Dice | Detection | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline KD | `___` | `___` | `___` | `___` | `___` | `___` | `___` |
| Downstream heads | `___` | `___` | `___` | `___` | `___` | `___` | `___` |
| Feature KD | `___` | `___` | `___` | `___` | `___` | `___` | `___` |

Selected student:

| Field | Fill in |
|---|---|
| Variant | `________` |
| Selection criterion | `________` |
| Reason it wins | `________` |
| Quality lost versus teacher | `________` |
| Compression ratio | `________` |

---

# 9. Export and Quantisation Evidence

## 9.1 Export Register

| Artifact | Runtime | Precision | Size MB | SHA-256 | Export command | Calibration manifest |
|---|---|---|---:|---|---|---|
| PyTorch checkpoint | PyTorch | FP32 | `___` | `________` | N/A | N/A |
| ONNX | ORT | FP32 | `___` | `________` | `________` | N/A |
| ONNX static/QDQ | ORT | INT8 | `___` | `________` | `________` | `________` |
| LiteRT | XNNPACK | FP32 | `___` | `________` | `________` | N/A |
| LiteRT | XNNPACK | INT8/mixed | `___` | `________` | `________` | `________` |

Calibration:

| Field | Fill in |
|---|---|
| Number of calibration images | `________` |
| Source split | `training` |
| Selection seed/method | `________` |
| Calibration manifest | `________` |
| Calibration manifest hash | `________` |
| Activation calibration method | `________` |
| Weight granularity | `________` |
| Operations left FP32 | `________` |

- [ ] No file under a directory named `test_sample_256` was used for final
      calibration unless independently proven to contain training images.
- [ ] The report identifies mixed-precision INT8 correctly if some operations
      remain FP32.

## 9.2 Numerical Equivalence

| Comparison | n | Mean abs. diff | P95 abs. diff | Max abs. diff | Output range valid? |
|---|---:|---:|---:|---:|---|
| PyTorch vs ONNX FP32 | `___` | `___` | `___` | `___` | `Yes / No` |
| PyTorch vs LiteRT FP32 | `___` | `___` | `___` | `___` | `Yes / No` |
| FP32 student vs ONNX INT8 | `___` | `___` | `___` | `___` | `Yes / No` |
| FP32 student vs LiteRT INT8 | `___` | `___` | `___` | `___` | `Yes / No` |

## 9.3 Quantisation Quality Cost

| Metric | Student FP32 | Student INT8 | Absolute delta | Relative delta |
|---|---:|---:|---:|---:|
| PSNR | `___` | `___` | `___` | `___` |
| SSIM | `___` | `___` | `___` | `___` |
| LPIPS | `___` | `___` | `___` | `___` |
| Independent embedding | `___` | `___` | `___` | `___` |
| Detection closure | `___` | `___` | `___` | `___` |
| Segmentation closure | `___` | `___` | `___` | `___` |
| Depth agreement | `___` | `___` | `___` | `___` |

Quantisation conclusion:

`____________________________________________________________________________`

---

# 10. Raspberry Pi 5 Deployment Evidence

## 10.1 Device and Software Configuration

| Field | Fill in |
|---|---|
| Raspberry Pi model | `________` |
| RAM variant | `________` |
| OS name/release | `________` |
| Kernel | `________` |
| Architecture | `________` |
| Python | `________` |
| Runtime/package versions | `________` |
| Thread count | `________` |
| CPU governor | `________` |
| Clock configuration | `________` |
| Cooling | `________` |
| Ambient temperature | `________` |
| Power supply | `________` |
| Other processes disabled? | `________` |
| Configuration log path | `________` |

## 10.2 Runtime-Only Benchmark

| Model | Runtime | Precision | Warm-ups | Runs | Mean ms | SD | P50 | P95 | FPS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Student | LiteRT/XNNPACK | FP32 | `___` | `___` | `___` | `___` | `___` | `___` | `___` |
| Student | LiteRT/XNNPACK | INT8 | `___` | `___` | `___` | `___` | `___` | `___` | `___` |
| Student | ONNX Runtime | FP32 | `___` | `___` | `___` | `___` | `___` | `___` | `___` |
| Student | ONNX Runtime | static INT8 | `___` | `___` | `___` | `___` | `___` | `___` | `___` |
| Student | ONNX Runtime | Q/DQ INT8 | `___` | `___` | `___` | `___` | `___` | `___` | `___` |

## 10.3 End-to-End Benchmark

Include image decode/acquisition as defined, resize, normalisation, layout
conversion, inference, clipping, and 8-bit RGB conversion.

| Stage | Mean ms | P95 ms |
|---|---:|---:|
| Input read/acquisition | `________` | `________` |
| Resize/normalise/layout | `________` | `________` |
| Runtime inference | `________` | `________` |
| Output clip/layout/uint8 | `________` | `________` |
| Total end to end | `________` | `________` |

| Final result | Fill in |
|---|---|
| End-to-end FPS | `________` |
| Meets 5 FPS? | `Yes / No` |
| Evidence CSV/log | `________` |
| Exact command | `________` |

## 10.4 Sustained Ten-Minute Test

| Measurement | Start | Maximum/minimum | End |
|---|---:|---:|---:|
| Mean latency per interval | `___` | `___` | `___` |
| FPS | `___` | `___` | `___` |
| CPU temperature | `___` | `___` | `___` |
| RSS memory MB | `___` | `___` | `___` |
| Swap MB | `___` | `___` | `___` |
| CPU frequency | `___` | `___` | `___` |

| Item | Fill in |
|---|---|
| Total frames | `________` |
| Throttling observed | `Yes / No` |
| Crashes/errors | `________` |
| Time-series CSV | `________` |
| Plot path | `________` |

---

# 11. Robustness Evidence

Use the same fixed subset and record its manifest.

| Field | Fill in |
|---|---|
| Subset size | `________` |
| Subset seed | `________` |
| Manifest path/hash | `________` |
| Final checkpoint | `________` |

| Perturbation | Level | PSNR delta | LPIPS delta | Detection closure delta | Segmentation closure delta | Failure note |
|---|---:|---:|---:|---:|---:|---|
| None | clean | `0` | `0` | `0` | `0` | baseline |
| Exposure gain | 0.8 | `___` | `___` | `___` | `___` | `________` |
| Exposure gain | 1.2 | `___` | `___` | `___` | `___` | `________` |
| Gamma | 0.8 | `___` | `___` | `___` | `___` | `________` |
| Gamma | 1.2 | `___` | `___` | `___` | `___` | `________` |
| Gaussian blur | sigma 1 | `___` | `___` | `___` | `___` | `________` |
| Gaussian blur | sigma 2 | `___` | `___` | `___` | `___` | `________` |
| Gaussian noise | sigma 0.01 | `___` | `___` | `___` | `___` | `________` |
| Gaussian noise | sigma 0.03 | `___` | `___` | `___` | `___` | `________` |

Robustness conclusion and operating limits:

`____________________________________________________________________________`

---

# 12. Qualitative Evidence

## 12.1 Selection Rules

| Set | Selection rule | n | Manifest |
|---|---|---:|---|
| Typical | nearest to median primary metric | `___` | `________` |
| Strong | top percentile, declared before viewing | `___` | `________` |
| Failure | worst LPIPS/downstream metric | `___` | `________` |
| Person/skin | reference mask/detection criterion | `___` | `________` |
| Quantisation | largest FP32-INT8 difference | `___` | `________` |

## 12.2 Figure Inventory

| Figure | Columns | Image stems | PNG path | PDF path | Caption verified? |
|---|---|---|---|---|---|
| Final teacher examples | NIR / teacher / RGB | `________` | `________` | `________` | `[ ]` |
| Baseline vs final | NIR / baseline / final / RGB | `________` | `________` | `________` | `[ ]` |
| Teacher/student/INT8 | RGB / teacher / FP32 / INT8 / NIR | `________` | `________` | `________` | `[ ]` |
| Failure cases | declared per figure | `________` | `________` | `________` | `[ ]` |
| Alignment failures | overlay/split | `________` | `________` | `________` | `[ ]` |

- [ ] Captions describe observed evidence, not an unsupported general claim.
- [ ] At least one genuine failure is shown and discussed.
- [ ] No cherry-picked sample is described as representative without a rule.

---

# 13. Optional Generalisation Evidence

Complete this only if thermal/SAR remains in the report. Otherwise remove the
section and its claims.

## 13.1 Thermal

| Field | Fill in |
|---|---|
| Dataset/version | `________` |
| Train/val/test counts | `________` |
| Pairing/alignment limitations | `________` |
| Config/checkpoint/hash | `________` |
| Training protocol | `________` |
| Evaluators | `________` |
| Raw artifacts | `________` |

| Metric | Raw thermal | Translated | RGB reference | Gap closure |
|---|---:|---:|---:|---:|
| CLIP cosine | `___` | `___` | `___` | `___` |
| Road Dice | `___` | `___` | `___` | `___` |
| Vegetation Dice | `___` | `___` | `___` | `___` |
| Person Dice | `___` | `___` | `___` | `___` |

## 13.2 SAR

| Field | Fill in |
|---|---|
| Dataset/version | `________` |
| Split counts | `________` |
| Preprocessing | `________` |
| Config/checkpoint/hash | `________` |
| Training protocol | `________` |
| Evaluators | `________` |
| Raw artifacts | `________` |

| Metric | Raw SAR | Translated | Optical reference | Gap closure |
|---|---:|---:|---:|---:|
| PSNR | `___` | `___` | `___` | `N/A` |
| SSIM | `___` | `___` | `___` | `N/A` |
| Classification agreement | `___` | `___` | `___` | `___` |
| RemoteCLIP cosine | `___` | `___` | `___` | `___` |

Generalisation claim supported:

`____________________________________________________________________________`

---

# 14. Reproducibility and User Guide Evidence

## 14.1 Repository

| Field | Fill in |
|---|---|
| Public/private repository URL | `________` |
| Final commit hash | `________` |
| Dirty-worktree status | `________` |
| Release/tag | `________` |
| Licence | `________` |
| Dataset access instructions | `________` |
| Model artifact access | `________` |

## 14.2 Environment

| Component | Version |
|---|---|
| OS | `________` |
| Python | `________` |
| PyTorch | `________` |
| Torchvision | `________` |
| CUDA | `________` |
| cuDNN | `________` |
| OpenCV | `________` |
| Transformers | `________` |
| ONNX | `________` |
| ONNX Runtime | `________` |
| LiteRT/TFLite runtime | `________` |
| NumPy | `________` |
| LPIPS | `________` |
| Other critical package | `________` |

| Artifact | Path |
|---|---|
| `pip freeze` | `________` |
| Conda/environment file | `________` |
| GPU information | `________` |
| Pi package list | `________` |

## 14.3 Reproduction Commands

| Result | Exact command | Expected output |
|---|---|---|
| Dataset audit | `________` | `________` |
| Teacher quality table | `________` | `________` |
| Downstream table | `________` | `________` |
| Teacher ablation | `________` | `________` |
| Student ablation | `________` | `________` |
| ONNX export | `________` | `________` |
| LiteRT export | `________` | `________` |
| Quantisation quality | `________` | `________` |
| Pi benchmark | `________` | `________` |
| Robustness sweep | `________` | `________` |
| Qualitative figures | `________` | `________` |
| Build report | `________` | `________` |

## 14.4 Deployment Procedure Validation

- [ ] Fresh environment created from the documented commands.
- [ ] Model downloaded/copied using the documented path.
- [ ] One NIR image translated successfully.
- [ ] Output image opened and validated.
- [ ] Live/folder inference command tested on the Pi.
- [ ] Troubleshooting covers shape, channel order, missing delegate, and memory.
- [ ] A second person or clean machine followed the guide, if possible.

Validation notes:

`____________________________________________________________________________`

---

# 15. Ethical, Legal, Safety, Environmental, and Social Evidence

## 15.1 Ethical and Privacy

| Question | Evidence-based answer |
|---|---|
| What personal data could be captured? | `________` |
| What lawful/ethical basis applies to public-space capture? | `________` |
| How was targeting of individuals avoided? | `________` |
| How were faces/plates anonymised? | `________` |
| What residual privacy risk remains? | `________` |
| How long were raw images retained? | `________` |
| Who could access the data? | `________` |
| Is the dataset publicly released? Why/why not? | `________` |

## 15.2 Safety

| Hazard | Mitigation | Evidence |
|---|---|---|
| Cycling while carrying equipment | `________` | `________` |
| Battery/power safety | `________` | `________` |
| Weather/electrical exposure | `________` | `________` |
| Thermal load on Pi | `________` | `________` |
| Public distraction/obstruction | `________` | `________` |

## 15.3 Bias and Misuse

| Issue | Evidence/discussion |
|---|---|
| London outdoor-scene bias | `________` |
| Weather/time-of-day bias | `________` |
| Person/skin performance | `________` |
| Surveillance misuse | `________` |
| Hallucinated colour/semantics | `________` |
| Unsuitable safety-critical use | `________` |

## 15.4 Environmental Impact

| Item | Fill in |
|---|---|
| Approximate training GPU hours | `________` |
| Number/type of major runs | `________` |
| Cloud region, if known | `________` |
| Carbon estimate, if defensible | `________` |
| Hardware reuse | `________` |
| Mitigations: spot instances/early stopping/etc. | `________` |
| Edge inference benefit/tradeoff | `________` |

---

# 16. Critical Evaluation Fill-In

These boxes supply the evidence needed for Chapter 7.

## 16.1 Achievement of Objectives

| Objective | Evidence IDs | Achieved? | Exact evidence-based conclusion |
|---|---|---|---|
| Paired dataset | `E__` | `Yes/Partial/No` | `________` |
| Translator quality | `E__` | `Yes/Partial/No` | `________` |
| Edge deployment | `E__` | `Yes/Partial/No` | `________` |
| Downstream recovery | `E__` | `Yes/Partial/No` | `________` |

## 16.2 Requirements Scorecard

| Requirement | Acceptance condition | Result | Evidence ID | Pass? |
|---|---|---|---|---|
| F1 Translation | valid same-size RGB output | `________` | `E__` | `[ ]` |
| F2 Geometry | alignment/geometry preserved | `________` | `E__` | `[ ]` |
| F3 Determinism | repeated 8-bit output identical | `________` | `E__` | `[ ]` |
| F4 Drop-in | standard RGB interface | `________` | `E__` | `[ ]` |
| N1 Throughput | >=5 FPS end to end | `________` | `E__` | `[ ]` |
| N2 Footprint | no swap/OOM while meeting N1 | `________` | `E__` | `[ ]` |
| N3 Utility | required positive gap closure | `________` | `E__` | `[ ]` |

## 16.3 Comparison with Existing Approaches

| Comparator | What is comparable? | Your result | Published result | Fair comparison? | Citation |
|---|---|---:|---:|---|---|
| Pix2Pix | same internal dataset/protocol | `___` | internal | `Yes` | `________` |
| Pix2Next | `________` | `___` | `___` | `________` | `________` |
| Other NIR-to-RGB work | `________` | `___` | `___` | `________` | `________` |
| Retraining downstream models | `________` | `___` | `___` | `________` | `________` |

Do not compare raw scores across different datasets as if they were controlled.

## 16.4 Did the Design Decisions Pay Off?

| Design decision | Supporting evidence | Contrary evidence | Final judgement |
|---|---|---|---|
| Shared-aperture capture | `________` | `________` | `________` |
| NAFNet over Pix2Pix | `________` | `________` | `________` |
| Foundation-feature loss | `________` | `________` | `________` |
| Teacher/student separation | `________` | `________` | `________` |
| PTQ over QAT | `________` | `________` | `________` |
| LiteRT/XNNPACK runtime | `________` | `________` | `________` |

## 16.5 Limitations

| Limitation | Evidence | Effect on claims | Mitigation/future test |
|---|---|---|---|
| No human task annotations | `________` | `________` | `________` |
| One city/scene domain | `________` | `________` | `________` |
| One sensor/rig | `________` | `________` | `________` |
| Limited training seeds | `________` | `________` | `________` |
| Evaluators overlap training losses | `________` | `________` | `________` |
| Pseudo-ground-truth agreement | `________` | `________` | `________` |
| Person/skin failures | `________` | `________` | `________` |
| Quantisation/runtime specificity | `________` | `________` | `________` |
| Alignment residuals | `________` | `________` | `________` |

## 16.6 Novelty

| Potential contribution | Evidence | Safe wording |
|---|---|---|
| Shared-aperture paired dataset | `________` | `________` |
| NAFNet for NIR-to-RGB | `________` | `________` |
| Foundation-feature ensemble objective | `________` | `________` |
| Downstream gap-closure harness | `________` | `________` |
| Pi 5 deployment path | `________` | `________` |

- [ ] Literature search supports any “first” claim.
- [ ] If not, wording is “to the best of the literature reviewed” rather than an
      absolute priority claim.

---

# 17. Conclusions and Further Work Fill-In

## 17.1 Summary of Achievements

Fill only after the results are frozen.

| Achievement | One-sentence conclusion | Evidence |
|---|---|---|
| Dataset | `________` | `E__` |
| Teacher | `________` | `E__` |
| Student | `________` | `E__` |
| Downstream behaviour | `________` | `E__` |
| Edge deployment | `________` | `E__` |

## 17.2 Most Important Quantitative Outcomes

| Outcome | Final number |
|---|---:|
| Dataset size | `________` |
| Best independent downstream gap closure | `________` |
| Number of positive tasks | `________ / ________` |
| Teacher-to-student compression | `________ x` |
| INT8 artifact size | `________ MB` |
| End-to-end Pi throughput | `________ FPS` |
| Quantisation quality cost | `________` |

## 17.3 Further Work Prioritisation

| Priority | Work item | Evidence motivating it | Expected benefit |
|---:|---|---|---|
| 1 | `________` | `________` | `________` |
| 2 | `________` | `________` | `________` |
| 3 | `________` | `________` | `________` |
| 4 | `________` | `________` | `________` |

## 17.4 Personal/Engineering Reflection

| Prompt | Fill in |
|---|---|
| Most difficult technical problem | `________` |
| Key insight used to solve it | `________` |
| Decision that changed during the project | `________` |
| Evidence that motivated the change | `________` |
| What you would do differently | `________` |
| Technical skill learned | `________` |
| Professional/project-management lesson | `________` |

---

# 18. Front Matter Fill-In

## 18.1 Abstract

| Required element | Final wording/fact |
|---|---|
| Problem | `________` |
| Method | `________` |
| Dataset | `________` |
| Main technical contribution | `________` |
| Main downstream result | `________` |
| Edge result | `________` |
| Main limitation | `________` |

- [ ] Every number in the abstract appears in a final evidence table.
- [ ] “Recovers most of the gap” is used only if the measured closures support it.

## 18.2 Plain-Language Summary

| Prompt | Fill in without specialist terminology |
|---|---|
| What problem was solved? | `________` |
| Why does it matter? | `________` |
| What was built? | `________` |
| How was it tested? | `________` |
| What worked? | `________` |
| What remains limited? | `________` |

## 18.3 Declaration of LLM Usage

Follow the institution’s required wording and disclose actual use.

| Item | Fill in |
|---|---|
| Tools used | `________` |
| Uses: planning/editing/code/debugging/etc. | `________` |
| Parts affected | `________` |
| How outputs were verified | `________` |
| Final declaration wording/source | `________` |

---

# 19. Final Report Consistency Audit

## 19.1 Numerical Consistency

- [ ] Dataset count is identical everywhere.
- [ ] Split seed and grouping are identical everywhere.
- [ ] Teacher run name is identical everywhere.
- [ ] Teacher parameter count is identical everywhere.
- [ ] Student architecture and parameter count are identical everywhere.
- [ ] Student loss weights match the final config.
- [ ] Evaluator panel is identical in introduction, test plan, and results.
- [ ] Runtime-only and end-to-end latency are labelled separately.
- [ ] FPS equals `1000 / mean_latency_ms` where appropriate.
- [ ] Abstract numbers match final tables.

## 19.2 Methodological Consistency

- [ ] Detection metric is named correctly; mAP is not claimed if only matched
      IoU/recall was calculated.
- [ ] Dice and IoU are not treated as synonyms.
- [ ] Relative depth is not reported in metres.
- [ ] CMMD is not called CLIP-FID.
- [ ] Pseudo-ground-truth agreement is not called task accuracy.
- [ ] Test bootstrap intervals are not presented as training-seed uncertainty.
- [ ] Test data were not used for quantisation calibration.
- [ ] All ablations use the same test manifest and preprocessing.

## 19.3 Build and Presentation

| Check | Result |
|---|---|
| LaTeX compile command | `________` |
| Compile date | `________` |
| Undefined references | `________` |
| Missing citations | `________` |
| Overfull boxes reviewed | `________` |
| Visible TODO/TBD count | `________` |
| Final PDF pages | `________` |
| Final PDF size | `________` |

- [ ] `sec:res_gapclosure` reference exists or is corrected.
- [ ] `sec:res_abl_seg` reference exists or is corrected.
- [ ] Appendix C is included or all references to it are removed.
- [ ] Critical Evaluation is substantive.
- [ ] Conclusions and Further Work is substantive.
- [ ] Ethical/legal/safety appendix is complete.
- [ ] User/deployment guide contains tested commands.
- [ ] Plain-language summary is complete.
- [ ] LLM declaration is complete.

---

# 20. Final Sign-Off

| Question | Answer |
|---|---|
| Are all mandatory evidence IDs E01-E25 complete? | `Yes / No` |
| Are all report TODO/TBD markers removed? | `Yes / No` |
| Does every headline claim have raw evidence? | `Yes / No` |
| Can every table be regenerated from saved commands? | `Yes / No` |
| Are failures and negative results disclosed? | `Yes / No` |
| Has the supervisor reviewed the frozen results? | `Yes / No` |
| Final report submission-ready? | `Yes / No` |

Final unresolved items:

1. `________________________________________________________________________`
2. `________________________________________________________________________`
3. `________________________________________________________________________`

