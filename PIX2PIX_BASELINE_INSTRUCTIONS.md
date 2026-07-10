# Pix2Pix Baseline Parity Instructions

Instructions for an agent filling in Pix2Pix baseline numbers in the final
report. Companion to `REPORT_EVIDENCE_WORKBOOK.md`; all rules in that file
(no estimated values, canonical test manifest, recorded provenance) apply
here too.

## Standing rule

**Every time the report quotes a number for the teacher or the student —
in a table or in prose — the corresponding Pix2Pix baseline number must be
quotable from the same protocol.** Tables that list Teacher/Student columns
must have a Pix2Pix column with a real value or an explicit `N/A - reason`.
Prose that states a teacher or student value or gap closure must either
state the Pix2Pix value alongside it or cite a table where it appears.

This applies to future edits as well: if a new metric is added for teacher
or student, the Pix2Pix run for that metric is part of the same task, not a
follow-up.

## Rules

- [ ] Every Pix2Pix value comes from an inference run of the registered
      Pix2Pix checkpoint on the canonical test manifest — never copied from
      an old run, another protocol, or estimated from nearby numbers.
- [ ] The same samples, order, resolution, and preprocessing are used as for
      the teacher and student rows in the same table.
- [ ] Gap closures use the formulas in `REPORT_EVIDENCE_WORKBOOK.md` §1.4.
- [ ] Negative closures are retained and reported, not clipped or hidden.
- [ ] Every value written into the LaTeX is first recorded in the fill-in
      tables below with its raw artifact path.

---

# 1. Source of Truth

## 1.1 Pix2Pix model register

Do not run anything until this is complete (mirror of workbook §3).

| Field | Fill in |
|---|---|
| Run name | `________` |
| Config path | `________` |
| Checkpoint path | `________` |
| Checkpoint SHA-256 | `________` |
| Epoch / selection rule | `________` |
| Parameter count (expect 54,414,531) | `________` |
| Test manifest path | `________` |
| Test manifest SHA-256 | `________` |
| Eval code commit | `________` |

## 1.2 Where outputs may already exist

Before running new inference, search for existing per-image Pix2Pix results
(detection JSON/CSV, embedding CSV, segmentation CSV, depth CSV) produced
under the canonical protocol. Record what was found:

| Artifact searched for | Found? | Path or `not found - reran` |
|---|---|---|
| Pix2Pix detection outputs | `Yes / No` | `________` |
| Pix2Pix CLIP-FID inputs (translated test set) | `Yes / No` | `________` |
| Pix2Pix embedding per-image CSV | `Yes / No` | `________` |
| Pix2Pix segmentation per-image CSV | `Yes / No` | `________` |
| Pix2Pix depth per-image CSV | `Yes / No` | `________` |

If an existing artifact cannot be proven to use the canonical test manifest
(hash match), do not use it — rerun.

## 1.3 Gap-closure conventions used in Chapter 7

Native-RGB outputs are the pseudo-ground-truth reference, so the reference
value is definitional, not measured:

- Agreement metrics (mAP vs pseudo-labels, Dice agreement, cosine):
  native RGB = 1.000, and
  `closure = (pix2pix - raw_nir) / (1.000 - raw_nir)`
- Depth normalised L1 (lower is better): native RGB = 0.000, and
  `closure = (raw_nir - pix2pix) / raw_nir`

Verify against published teacher/student values before trusting your
pipeline (e.g., YOLOv8m teacher: (0.308 − 0.224) / (1 − 0.224) = 0.108 ✓;
depth FP32 student: (0.677 − 0.424) / 0.677 = 0.374 ✓).

---

# 2. Required Values

Fill every blank, then apply to `chapters/Chapter7.tex` as directed in §3.

## 2.1 Detection (replaces the four `TBD`s in `tab:res_detection`)

| Detector | Metric | Pix2Pix value | Pix2Pix closure | Raw artifact path |
|---|---|---:|---:|---|
| YOLOv8m | mAP@0.5 | `________` | `________` | `________` |
| YOLOv8m | mAP@0.5:0.95 | `________` | `________` | `________` |
| RT-DETR-L | mAP@0.5 | `________` | `________` | `________` |
| RT-DETR-L | mAP@0.5:0.95 | `________` | `________` | `________` |

Note: the ablation table (`tab:abl_summary`) already prints a Pix2Pix YOLO
closure of −0.190. If your measured closure differs, see §4 consistency
checks — one of the two protocols is wrong and must be reconciled, not
averaged.

## 2.2 Distribution metric (replaces the `—` in `tab:res_pixel_embedding`)

| Metric | Pix2Pix value | Raw artifact path |
|---|---:|---|
| CLIP-FID (CMMD — confirm naming per workbook §7.2) | `________` | `________` |

## 2.3 Embedding closures (for prose parity in the embedding subsection)

Cosine means already appear in `tab:res_pixel_embedding`; compute the
closures so prose can quote them next to the student closures 0.566/0.525
(DINOv2-L) and 0.538/0.511 (SigLIP2).

| Evaluator | Pix2Pix cosine (from table) | Pix2Pix closure | Notes |
|---|---:|---:|---|
| DINOv2-L | 0.7229 | `________` | expected ≈ −0.02 (negative — report it) |
| SigLIP2 | 0.8940 | `________` | `________` |
| DINOv2-S | 0.8072 | `________` | supporting evidence only |
| CLIP-B/16 | 0.8774 | `________` | supporting evidence only |

## 2.4 Segmentation closures (for prose parity and `tab:res_gapclosure`)

Class-mean Dice values already appear in the segmentation tables.

| Task | Pix2Pix class-mean Dice (from table) | Pix2Pix closure |
|---|---:|---:|
| ADE20K (4-class mean) | 0.750 | `________` |
| Cityscapes (mean excl. traffic light) | 0.572 | `________` |

## 2.5 Depth closure

| Metric | Pix2Pix value (from table) | Pix2Pix closure |
|---|---:|---:|
| Normalised L1 | 0.678 | `________` (expected ≈ 0 or negative) |

## 2.6 Headline gap-closure summary additions

These populate a new Pix2Pix column (or rows) in `tab:res_gapclosure`:

| Task | Primary metric | Pix2Pix value | Pix2Pix closure |
|---|---|---:|---:|
| Detection 1 | YOLOv8m mAP@0.5 | `________` | `________` |
| Detection 2 | RT-DETR-L mAP@0.5 | `________` | `________` |
| Segmentation 1 | ADE class-mean Dice | 0.750 | `________` |
| Segmentation 2 | Cityscapes class-mean Dice | 0.572 | `________` |
| Depth | Normalised L1 | 0.678 | `________` |
| Independent embedding | DINOv2-L cosine | 0.723 | `________` |
| Independent embedding | SigLIP2 cosine | 0.894 | `________` |

---

# 3. Where to Apply the Values in `chapters/Chapter7.tex`

Work through each location in order. Do not change teacher/student values
while doing this; only add/replace Pix2Pix material.

| # | Location (anchor) | What to do |
|---|---|---|
| 1 | `tab:res_detection` | Replace the four `TBD` cells with §2.1 values. |
| 2 | `tab:res_pixel_embedding`, CLIP-FID row | Replace `—` with §2.2 value. |
| 3 | Detection prose (paragraph beginning "The teacher improves both metrics…") | Add one sentence quoting the Pix2Pix mAP@0.5 closures from §2.1 next to the teacher (0.108 / 0.194) and student (0.077 / 0.072) closures. |
| 4 | Embedding prose (paragraph beginning "The teacher is strongest on every representation metric…") | Where student closures 0.566/0.525 and 0.538/0.511 are quoted, add the Pix2Pix closures from §2.3 in the same sentence or the next one. |
| 5 | Segmentation prose (paragraph beginning "Both segmentation tasks show a larger recovery…") | Add Pix2Pix closures from §2.4 alongside the quoted student closures 0.587/0.569 and 0.351/0.311. |
| 6 | Depth prose (paragraph beginning "The FP32 and INT8 students both improve…") | Add the Pix2Pix closure from §2.5 alongside the student closures 0.374/0.328. |
| 7 | `tab:res_gapclosure` | Add a Pix2Pix column (value + closure) using §2.6, and update the caption to say Pix2Pix is included as the conventional baseline. |
| 8 | Paragraph after `tab:res_gapclosure` ("Pix2Pix is the standard baseline…") | Verify the quoted 0.723 (DINOv2-L) and 0.678 (depth) still match the tables; extend with the measured detection numbers from §2.1 if they strengthen or contradict the argument as written. |
| 9 | Key Findings bullet 1 | Verify the claim "recovers almost none of the downstream behaviour" against the measured detection/segmentation closures. Note: ADE closure 0.18→ if §2.4 gives a clearly positive ADE closure, soften the bullet to name where Pix2Pix does and does not recover behaviour. |

Also update `REPORT_EVIDENCE_WORKBOOK.md`: record the detection artifacts
under E10, the CLIP-FID artifact under E09, and complete the Pix2Pix row of
the Model and Artifact Register (§3) from §1.1 above.

---

# 4. Consistency Checks (must pass before sign-off)

- [ ] Pix2Pix PSNR: `tab:res_imagequality` says 24.03, `tab:abl_summary`
      says 23.69. Raw NIR also differs (16.65 vs 16.82). Determine which
      protocol each table used; either bring the ablation table onto the
      canonical protocol or add an explicit caption note that the ablation
      table uses a different evaluation subset. Record the resolution:
      `________________________________________________________________`
- [ ] Pix2Pix YOLO closure in `tab:abl_summary` (−0.190) agrees with the
      §2.1 measurement, or the discrepancy is resolved the same way.
- [ ] Pix2Pix ADE closure in `tab:abl_summary` (0.307) agrees with §2.4.
- [ ] Every Pix2Pix number quoted in prose matches a table in the same
      chapter exactly (0.723, 0.678, 24.03, 0.752, 0.185, 54.4 M).
- [ ] No `TBD`, `—`, or blank remains in any Pix2Pix cell of Chapter 7.
- [ ] All new values use the same decimal precision as the neighbouring
      teacher/student values in the same table.
- [ ] Negative Pix2Pix closures (expected on DINOv2-L and depth) are
      printed with their sign, not as `N/A` or 0.
- [ ] Search the other chapters (`Glob chapters/*.tex`, grep for
      `Pix2Pix|pix2pix`) for quoted teacher/student numbers lacking a
      Pix2Pix counterpart; list anything found here rather than silently
      editing other chapters:
      `________________________________________________________________`

# 5. Sign-Off

| Question | Answer |
|---|---|
| Were any Pix2Pix values reused from a non-canonical run? | `Yes / No` |
| Are all §2 blanks filled with measured values or `N/A - reason`? | `Yes / No` |
| Do all §4 checks pass? | `Yes / No` |
| Are raw artifacts and commands recorded in the workbook? | `Yes / No` |

Commands used (exact, reproducible):

| Result | Command |
|---|---|
| Pix2Pix detection eval | `________` |
| Pix2Pix CLIP-FID | `________` |
| Closure computation | `________` |
