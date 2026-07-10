# Agent brief: gather everything needed to maximise WACV 2027 acceptance odds

## Context (read first)

- **Paper:** `paper/wacv2027_draft.tex` (builds with `lualatex → biber → lualatex`; currently 7 pp incl. refs). Condensed from the MEng thesis in `chapters/` + `main.tex`.
- **Claim:** NIR→RGB translation for frozen RGB models should be trained/selected/evaluated on downstream model behaviour (gap closure), not pixel fidelity. Pix2Pix ties our student on PSNR/SSIM but recovers ~no downstream behaviour.
- **Deployed model in paper:** `student_l1b_c` (feature-KD NAFNet-16, thesis "selected student"). Decision made: keep it despite SegFormer/DepthAnything overlap with the eval panel — do NOT swap models without asking.
- **Target:** WACV 2027, full-paper deadline **29 Aug 2026** (OpenReview; paper-enrollment step earlier — verify). Fallback: PBVS @ CVPR 2027.
- **Key data locations:**
  - Canonical eval numbers: `NIR-RGB/experiments/eval_full_v4_canonical/summary.csv` (+ `ci/summary_with_ci.csv`, per-image CSVs)
  - INT8 eval: `NIR-RGB/experiments/eval_full_v5_int8/`
  - Ablations: `NIR-RGB/experiments/eval_full_v5_ablations/`
  - Second split-seed eval: `NIR-RGB/experiments/eval_full_v6_seed8065/`
  - Robustness: `NIR-RGB/experiments/robustness_v1/`
  - Training configs: `NIR-RGB/configs/`, run scripts `NIR-RGB/run-*.sh`, eval scripts `NIR-RGB/eval_*.py`, `NIR-RGB/scripts/`
  - Experiment tracking: Weights & Biases (ask user for project link/access)
  - Bibliography: `references.bib` (repo root)

Priorities: **P0 = paper is rejected/desk-rejected without it. P1 = materially raises acceptance odds. P2 = nice to have.**

---

## A. Verify existing numbers (P0, no training needed)

- [ ] **Resolve the `l1boost` vs `teacher_15b` discrepancy.** In `eval_full_v4_canonical/summary.csv` the run `l1boost` (epoch 191) beats the paper's teacher `teacher_15b` on nearly everything (PSNR 27.22, YOLO 0.320, RT-DETR 0.400, depth 0.317, CLIP-FID 2.06). Find out which config/run `l1boost` is (check `NIR-RGB/configs/`, W&B run names, `version_history.txt`). If it is a legitimate teacher candidate trained on the same split, the paper's teacher row is understating our own results; report findings to user before changing anything.
- [ ] **Recompute every number in the draft's tables** from the CSVs above and diff against the .tex (quality table, closure table, ablation table, edge table figures from thesis Ch.7). Flag any mismatch.
- [ ] **Confirm the exact checkpoint/config provenance** for every row (model name → config file → checkpoint) so the reproducibility statement is defensible.

## B. New experiments to queue (P0/P1 — need GPU; user runs them, agent prepares)

- [ ] **P0 Multi-seed runs.** Prepare configs + run scripts for 2 extra seeds of: (1) final teacher recipe, (2) No-FM ablation, (3) selected student distillation. Note `eval_full_v6_seed8065` already covers a second *data-split* seed for teacher-side runs — decide and document whether we report split-seeds, training-seeds, or both. Deliverable: mean±std for the headline and ablation tables.
- [ ] **P1 One published NIR→RGB baseline.** Survey recent NIR colourisation methods **with public code** (start: VCIP NIR colourisation challenge entries, ATcycleGAN, CoColor, MFF/MPFNet-style, Pix2Next reversed). Pick 1–2 that can train on our 256×256 paired split; prepare training config on our canonical split and run through `eval_all.py` + downstream harness. Deliverable: extra row(s) in quality + closure tables.
- [ ] **P1 Pix2Pix + foundation-ensemble loss.** Disentangles backbone vs loss. Check whether the existing Pix2Pix training path (`PIX2PIX_BASELINE_INSTRUCTIONS.md`, `NIR-RGB/configs/baseline.yaml`) can accept the FM loss terms; prepare the config. Deliverable: one ablation row.
- [ ] **P2 External-data sanity check.** Run the trained student on EPFL RGB-NIR scenes (imperfect pairs — embedding/qualitative eval only) to show it doesn't collapse off-distribution.

## C. Literature gathering (P0, desk work)

- [ ] **NIR colourisation related-work paragraph.** Collect 6–10 citations: NIR/thermal colourisation GANs, VCIP challenge reports, recent diffusion-based cross-spectral work, task-driven translation/domain-adaptation papers. Produce BibTeX entries (add to `references.bib`) + a drafted paragraph for §2, replacing the `\TODO{cite recent NIR colourisation work...}`.
- [ ] **Positioning check.** For each collected paper note: dataset used, direction (NIR→RGB vs RGB→NIR), whether they evaluate downstream tasks, model size / edge deployment. Build a small comparison table (candidate supplementary material) showing no prior work combines simultaneous paired capture + behaviour-centred multi-task eval + edge deployment.

## D. Dataset release (P0 — promised in the paper)

- [ ] Choose hosting (Zenodo or HuggingFace Datasets; both give a DOI/stable link). Draft the dataset card: capture protocol, alignment pipeline, 80/10/10 split files, licence (suggest CC BY-NC 4.0 — confirm with user), privacy statement ported from thesis Appendix A.
- [ ] Package: aligned 256×256 pairs + split manifests + alignment metadata. Verify no identifiable faces/plates survive in a random 500-image audit.
- [ ] Replace `\TODO{add dataset link}` (anonymised placeholder for review; real link at camera-ready).

## E. Submission mechanics (P0, deadline-driven)

- [ ] **Download the official WACV 2027 author kit** (wacv.thecvf.com → Author Guidelines) and port the draft: template, 8 pp excl. refs, anonymise (authors, dataset link, acknowledgements, no thesis self-reference). Keep `wacv2027_draft.tex` as the working copy; produce `wacv2027_submission/`.
- [ ] **Confirm dates on the official page:** paper-enrollment deadline vs full-paper deadline (29 Aug 2026), supplementary deadline, round-2 dates.
- [ ] **Track choice:** pull the review criteria for Applications vs Evaluation & Datasets tracks; summarise for the user + supervisor to decide.
- [ ] OpenReview profiles: remind user that **all** co-authors need valid profiles before the deadline (desk-reject otherwise).
- [ ] **Supplementary material:** assemble from existing assets — full 18-class Cityscapes table, student-ablation table, robustness table, rig photos/BOM, alignment pipeline figures (`imgs/pipeline_report/`), privacy statement, harness unit-test description.

## F. Writing polish (P1, after A–C land)

- [ ] Fold new baseline + seed results into tables; delete the corresponding `\TODO`s.
- [ ] Check figure quality at print size (current PDF ~8 MB; `random_no_people_grid.png` legibility in one column — regenerate tighter crop if needed).
- [ ] Consistency pass with supervisor feedback already applied to thesis: "behavioural agreement, not accuracy" phrasing everywhere; "homography-aligned" not "pixel-perfect"; decisive one-line finding in abstract + conclusion.
- [ ] Final compile check: no overfull boxes, no undefined refs, page budget with references ≤ 8+refs.

---

**Do not do without asking the user:** swapping the deployed student model, changing the title, contacting anyone, uploading the dataset publicly, or submitting anything.
