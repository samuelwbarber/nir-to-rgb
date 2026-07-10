# Speaker notes — Edge-Deployable NIR-to-RGB Translation

**Format:** 15 min talk + 10 min Q&A. Target landing **14:30** to leave buffer.
**The one through-line (say it on slide 3, pay it off on slides 9 & 11):**
*A translation pre-processor should be judged by how much frozen-model behaviour it recovers — not by how good the image looks.*

Cumulative time targets are in **[brackets]** — glance at the clock when you change slides and check you're on pace.

---

## 1. Title — 0:15  **[0:15]**
- "Good [morning]. I'm Sam Barber. My project is an edge-deployable NIR-to-RGB translator that lets ordinary RGB vision models work on near-infrared input."
- Don't linger. Move.

## 2. The problem — 1:10  **[1:25]**
- NIR is everywhere useful — night vision, biometrics, agriculture, inspection.
- But the whole vision toolchain is trained on visible RGB. On NIR it breaks.
- Gesture across the three: "Same scene, RGB works, NIR fails — a detector, and two segmentation models. No colour, materials look different."
- Punch: "Retraining every downstream model on labelled NIR is impractical."
- **Bridge:** "So instead of changing the models, change the input."

## 3. The idea + thesis — 1:10  **[2:35]**
- Walk the arrow: NIR sensor → translator → *frozen, unmodified* RGB model.
- **Read the blue box almost verbatim — this is the spine of the talk.**
- "Everything I show next is evidence for that one claim."
- **Bridge:** "Here's what the deliverable actually has to do."

## 4. Objectives — 0:40  **[3:15]**
- Four objectives: collect aligned data; train a semantics-preserving translator; deploy on a Pi at ~5 FPS; and prove it with downstream gap closure.
- "Note the bar: not photorealism — downstream usefulness."
- **Bridge:** "Why isn't this just standard image translation?"

## 5. Why it's hard + prior work — 0:50  **[4:05]**  *(most compressible — cut here if behind)*
- Hard: colour is absent (one-to-many), materials differ, paired data is scarce because of parallax.
- Prior work: GANs and diffusion, judged on image quality; diffusion is iterative + stochastic — wrong for a deterministic edge pre-processor.
- "That gives my two design bets: a deterministic restoration backbone, and feature-space supervision."
- **Bridge:** "First, the data."

## 6. Data: the rig — 1:20  **[5:25]**
- "I built a capture rig: a Pi 5, two camera modules, one behind an 850 nm filter."
- **The key idea:** beam-splitter cube → both cameras share an optical centre → no parallax → a *single* homography aligns the whole frame regardless of depth.
- "20+ hours cycling London, filtered and aligned down to 12,536 pairs."
- **Bridge:** "Now the model — and the core idea of the project."

## 7. Core method: foundation-feature supervision — 1:50  **[7:15]**  *(give this room — biggest idea)*
- Backbone: NAFNet, an activation-free restoration net. Chosen for a clean, quantisation-friendly operator set — unusual for a synthesis task.
- **The key swap:** "Standard translators use one VGG perceptual loss. I replace it with a frozen *ensemble* — DINOv2, CLIP, ResNet-50 — plus L1, MS-SSIM, a light GAN term."
- Point at the equation: "Train so each frozen extractor sees the translation the way it sees real RGB. Then any downstream model built on those features behaves more like it does on RGB."
- "That's the mechanism behind the whole thing."
- **Bridge:** "But the teacher is 116 million parameters — too big for a Pi."

## 8. Making it deployable — 1:00  **[8:15]**
- Walk the arrow: teacher → distil 67× → 1.7M student → mixed-INT8 → 2.3 MB.
- "Student keeps the activation-free blocks, so the same clean export. Distilled on the teacher's outputs plus feature losses — downstream fidelity, not pixel mimicry."
- "Exported to LiteRT for the CPU and a Hailo HEF for the NPU."
- **Bridge:** "So does it actually work? First, the eye test."

## 9. Does it actually work? (qualitative) — 0:55  **[9:10]**
- "Rows: NIR in at the top, ground truth at the bottom, and the models in between."
- "All of them produce plausible colour. Pix2Pix — row two — even looks competitive."
- **Set the trap:** "Hold that thought — keep your eye on Pix2Pix."
- **Bridge:** "Because the eye test is exactly what misleads you. Here's how I measure it properly."

## 10. Gap closure — 1:05  **[10:15]**
- "Run a frozen model on three inputs — raw NIR, my translation, real RGB — and ask how far translation moves it toward the RGB result."
- "Zero = no better than NIR. One = full RGB behaviour recovered. Negative = made it worse."
- "Reference is the model's own RGB output, so no human labels needed — across detection, segmentation, depth, and held-out embeddings."
- **Bridge:** "Here's the headline."

## 11. Headline result — 1:45  **[12:00]**  *(the money slide — slow down)*
- "Both student precisions: positive on all five primary tasks. Pre-specified bar was three of five."
- **The payoff:** "Now Pix2Pix. It tied the student on PSNR and SSIM — looked fine. But downstream it's *negative* on both detectors, and on the independent probes it sits at the no-translation level."
- "So a model that looked competitive recovers almost nothing. That's the central finding: pixel fidelity doesn't predict downstream usefulness."
- (If asked live why only DINOv2-L/SigLIP2: "those were held out of training — I have the others in backup.")
- **Bridge:** "And it runs on the device."

## 12. Edge + live demo — 1:10  **[13:10]**
- "On the Pi CPU, the INT8 student hits 5.05 FPS — meets target. On the Hailo NPU, 27 FPS."
- "INT8 is actually *faster* than FP32 here — narrower kernels, less bandwidth. Under 50 MB, no swap, no throttling."
- Point at screenshot: "This is the live web app on the Pi — NIR, translation, and a reference, all segmented in real time."
- **Bridge:** "Briefly, the honest limitations."

## 13. Limitations — 0:40  **[13:50]**
- Hit fast, don't dwell: behaviour not accuracy (pseudo-labels); people untested (privacy); London-only; breaks on SAR; CPU margin slim so NPU is recommended.
- "Stating these is part of the result — I know where it stops working."
- **Bridge:** "To wrap up."

## 14. Conclusion — 0:40  **[14:30]**
- Restate the blue box: "Pixel fidelity doesn't predict downstream usefulness — supervising and evaluating in the models' feature spaces is what recovers behaviour."
- Four deliverables in one breath: rig + 12,536 pairs; 1.7M distilled student; positive closure on all five tasks; real-time on the Pi.
- "Thank you — happy to take questions."

---

## Delivery reminders
- **Pace by the clock**, not by feel. If you hit slide 6 after 5:30, you're on track.
- If behind: compress **slide 5** and **slide 13** — they carry the least unique content.
- Say "**behaviour agreement**", never "accuracy" — your report is careful about this and markers will notice.
- Pause after the Pix2Pix reveal on slide 11. Let it land.
- Pre-load the demo screenshot/clip; never live-network to the Pi mid-talk.

## Q&A — likely questions → where to go
| Question | Answer / backup slide |
|---|---|
| Pseudo-labels — isn't it circular? | Independent probes (DINOv2-L, SigLIP2) held out of training, still close >0.5. |
| What about DINOv2-S / CLIP closures? | **Backup: embedding agreement** — higher, but training-overlapping, so supporting only. |
| Why is detection weaker than segmentation? | Small objects + fine texture reproduced least faithfully; seg/depth lean on large structure. |
| Why NAFNet for a synthesis task? | **Backup: why NAFNet** — activation-free → predictable ranges → clean INT8/edge export. |
| How did you actually align the pairs? | **Backup: producing aligned pairs** — beam-splitter removes parallax → one homography. |
| How general is the method? | **Backup: generalises (and breaks)** — transfers to thermal, breaks on SAR. |
| Why not diffusion? | Iterative + stochastic — violates latency budget and determinism (F3). |
| Why not just fine-tune the downstream models? | Needs weights + labelled NIR per task; out of scope. Translator leaves them unchanged. |
| Pixel metrics for Pix2Pix vs student? | **Backup: image-quality metrics** — level on PSNR/SSIM. |

**If you don't know:** say so, then reason out loud. Honesty is explicitly marked.
