# Pi 5 / Hailo-8L data collection

Goal: gather the NPU benchmark + reproducibility data needed to (a) close the N1/N2 edge
requirements with a *real* Hailo-8L result, (b) build the three-backend latency table
(CPU-TFLite INT8 / CPU-TFLite FP32 / Hailo NPU), and (c) document the live webapp demo.

Both HEFs are already on the Pi:
- `student_small_nir_to_rgb.hef`  (small width student, 17.68 MB, 9 contexts)
- `student_nir_to_rgb.hef`        (large student, 50.41 MB, 13 contexts)

---

## Part 0 — Run this ONE block, then send me the file

SSH into the Pi and paste this whole block. It writes everything to
`~/hailo_report_data.txt`. When it finishes, send me that file (or paste its contents).

```bash
cd ~
OUT=~/hailo_report_data.txt
SMALL=$(find ~ -name 'student_small_nir_to_rgb.hef' 2>/dev/null | head -1)
LARGE=$(find ~ -name 'student_nir_to_rgb.hef' 2>/dev/null | grep -v small | head -1)

{
echo "######## SECTION 1: SYSTEM / REPRODUCIBILITY ########"
echo "## date"; date -u
echo "## pi model"; cat /proc/device-tree/model; echo
echo "## os"; cat /etc/os-release | grep PRETTY_NAME
echo "## kernel"; uname -a
echo "## cpu governor"; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u
echo "## ram"; free -h
echo "## hailort version"; hailortcli --version
echo "## hailo packages"; dpkg -l | grep -i hailo
echo "## pcie device"; lspci | grep -i hailo
echo "## small hef path: $SMALL"
echo "## large hef path: $LARGE"
echo "## hef sizes"; ls -la "$SMALL" "$LARGE"

echo; echo "######## SECTION 2: DEVICE IDENTIFY ########"
hailortcli fw-control identify

echo; echo "######## SECTION 3: HEF METADATA ########"
echo "=== SMALL ==="; hailortcli parse-hef "$SMALL"
echo "=== LARGE ==="; hailortcli parse-hef "$LARGE"

echo; echo "######## SECTION 4: NPU BENCHMARK (headline numbers) ########"
echo "=== SMALL benchmark (15s) ==="; hailortcli benchmark "$SMALL" -t 15
echo "=== LARGE benchmark (15s) ==="; hailortcli benchmark "$LARGE" -t 15

echo; echo "######## SECTION 5: EXPLICIT LATENCY (hw + full) ########"
echo "=== SMALL latency ==="; hailortcli run "$SMALL" --measure-latency
echo "=== LARGE latency ==="; hailortcli run "$LARGE" --measure-latency

echo; echo "######## DONE ########"
} 2>&1 | tee "$OUT"

echo
echo ">>> Wrote $OUT  (send this file back)"
```

Then get the file to me with **pscp from your Windows machine** (replace IP if different):

```powershell
pscp -pw password pi@<PI_IP>:/home/pi/hailo_report_data.txt C:\Users\sbarb\Downloads\final-report\
```

…or just copy-paste the contents of `hailo_report_data.txt` back into chat.

---

## Part 5b — Sustained thermal + power (run separately, ~60 s)

This is the N2 / thermal-stability evidence. It logs Pi temperature and throttle state
once a second **while** a 60-second NPU benchmark runs, so we can show no throttling.

```bash
SMALL=$(find ~ -name 'student_small_nir_to_rgb.hef' 2>/dev/null | head -1)

# background temp/throttle logger
( for i in $(seq 1 65); do
    printf "%s temp=%s throttled=%s\n" "$(date +%H:%M:%S)" \
      "$(vcgencmd measure_temp)" "$(vcgencmd get_throttled)"
    sleep 1
  done > ~/hailo_thermal_log.txt ) &

# 60-second sustained NPU run
hailortcli benchmark "$SMALL" -t 60 2>&1 | tee ~/hailo_sustained_small.txt
wait
echo ">>> send hailo_thermal_log.txt and hailo_sustained_small.txt"
```

Optional Hailo on-chip power (only works on some Hailo-8L boards; harmless if it errors):

```bash
hailortcli measure-power "$SMALL" -t 15 2>&1 | tee ~/hailo_power.txt
```

---

## Part 6 — CPU vs NPU, apples-to-apples (optional but valuable)

The report's Chapter 8 already has CPU-TFLite numbers (INT8 6.19 FPS, FP32 5.07 FPS via
XNNPACK, 4 threads). If your benchmark harness is still on the Pi, re-run it so the CPU and
NPU numbers come from the **same device, same day, same input size (256×256)**. Send me:

- CPU-TFLite **INT8** mean / p95 latency + FPS
- CPU-TFLite **FP32** mean / p95 latency + FPS
- (NPU numbers come from Part 0 Sections 4–5)

If the harness is gone, just confirm the Chapter 8 numbers are still the ones to cite.

---

## Part 7 — Webapp / live demo details (for the Implementation + Results write-up)

Please tell me (a sentence each is fine):

1. **Which backend the webapp actually runs** the translator on — CPU TFLite, or the
   Hailo HEF? (The screenshot header said `PI CPU TFLITE MODEL`, so I assume CPU for now.)
2. **Per-stage pipeline timing** if the app logs it: capture → preprocess → translate →
   segmentation×2 → display. (The screenshot showed Inference 906 ms, Pipeline 0.8 fps,
   Segmentation pair 276.7 ms — I want the rest of the breakdown if available.)
3. **Segmentation model** used in the demo (name + which dataset/classes — looks like
   Cityscapes palette).
4. **The two segmentation passes** (NIR-seg and generated-RGB-seg) — run on CPU or NPU?
5. **Reference RGB source** in the "RGB REFERENCE (ALIGNED)" panel — is that the co-axial
   camera's RGB sensor, live and homography-aligned?
6. A **clean screenshot or two** of the demo for a figure (you already sent one — more
   scenes welcome).

> Note on the live metrics (PSNR 11.53 dB, SSIM 0.217, Mask agreement 47.8%): these are
> live, unaligned, single-frame, out-of-distribution numbers and are **not** comparable to
> the held-out test metrics (Table 7.1). I'll frame them in the report as a qualitative
> live demonstration only — but confirm the capture really is unaligned so I caption it
> correctly.

---

## What each piece feeds in the report

| Data | Report use |
|------|-----------|
| Section 1 (system/versions) | AppendixB reproducibility + benchmark methodology (Ch. 6) |
| Section 2–3 (identify, parse-hef) | Confirms Hailo-8L arch, contexts, I/O shapes (Ch. 5) |
| Section 4–5 (benchmark, latency) | **Headline NPU FPS/latency** → three-backend table, closes N1 |
| Part 5b (thermal/power) | Sustained-stability evidence → closes N2 |
| Part 6 (CPU re-run) | Apples-to-apples CPU-vs-NPU speedup table |
| Part 7 (webapp) | Implementation §Edge Deployment + Results live-demo figure |
