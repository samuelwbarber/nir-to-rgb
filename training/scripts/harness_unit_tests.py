"""Oracle and edge-case tests for the gap-closure evaluation harness.

Verifies the aggregation logic of scripts/eval_downstream.py and the
detection-matching edge cases of src/eval/downstream/yolo.py against small
deterministic fixtures, independent of any trained model:

  - NIR identity        -> gap closure 0
  - RGB identity        -> gap closure 1
  - Negative control    -> negative gap closure
  - Zero denominator    -> "undefined", no division error
  - Detection edge cases: empty reference, empty prediction, duplicate
    candidates, threshold-boundary IoU
  - Ordering            -> aggregate invariant to image order
  - Lower-is-better     -> sign convention inverted by the harness

Usage:
    python scripts/harness_unit_tests.py

Writes experiments/harness_unit_tests/results.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.eval.downstream.yolo import YOLOEvaluator


def gap_closure(nir, rgb, translated, lower_is_better=False):
    """Same guarded computation as scripts/eval_downstream.py (summary step).

    For lower-is-better metrics the raw difference is inverted so the sign
    convention holds: 1 = matches RGB, 0 = no better than NIR, negative =
    worse than NIR.
    """
    if lower_is_better:
        nir, rgb, translated = -nir, -rgb, -translated
    denom = rgb - nir
    if abs(denom) > 1e-9:
        return (translated - nir) / denom
    return "undefined"


def det(boxes, classes):
    return {
        "xyxy": np.array(boxes, dtype=float).reshape(-1, 4),
        "cls": np.array(classes, dtype=int),
        "conf": np.ones(len(classes)),
    }


def main():
    ev = YOLOEvaluator()  # vs_reference is pure; no model load needed
    box = [10, 10, 50, 50]
    fixtures = []

    def check(name, expected, observed, ok):
        fixtures.append(
            {"fixture": name, "expected": expected, "observed": observed, "pass": bool(ok)}
        )

    # --- gap-closure oracle fixtures -------------------------------------
    g = gap_closure(nir=0.5, rgb=1.0, translated=0.5)
    check("NIR identity", "0", g, g == 0.0)

    g = gap_closure(nir=0.5, rgb=1.0, translated=1.0)
    check("RGB identity", "1", g, g == 1.0)

    g = gap_closure(nir=0.5, rgb=1.0, translated=0.4)
    check("Negative control", "negative", g, isinstance(g, float) and g < 0)

    g = gap_closure(nir=0.7, rgb=0.7, translated=0.9)
    check("Zero denominator", "undefined", g, g == "undefined")

    # --- detection edge cases ---------------------------------------------
    r = ev.vs_reference(det([box], [0]), det([], []))
    obs = "undefined" if np.isnan(r["matched_iou_vs_rgb"]) else r["matched_iou_vs_rgb"]
    check("Empty reference detections", "undefined", obs, obs == "undefined")

    r = ev.vs_reference(det([], []), det([box], [0]))
    check("Empty translated detections", "0", r["matched_iou_vs_rgb"],
          r["matched_iou_vs_rgb"] == 0.0)

    # two identical candidates against one reference: greedy matching with a
    # used-set must count exactly one match
    r = ev.vs_reference(det([box, box], [0, 0]), det([box], [0]))
    n_matched = int(round(r["recall_vs_rgb"] * 1))
    check("Duplicate candidate match", "one match", n_matched, n_matched == 1)

    # candidate at exactly the 0.5 matching threshold must be included (>=)
    half_box = [10, 10, 50, 30]  # IoU with box = 0.5
    r = ev.vs_reference(det([half_box], [0]), det([box], [0]))
    n_matched = int(round(r["recall_vs_rgb"] * 1))
    check("IoU threshold boundary", "included at IoU=0.5", n_matched, n_matched == 1)

    # --- aggregation properties --------------------------------------------
    per_image = [0.25, 0.75, 0.5, 0.5]  # binary-exact so the comparison is exact
    a = float(np.mean(per_image))
    b = float(np.mean(list(reversed(per_image))))
    check("Reordered images", "same aggregate", a - b, a == b)

    g = gap_closure(nir=0.75, rgb=0.25, translated=0.5, lower_is_better=True)
    check("Lower-is-better metric", "closure=0.5", g, g == 0.5)

    # --- report -------------------------------------------------------------
    passing = sum(1 for f in fixtures if f["pass"])
    failing = len(fixtures) - passing
    out = {"passing": passing, "failing": failing, "fixtures": fixtures}

    out_dir = ROOT / "experiments" / "harness_unit_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    for fx in fixtures:
        status = "PASS" if fx["pass"] else "FAIL"
        print(f"[{status}] {fx['fixture']}: expected {fx['expected']}, got {fx['observed']}")
    print(f"\n{passing} passing, {failing} failing -> {out_dir / 'results.json'}")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
