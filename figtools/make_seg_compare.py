"""Parse seg_compare_results.txt -> per-category mean Dice per condition,
a LaTeX-ready table dump, and a grouped bar chart.

Usage: py -3 make_seg_compare.py <seg_compare_results.txt> <out_chart.pdf>
"""
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDS = ["baseline", "ab03", "ab04", "ab06", "ab08", "raw"]
src, out = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8", errors="replace").read().splitlines()

cat = None
# data[cat][cond] = list of per-model mean Dice
data = defaultdict(lambda: defaultdict(list))
val_re = re.compile(r"([01]\.\d+)\s*±")

for ln in lines:
    m = re.match(r"CATEGORY:\s*(\w+)", ln.strip())
    if m:
        cat = m.group(1)
        continue
    if cat and re.match(r"\s*\w+_\w+", ln) and "±" in ln:
        vals = val_re.findall(ln)
        if len(vals) == 6:
            for cond, v in zip(CONDS, vals):
                data[cat][cond].append(float(v))

cats = ["TREE", "PERSON", "ROAD", "BUILDING"]
order = ["raw", "baseline", "ab03", "ab04", "ab06", "ab08"]
label = {"raw": "raw NIR", "baseline": "01 baseline", "ab03": "03 region-L1",
         "ab04": "04 NIR-grad", "ab06": "06 feat-div", "ab08": "08 ensemble"}

means = {c: {k: float(np.mean(data[c][k])) for k in order} for c in cats}
overall = {k: float(np.mean([v for c in cats for v in data[c][k]])) for k in order}

print("=== mean Dice (avg over 9 models) ===")
print("cat       " + "  ".join(f"{label[k]:>12}" for k in order))
for c in cats:
    print(f"{c:9} " + "  ".join(f"{means[c][k]:12.3f}" for k in order))
print(f"{'OVERALL':9} " + "  ".join(f"{overall[k]:12.3f}" for k in order))

print("\n=== LaTeX rows (raw, 01, 03, 04, 06, 08) ===")
for c in cats:
    print(f"{c.title()} & " + " & ".join(f"{means[c][k]:.3f}" for k in order) + r" \\")
print(r"\textbf{Mean} & " + " & ".join(f"{overall[k]:.3f}" for k in order) + r" \\")

# grouped bar chart
fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(cats))
w = 0.14
colors = ["#9e9e9e", "#8c6bb1", "#41ab5d", "#4292c6", "#fb8d3d", "#e6194b"]
for i, k in enumerate(order):
    ax.bar(x + (i - 2.5) * w, [means[c][k] for c in cats], w, label=label[k], color=colors[i])
ax.set_xticks(x); ax.set_xticklabels([c.title() for c in cats])
ax.set_ylabel("mean Dice vs RGB ground truth")
ax.set_ylim(0.5, 1.0); ax.grid(axis="y", alpha=0.3)
ax.legend(ncol=6, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
fig.tight_layout()
fig.savefig(out)
print("\nwrote", out)
