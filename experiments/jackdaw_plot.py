"""Figure: jackdaw flocks (Dryad) vs single-kind 2D model, label-free statistics side by side."""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
data = pd.read_csv(sys.argv[1]); model = pd.read_csv(sys.argv[2]); out = sys.argv[3]
model["w"] = model["label"].str.extract(r"w=([\d.]+)").astype(float)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d2"
C_DATA, C_W8, C_W9 = "#2a78d6", "#eb6834", "#1baf7a"
panels = [("nn", "Nearest-neighbour distance\n(body lengths, horizontal)"), ("phi", "Polarization Φ"),
          ("aspect", "Flock aspect ratio\n(long / short axis)"), ("cos_long", "Long axis along heading\n|cos angle|"),
          ("xi_rel", "Correlation length\n/ flock size"), ("persist_1.0s", "Neighbours kept after 1 s\n(share of 6 nearest)")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})
fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
groups = [("Jackdaw flocks\n(6 events)", data, C_DATA), ("Model\nw = 0.8", model[model.w == 0.8], C_W8), ("Model\nw = 0.9", model[model.w == 0.9], C_W9)]
for ax, (col, title) in zip(axes.flat, panels):
    for i, (lab, d, c) in enumerate(groups):
        y = d[col].to_numpy()
        x = i + np.linspace(-0.12, 0.12, len(y)) if len(y) > 1 else [i]
        ax.scatter(x, y, s=36, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        ax.hlines(np.nanmedian(y), i - 0.25, i + 0.25, color=INK2, lw=2, zorder=2)
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(-0.6, 2.6)
fig.suptitle("Jackdaw flocks vs single-kind 2D model (dots = runs/flocks, bar = median)", x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(out, dpi=200)
print("saved", out)
