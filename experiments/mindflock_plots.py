"""Figures and a video from the results of mindflock_compare.py (reads MF_OUT, default output/mindflock3d).

    python experiments/mindflock_plots.py            # figures
    VIDEO=1 python experiments/mindflock_plots.py    # also 3D videos (ffmpeg)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mindflock3d as mf

OUT = os.environ.get("MF_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "mindflock3d"))
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3dd", "#a3a29c"
BLUE, ORANGE = "#2a78d6", "#eb6834"
KIND = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "savefig.dpi": 200,
                     "legend.frameon": False})
ORDER = mf.CONTROLLERS
LAB = mf.LABEL


def dotpanel(ax, df, col, title, xlim=None):
    g = df.groupby("ctrl")[col]
    m, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
    for y, c in enumerate(ORDER[::-1]):
        col_ = BLUE if c == "mindflock" else (INK2 if c == "mindflock_plus" else MUTED)
        ax.plot([lo[c], hi[c]], [y, y], color=col_, lw=2, solid_capstyle="round")
        ax.plot(m[c], y, "o", color=col_, ms=6, mec="white", mew=1.5)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([LAB[c] for c in ORDER[::-1]])
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    if xlim: ax.set_xlim(*xlim)


def fig_main():
    df = pd.read_csv(os.path.join(OUT, "main_per_run.csv"))
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
    dotpanel(axs[0], df, "seg", "Segregation (0 mixed, 1 fully sorted)", (-0.1, 1.05))
    dotpanel(axs[1], df, "C_R_min", "Cohesion: lowest share in largest\nconnected group (range R)", (0, 1.05))
    dotpanel(axs[2], df, "collisions_per_drone_min", "Collisions per drone per minute")
    dotpanel(axs[3], df, "d_min", "Closest approach (drone diameters)")
    axs[3].axvline(1, color=ORANGE, lw=1, ls=(0, (3, 2))); axs[3].text(1.05, 6.3, "collision", color=INK2, fontsize=7.5)
    fig.text(0.01, -0.04, "N = 30 drones, 3 kinds, 10 runs of 300 s each; dot = mean, bar = 10th to 90th percentile.",
             color=INK2, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_main.png"), bbox_inches="tight"); plt.close(fig)


def fig_delta():
    df = pd.read_csv(os.path.join(OUT, "delta_summary.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.8))
    for c, col, lab in (("mindflock", BLUE, LAB["mindflock"]), ("mindflock_plus", INK2, LAB["mindflock_plus"])):
        d = df[df.ctrl == c]
        axs[0].plot(d.delta0, d.seg, "-o", color=col, lw=2, ms=5, mec="white", label=lab)
        axs[1].plot(d.delta0, d.seg_and_cohesive, "-o", color=col, lw=2, ms=5, mec="white", label=lab)
    axs[0].set_title("Segregation", loc="left", fontsize=9); axs[1].set_title("Share of runs segregated (≥ 0.5) and whole", loc="left", fontsize=9)
    for a in axs:
        a.set_xlabel("δ, attention to other kinds"); a.yaxis.grid(True, color=GRID, lw=0.6); a.set_ylim(-0.05, 1.05)
    axs[0].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_delta.png"), bbox_inches="tight"); plt.close(fig)


def fig_robust():
    df = pd.read_csv(os.path.join(OUT, "robust_summary.csv"))
    conds = list(dict.fromkeys(df.condition))
    metrics = [("seg", "Segregation", "Blues", (0, 1)), ("cohesive_share", "Runs that stayed whole", "Blues", (0, 1)),
               ("collisions_per_drone_min", "Collisions per drone per min", "Oranges", None)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    for ax, (col, title, cmap, lim) in zip(axs, metrics):
        Z = df.pivot(index="condition", columns="ctrl", values=col).loc[conds, ORDER].values
        vmax = lim[1] if lim else max(np.nanmax(Z), 1e-3)
        ax.imshow(Z, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z[i, j]
                ax.text(j, i, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.6 * vmax else INK)
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LAB[c] for c in ORDER], rotation=40, ha="right")
        ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds)
        ax.set_title(title, loc="left", fontsize=9)
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_robust.png"), bbox_inches="tight"); plt.close(fig)


def fig_snapshots(T=3000, seed=100):
    fig = plt.figure(figsize=(13, 6))
    for k, c in enumerate(ORDER):
        r = mf.run(c, seed, dict(mf.BASE), T, every=100)
        X, V, kind, alive = r["final"]
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        Y = X - X.mean(0)
        for g in range(3):
            s = kind == g
            ax.scatter(*Y[s].T, s=14, color=KIND[g], depthshade=False, edgecolors="white", linewidths=0.4)
        lim = max(10, np.percentile(np.abs(Y), 90))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_title(f"{LAB[c]}\nseg {r['summary']['seg']:.2f}, whole {r['summary']['C_R_min']:.2f}", fontsize=8)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_snapshots.png"), bbox_inches="tight"); plt.close(fig)


def video(ctrl, seed=100, T=3000, every=10, changes=None, name=None):
    import matplotlib.animation as animation
    r = mf.run(ctrl, seed, dict(mf.BASE, **(changes or {})), T, every=100, frames_every=every)
    fig = plt.figure(figsize=(5.5, 5.5)); ax = fig.add_subplot(111, projection="3d")
    kind = r["final"][2]

    def draw(i):
        ax.cla()
        X, V, alive = r["frames"][i]
        c = X[alive].mean(0); Y = X - c
        for g in range(3):
            s = (kind == g) & alive
            ax.scatter(*Y[s].T, s=16, color=KIND[g], depthshade=False, edgecolors="white", linewidths=0.4)
        ax.set_xlim(-15, 15); ax.set_ylim(-15, 15); ax.set_zlim(-15, 15)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_title(f"{LAB[ctrl]}   t = {i * every * 0.1:.0f} s", fontsize=9)
    ani = animation.FuncAnimation(fig, draw, frames=len(r["frames"]))
    ani.save(os.path.join(OUT, name or f"video_{ctrl}.mp4"), writer=animation.FFMpegWriter(fps=20, bitrate=1800))
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_main, fig_delta, fig_robust):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    fig_snapshots()
    if os.environ.get("VIDEO") == "1":
        for c in ("mindflock", "boids_signed", "mindflock_plus"):
            video(c)
    print("figures in", OUT)
