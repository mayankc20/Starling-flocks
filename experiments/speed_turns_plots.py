"""Figures and a video from speed_turns.py (reads ST_OUT, default output/speed_turns).

    python experiments/speed_turns_plots.py           # figures
    VIDEO=1 python experiments/speed_turns_plots.py   # also a video of the combined manoeuvre (ffmpeg)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mindflock_speed as ms
from speed_turns import settings, PULSE

OUT = os.environ.get("ST_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "speed_turns"))
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3dd", "#a3a29c"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
KIND = [BLUE, ORANGE, GREEN]
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "savefig.dpi": 200,
                     "legend.frameon": False})


def dotpanel(ax, df, labels, col, title, xlim=None, colour=None):
    g = df.groupby("setting")[col]
    m, lo, hi = g.mean(), g.min(), g.max()
    for y, lb in enumerate(labels[::-1]):
        c = colour(lb) if colour else BLUE
        ax.plot([lo[lb], hi[lb]], [y, y], color=c, lw=2, solid_capstyle="round", alpha=0.6)
        ax.plot(m[lb], y, "o", color=c, ms=5.5, mec="white", mew=1.2)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels[::-1])
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    if xlim: ax.set_xlim(*xlim)


def rule_colour(lb):
    return {"const": MUTED, "match": BLUE, "born": ORANGE}.get(lb.split("| ")[-1].split(",")[0], INK2)


def fig_speed(df):
    labels = [lb for pt, lb, _ in settings() if pt == "speed"]
    d = df[df.part == "speed"]
    fig, axs = plt.subplots(1, 4, figsize=(12, 4.6), sharey=True)
    dotpanel(axs[0], d, labels, "seg", "segregation (≥ 0.5 = segregated)", (-0.2, 1), rule_colour)
    dotpanel(axs[1], d, labels, "C_R_min", "lowest share in largest group", (0, 1.05), rule_colour)
    dotpanel(axs[2], d, labels, "collisions_per_drone_min", "collisions / drone / min", None, rule_colour)
    dotpanel(axs[3], d, labels, "speed_sd", "spread of speeds (BL/step)", None, rule_colour)
    axs[0].axvline(0.5, color=GRID, lw=1); axs[1].axvline(0.99, color=GRID, lw=1)
    fig.suptitle("Variable speed: grey = constant speed rule, blue = speed matching, orange = Born rule on speed "
                 "(dot = mean, bar = range over 6 seeds)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_speed.png"), bbox_inches="tight"); plt.close(fig)


def fig_turns(df):
    d = df[(df.part == "turns") & ~df.setting.str.contains("standing")].copy()
    d["frac"] = d.setting.str.extract(r"^(\d+) %").astype(int)
    d["angle"] = d.setting.str.extract(r"\| (\d+) deg").astype(int)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    cols = {20: MUTED, 50: ORANGE, 100: BLUE}
    for fr, g in d.groupby("frac"):
        m = g.groupby("angle").mean(numeric_only=True)
        kw = dict(color=cols[fr], marker="o", ms=4, label=f"{fr} % informed")
        axs[0].plot(m.index, m.turn_lag_s, **kw)
        axs[1].plot(m.index, m.turn_followed, **kw)
        axs[2].plot(m.index, m.turn_C_R_min, **kw)
        axs[3].plot(m.index, m.turn_coll, **kw)
    for ax, t in zip(axs, ["time to within 20° of new goal (s)\n(15 = not within 15 s)",
                           "share of turns completed\nwithin 15 s", "lowest share in largest group\nin the 15 s after a turn",
                           "collisions per turn\n(whole flock, 15 s after)"]):
        ax.set_title(t, loc="left", fontsize=8.5); ax.set_xticks([45, 90, 135, 180]); ax.set_xlabel("turn angle (deg)")
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_turns.png"), bbox_inches="tight"); plt.close(fig)


def fig_traces(seed=200):
    man = dict(PULSE, frac_turn=1.0, turn_deg=90, regime="steps", speed_rule="match")
    r = ms.run(seed, ms.params(**man), 3000)
    d = r["rec"]; t = d.t * 0.1
    fig, axs = plt.subplots(4, 1, figsize=(8, 6.5), sharex=True)
    axs[0].plot(t, d.cmd, color=MUTED, lw=1.2, label="commanded speed")
    axs[0].plot(t, d.speed, color=BLUE, lw=1.2, label="mean speed of the flock")
    axs[0].fill_between(t, d.speed - d.speed_sd, d.speed + d.speed_sd, color=BLUE, alpha=0.15, lw=0)
    axs[0].set_ylabel("BL / step"); axs[0].legend(fontsize=7.5, ncol=2, loc="upper left")
    axs[1].plot(t, d.goal_err_deg, color=ORANGE, lw=1.2); axs[1].set_ylabel("heading off goal (deg)")
    axs[2].plot(t, d.seg, color=BLUE, lw=1.2); axs[2].set_ylabel("segregation"); axs[2].set_ylim(-0.2, 1)
    axs[3].plot(t, d.C_R, color=INK2, lw=1.2); axs[3].set_ylabel("share in\nlargest group"); axs[3].set_ylim(0, 1.05)
    axs[3].set_xlabel("time (s)")
    for ax in axs:
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].set_title("Speed steps every 30 s and 90° turn commands every 20 s to all drones (speed matching rule)",
                     loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_traces.png"), bbox_inches="tight"); plt.close(fig)


def video(seed=200, T=1200, every=4):
    import matplotlib.animation as animation
    man = dict(PULSE, frac_turn=1.0, turn_deg=90, regime="steps", speed_rule="match", step_s=20.0)
    r = ms.run(seed, ms.params(**man), T, every=every, frames_every=every)
    kind = r["final"][2]
    rec = r["rec"]
    trail = [f[0].mean(0) for f in r["frames"]]
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection="3d"); ax2 = fig.add_subplot(122)

    def draw(i):
        ax.cla(); ax2.cla()
        X, V, alive = r["frames"][i]
        c = X.mean(0); Y = X - c
        for g in range(3):
            s = kind == g
            ax.scatter(*Y[s].T, s=16, color=KIND[g], depthshade=False, edgecolors="white", linewidths=0.4)
        ax.set_xlim(-15, 15); ax.set_ylim(-15, 15); ax.set_zlim(-15, 15)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        row = rec.iloc[i]
        ax.set_title(f"t = {i * every * 0.1:.0f} s   command {row.cmd:.2f}, mean speed {row.speed:.2f} BL/step\n"
                     f"seg {row.seg:.2f}   heading off goal {row.goal_err_deg:.0f}°", fontsize=8.5)
        tr = np.array(trail[:i + 1])
        ax2.plot(tr[:, 0], tr[:, 1], color=MUTED, lw=1)
        for g in range(3):
            s = kind == g
            ax2.scatter(X[s, 0], X[s, 1], s=6, color=KIND[g])
        allt = np.array(trail)
        ax2.set_xlim(allt[:, 0].min() - 20, allt[:, 0].max() + 20); ax2.set_ylim(allt[:, 1].min() - 20, allt[:, 1].max() + 20)
        ax2.set_aspect("equal"); ax2.set_title("path seen from above", fontsize=8.5)
    ani = animation.FuncAnimation(fig, draw, frames=len(r["frames"]))
    ani.save(os.path.join(OUT, "video_speed_turns.mp4"), writer=animation.FFMpegWriter(fps=25, bitrate=2000))
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(OUT, "per_run.csv"))
    fig_speed(df); fig_turns(df); fig_traces()
    if os.environ.get("VIDEO") == "1":
        video()
