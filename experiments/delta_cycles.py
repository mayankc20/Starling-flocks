"""Cycles of mixing and segregation from delta0(t), with w fixed at 0.8.

delta0 is the share of attention given to other kinds (2/3 = kind ignored for G = 3). The flock starts fully mixed and
is never given a sorting phase. Runs:
  const_X    delta0 = X fixed (X = 0.4, 0.5, 0.6): where does mixing settle, and how fast?
  cycle_P    delta0(t) = 0.45 + 0.05 cos(2 pi t / P), i.e. 0.5 -> 0.4 -> 0.5, for P = 1000, 2000, 4000 steps
             (100, 200, 400 s at 0.1 s per step)
The range can be changed with D_LO, D_HI; the periods with PERIODS; ORDERED=1 starts with all headings aligned.

    python experiments/delta_cycles.py
    QUICK=1 python experiments/delta_cycles.py

Results go to DC_OUT (default output/delta_cycles).
"""
import os, sys, time
os.environ.setdefault("OUT", os.path.join(os.path.dirname(__file__), "..", "output", "model_cache"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from joblib import Parallel, delayed
import flock_model as fm
from w_sweep import shape

DT, W = 0.1, 0.8
D_LO, D_HI = float(os.environ.get("D_LO", 0.4)), float(os.environ.get("D_HI", 0.5))
CONSTS = [float(x) for x in os.environ.get("CONSTS", "0.4,0.5,0.6").split(",") if x]
PERIODS = [int(x) for x in os.environ.get("PERIODS", "1000,2000,4000").split(",") if x]
N, T, EVERY, FRAME_EVERY = 240, int(os.environ.get("T", 12000)), 20, 10
ORDERED = os.environ.get("ORDERED", "0") == "1"
SEEDS = [6100, 6101]
if os.environ.get("QUICK", "0") == "1":
    N, T, EVERY, FRAME_EVERY, SEEDS, CONSTS, PERIODS = 60, 300, 10, 10, [6100], [0.5], [100]
DC_OUT = os.environ.get("DC_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "delta_cycles"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))


def delta(label, t):
    kind, x = label.split("_")
    if kind == "const":
        return float(x)
    P = int(x)
    return 0.5 * (D_LO + D_HI) + 0.5 * (D_HI - D_LO) * np.cos(2 * np.pi * t / P)


def one(label, seed, film):
    p = fm.job_params([("N", N), ("w", W)])
    case = fm.CASES["full"]
    dirs = np.linspace(0, 2 * np.pi, p["n_dir"], endpoint=False)
    grid = (dirs, np.cos(dirs), np.sin(dirs))
    X, theta, kind, rng = fm.start(p, seed)
    if ORDERED:                                              # mixed but already flying one way: avoids early break-offs
        theta = (0.1 * rng.standard_normal(N)) % (2 * np.pi)
    same = kind[:, None] == kind[None, :]
    mem, rows, fr = None, [], []
    for t in range(T + 1):
        d = delta(label, t)
        if t % EVERY == 0:
            rows.append(dict(run=label, seed=seed, t=t, delta0=d, **shape(X, theta, kind, p)))
        if film and t % FRAME_EVERY == 0:
            fr.append((t, X.astype(np.float32), theta.astype(np.float32), d))
        if t < T:
            p["delta0"] = d
            S = np.where(same, 1.0, fm.similarity_s(p["G"], d))
            X, theta, mem = fm.step(X, theta, mem, kind, S, case, rng, grid, p)
    return rows, (label, seed, kind, fr)


def movie(label, kind, fr, ts, path, half=40.0):
    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_axes([0.01, 0.02, 0.55, 0.9])
    axs = [fig.add_axes([0.63, 0.70 - 0.29 * i, 0.35, 0.22]) for i in range(3)]
    d = ts[(ts["run"] == label) & (ts["seed"] == ts["seed"].min())]
    for a, (k, lab, c) in zip(axs, [("delta0", "δ (w = 0.8 fixed)", "#eb6834"), ("M", "mixing M", "#1baf7a"),
                                    ("density", "density", "#4a3aa7")]):
        a.plot(d["t"] * DT, d[k], color=c, lw=1)
        a.set_title(lab, fontsize=8, loc="left"); a.tick_params(labelsize=7); fm.style(a)
    axs[-1].set_xlabel("s (0.1 s per step)", fontsize=7)
    cursors = [a.axvline(0, color=fm.INK, lw=0.8) for a in axs]
    cols = np.array(fm.KIND_COLORS)[kind]
    q, title, c0 = None, ax.set_title("", fontsize=9), None
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    def draw(i):
        nonlocal q, c0
        t, X, th, dl = fr[i]
        lab = fm.groups(X, fm.BASE["R"])
        c = X[lab == np.bincount(lab).argmax()].mean(0)
        c0 = c if c0 is None else 0.8 * c0 + 0.2 * c
        Y = X - c0
        if q is not None:
            q.remove()
        q = ax.quiver(Y[:, 0], Y[:, 1], np.cos(th), np.sin(th), color=cols, angles="xy", scale_units="xy",
                      scale=0.6, width=0.005, headwidth=3, headlength=3)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        title.set_text(f"{label}   t = {t * DT:5.0f} s   δ = {dl:.3f}")
        for cu in cursors:
            cu.set_xdata([t * DT, t * DT])
        return [q, title] + cursors

    anim = animation.FuncAnimation(fig, draw, frames=len(fr), blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=30, bitrate=2400), dpi=110)
    plt.close(fig)


def traces_plot(ts, path):
    runs = list(dict.fromkeys(ts["run"]))
    keys = [("delta0", "δ"), ("M", "mixing M"), ("density", "density"), ("clusters_5", "contact clusters ≥5"),
            ("C", "one group at sight range, C"), ("Phi", "polarization Φ")]
    fig, axes = plt.subplots(len(keys), len(runs), figsize=(2.9 * len(runs), 1.45 * len(keys)), sharex=True, sharey="row", squeeze=False)
    shades = ["#4a3aa7", "#eb6834", "#1baf7a"]
    for j, r in enumerate(runs):
        for i, (k, lab) in enumerate(keys):
            ax = axes[i, j]
            for c, (s, d) in zip(shades, ts[ts["run"] == r].groupby("seed")):
                ax.plot(d["t"] * DT, d[k], lw=0.8, color=c)
            if i == 0:
                ax.set_title(r, fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
            ax.tick_params(labelsize=7); fm.style(ax)
        axes[-1, j].set_xlabel("s")
    fig.suptitle(f"w = {W}, mixed start, N = {N}; one colour per seed", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(DC_OUT, exist_ok=True)
    t0 = time.time()
    labels = [f"const_{x:g}" for x in CONSTS] + [f"cycle_{P}" for P in PERIODS]
    jobs = [(l, s) for l in labels for s in SEEDS]
    outs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one)(l, s, l.startswith("cycle") and s == SEEDS[0]) for l, s in jobs)
    ts = pd.DataFrame([r for rows, _ in outs for r in rows])
    ts.to_csv(os.path.join(DC_OUT, "delta_cycles_timeseries.csv"), index=False)
    traces_plot(ts, os.path.join(DC_OUT, "delta_cycles_traces.png"))
    late = ts[ts["t"] >= T // 2]
    summ = late.groupby("run")[["M", "density", "clusters_5", "C", "Phi"]].agg(["mean", "min", "max"]).round(3)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(DC_OUT, "delta_cycles_summary.csv"))
    films = [f for _, f in outs if f[3]]
    Parallel(n_jobs=N_JOBS)(delayed(movie)(l, kind, fr, ts, os.path.join(DC_OUT, f"delta_{l}.mp4")) for l, s, kind, fr in films)
    pd.set_option("display.width", 250)
    print(summ)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {DC_OUT}")
