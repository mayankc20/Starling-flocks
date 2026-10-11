"""How fast can w and delta0 be cycled before the flock stops following?

Time scale: one model step is read as 0.1 s (the order of a starling's reaction time). The agent then moves
v0 = 0.5 body lengths per step = 5 body lengths/s, and the perceptual memory mu = 0.95 forgets over ~20 steps = 2 s.

Each run first sorts at w = 0.8, delta0 = 0.3 for WARM steps, then follows the "breathing" schedule of
experiments/dynamic_w_delta.py at period P:
    w(t) = 0.84 + 0.12 sin(2 pi t / P),   delta0(t) = 0.3 + 0.25 sin(2 pi t / P)
for P from 2 s to 200 s. The response of each measure is the amplitude of its component at the driving period
(and the phase lag behind the drive).

    python experiments/fast_cycles.py
    QUICK=1 python experiments/fast_cycles.py

Results go to FAST_OUT (default output/fast_cycles).
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

DT = 0.1                                   # seconds per step
QUICK = os.environ.get("QUICK", "0") == "1"
N, WARM, RUN, EVERY = 240, 3000, 4000, 2
PERIODS = [20, 50, 100, 200, 500, 2000]    # steps: 2, 5, 10, 20, 50, 200 s
SEEDS = [7000, 7001, 7002]
MOVIE_P, MOVIE_STEPS = 100, 600            # animate the last 600 steps (60 s) at this period
if QUICK:
    N, WARM, RUN, PERIODS, SEEDS, MOVIE_STEPS = 60, 100, 200, [20, 100], [7000], 100
FAST_OUT = os.environ.get("FAST_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "fast_cycles"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))
KEYS = ["density", "M", "clusters_5", "aspect", "Phi", "nn"]


def drive(t, P):
    if t < WARM:
        return 0.8, 0.3
    ph = np.sin(2 * np.pi * (t - WARM) / P)
    return 0.84 + 0.12 * ph, 0.3 + 0.25 * ph


def one(P, seed, film):
    p = fm.job_params([("N", N)])
    case = fm.CASES["full"]
    dirs = np.linspace(0, 2 * np.pi, p["n_dir"], endpoint=False)
    grid = (dirs, np.cos(dirs), np.sin(dirs))
    X, theta, kind, rng = fm.start(p, seed)
    same = kind[:, None] == kind[None, :]
    T = WARM + RUN
    mem, rows, fr = None, [], []
    for t in range(T + 1):
        w, d = drive(t, P)
        if t >= WARM and t % EVERY == 0:
            rows.append(dict(P=P, seed=seed, t=t, w=w, delta0=d, **shape(X, theta, kind, p)))
        if film and t >= T - MOVIE_STEPS:
            fr.append((t, X.astype(np.float32), theta.astype(np.float32), w, d))
        if t < T:
            p["w"], p["delta0"] = w, d
            S = np.where(same, 1.0, fm.similarity_s(p["G"], d))
            X, theta, mem = fm.step(X, theta, mem, kind, S, case, rng, grid, p)
    return rows, (P, seed, kind, fr)


def response(d, P):
    """Amplitude and lag (s) of each measure at the driving period; drive phase is sin(2 pi (t-WARM)/P)."""
    t = d["t"].values - WARM
    z = np.exp(-2j * np.pi * t / P)
    out = {}
    for k in KEYS:
        y = d[k].values - d[k].values.mean()
        c = 2 * (y * z).mean()                                 # y ~ |c| sin(2 pi t / P + phi)
        amp, phi = abs(c), np.angle(c * 1j)                   # phase relative to the sine drive
        out[f"{k}_amp"] = amp
        out[f"{k}_lag_s"] = (-phi % (2 * np.pi)) / (2 * np.pi) * P * DT
        out[f"{k}_mean"] = d[k].mean()
    return out


def response_plot(resp, path):
    panels = [("density", "density of main cluster"), ("M", "mixing M"), ("clusters_5", "contact clusters ≥5"),
              ("aspect", "elongation")]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.0))
    g = resp.groupby("P")
    for ax, (k, lab) in zip(axes, panels):
        x = resp["P"] * DT
        ax.plot(x, resp[f"{k}_amp"], "o", ms=3, color="#9a9994", alpha=0.7)
        m = g[f"{k}_amp"].mean()
        ax.plot(m.index * DT, m.values, "-", color="#4a3aa7", lw=1.6)
        ax.set_xscale("log"); ax.set_xlabel("period of the cycle (s, at 0.1 s/step)")
        ax.set_title(f"{lab}: swing amplitude", fontsize=9); fm.style(ax)
    fig.suptitle("How much the flock follows a w/δ cycle of a given period (3 seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def traces_plot(ts, path):
    Ps = sorted(ts["P"].unique())
    fig, axes = plt.subplots(3, len(Ps), figsize=(2.9 * len(Ps), 6), sharey="row", squeeze=False)
    for j, P in enumerate(Ps):
        d = ts[(ts["P"] == P) & (ts["seed"] == ts["seed"].min())]
        span = min(len(d), max(int(4 * P / EVERY), int(400 / EVERY)))     # four periods, at least 40 s
        d = d.iloc[-span:]
        tt = (d["t"] - d["t"].iloc[0]) * DT
        axes[0, j].plot(tt, d["w"], color="#4a3aa7", lw=1); axes[0, j].plot(tt, d["delta0"], color="#eb6834", lw=1)
        axes[1, j].plot(tt, d["density"], color="#4a3aa7", lw=1)
        axes[2, j].plot(tt, d["M"], color="#1baf7a", lw=1)
        axes[0, j].set_title(f"period {P * DT:g} s", fontsize=9)
        axes[2, j].set_xlabel("s")
        for i in range(3):
            fm.style(axes[i, j]); axes[i, j].tick_params(labelsize=7)
    axes[0, 0].set_ylabel("w (purple), δ (orange)", fontsize=8)
    axes[1, 0].set_ylabel("density", fontsize=8); axes[2, 0].set_ylabel("mixing M", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def movie(kind, fr, path, half=35.0):
    fig, ax = plt.subplots(figsize=(5.5, 5.8))
    cols = np.array(fm.KIND_COLORS)[kind]
    q, title = None, ax.set_title("", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    c0 = None

    def draw(i):
        nonlocal q, c0
        t, X, th, w, d = fr[i]
        lab = fm.groups(X, fm.BASE["R"])
        c = X[lab == np.bincount(lab).argmax()].mean(0)
        c0 = c if c0 is None else 0.8 * c0 + 0.2 * c                      # smooth camera
        Y = X - c0
        if q is not None:
            q.remove()
        q = ax.quiver(Y[:, 0], Y[:, 1], np.cos(th), np.sin(th), color=cols, angles="xy", scale_units="xy",
                      scale=0.6, width=0.005, headwidth=3, headlength=3)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        title.set_text(f"cycle period {MOVIE_P * DT:g} s   time {(t - fr[0][0]) * DT:5.1f} s   w = {w:.2f}   δ = {d:.2f}")
        return [q, title]

    anim = animation.FuncAnimation(fig, draw, frames=len(fr), blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=20, bitrate=2400), dpi=110)    # 20 fps = 2x real time
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FAST_OUT, exist_ok=True)
    t0 = time.time()
    jobs = [(P, s) for P in PERIODS for s in SEEDS]
    outs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one)(P, s, P == MOVIE_P and s == SEEDS[0]) for P, s in jobs)
    ts = pd.DataFrame([r for rows, _ in outs for r in rows])
    ts.to_csv(os.path.join(FAST_OUT, "fast_timeseries.csv"), index=False)
    resp = pd.DataFrame([dict(P=P, period_s=P * DT, seed=s, **response(d, P)) for (P, s), d in ts.groupby(["P", "seed"])])
    resp.to_csv(os.path.join(FAST_OUT, "fast_response_per_run.csv"), index=False)
    summ = resp.drop(columns="seed").groupby(["P", "period_s"]).mean().round(3)
    summ.to_csv(os.path.join(FAST_OUT, "fast_response_summary.csv"))
    response_plot(resp, os.path.join(FAST_OUT, "fast_response.png"))
    traces_plot(ts, os.path.join(FAST_OUT, "fast_traces.png"))
    for P, s, kind, fr in [f for _, f in outs if f[3]]:
        movie(kind, fr, os.path.join(FAST_OUT, f"fast_period_{P * DT:g}s.mp4"))
    pd.set_option("display.width", 220)
    print(summ[[c for c in summ.columns if c.endswith("_amp") or c.startswith("density_lag")]])
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {FAST_OUT}")
