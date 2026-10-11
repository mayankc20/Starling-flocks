"""w and delta0 that change over time, to make the flock change shape, density and mixing while it moves.

The w sweep (experiments/w_sweep.py) found three regimes of the full model:
  w ~ 0.6-0.8   three single-kind subgroups, dense, elongated along motion
  w ~ 0.9-0.95  kinds merge into one looser, partly mixed flock
  w = 1.0       no attraction: sparse, fully mixed, fragments
delta0 is the share of attention given to other kinds (paper: 0.3; 0 = only own kind, 2/3 = kind ignored for G = 3);
it sets the similarity s used to pick partners. Here w(t) and delta0(t) are driven across those regimes, lambda stays 50.
The model itself is unchanged: step() reads p["w"] at every step, and the similarity matrix S is rebuilt from delta0(t).

Schedules (after WARM steps at w = 0.8, delta0 = 0.3):
  constant   w = 0.8, delta0 = 0.3 (control)
  breathing  w(t) = 0.84 + 0.12 sin(2 pi t / P), delta0(t) = 0.3 + 0.25 sin(2 pi t / P):
             tight and segregated (w 0.72, delta0 0.05) <-> loose and mixed (w 0.96, delta0 0.55)
  deep       slower and wider: period 2P, w in [0.75, 1.0], delta0 in [0.0, 0.66] (0.66: kind ignored)
  drift      w and delta0 follow independent slow random walks pulled back to a centre
             (Ornstein-Uhlenbeck), w in [0.6, 1.0], delta0 in [0.0, 0.6]

    python experiments/dynamic_w_delta.py       # 3 schedules x 3 seeds, animations for the first seed
    QUICK=1 python experiments/dynamic_w_delta.py

Results go to DYN_OUT (default output/dynamic).
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

QUICK = os.environ.get("QUICK", "0") == "1"
N, WARM, T, P, EVERY, FRAME_EVERY = 240, 1000, 9000, 2000, 25, 10
SEEDS = [6000, 6001, 6002]
SCHEDULES = os.environ.get("SCHEDULES", "constant,breathing,drift").split(",")
if QUICK:
    N, WARM, T, P, EVERY, FRAME_EVERY, SEEDS = 60, 50, 400, 150, 25, 10, [6000]
DYN_OUT = os.environ.get("DYN_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "dynamic"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))
W0, D0 = 0.8, 0.3


def schedule(name, seed):
    """Arrays w[t], delta0[t] for t = 0..T-1."""
    t = np.arange(T)
    w, dl = np.full(T, W0), np.full(T, D0)
    on = t >= WARM
    s = t[on] - WARM
    if name == "breathing":
        ph = np.sin(2 * np.pi * s / P)
        w[on], dl[on] = 0.84 + 0.12 * ph, D0 + 0.25 * ph
    elif name == "deep":
        ph = np.sin(2 * np.pi * s / (2 * P))
        w[on], dl[on] = 0.875 + 0.125 * ph, 0.33 + 0.33 * ph
    elif name == "drift":
        rng = np.random.default_rng(seed + 777)
        tau_w, tau_l = P / 3, P / 4                         # correlation times, steps
        x, y = W0, D0
        cw, cl = 0.84, 0.3                                  # centres the walks are pulled back to
        sw, sl = 0.10, 0.18                                 # stationary standard deviations
        for k in s:
            x += (cw - x) / tau_w + sw * np.sqrt(2 / tau_w) * rng.standard_normal()
            y += (cl - y) / tau_l + sl * np.sqrt(2 / tau_l) * rng.standard_normal()
            x, y = np.clip(x, 0.6, 1.0), np.clip(y, 0.0, 0.6)
            w[WARM + k], dl[WARM + k] = x, y
    return w, dl


def one(name, seed, frames):
    p = fm.job_params([("N", N)])
    case = fm.CASES["full"]
    dirs = np.linspace(0, 2 * np.pi, p["n_dir"], endpoint=False)
    grid = (dirs, np.cos(dirs), np.sin(dirs))
    X, theta, kind, rng = fm.start(p, seed)
    same = kind[:, None] == kind[None, :]
    ws, dls = schedule(name, seed)
    mem, rows, fr = None, [], []
    for t in range(T + 1):
        if t % EVERY == 0:
            rows.append(dict(schedule=name, seed=seed, t=t, w=ws[min(t, T - 1)], delta0=dls[min(t, T - 1)],
                             **shape(X, theta, kind, p)))
        if frames and t % FRAME_EVERY == 0:
            fr.append((t, X.astype(np.float32), theta.astype(np.float32)))
        if t < T:
            p["w"], p["delta0"] = ws[t], dls[t]
            S = np.where(same, 1.0, fm.similarity_s(p["G"], p["delta0"]))   # delta0 = 0 gives s = 0: other kinds never picked
            X, theta, mem = fm.step(X, theta, mem, kind, S, case, rng, grid, p)
    return rows, (name, seed, kind, fr)


def follow(X, R):
    """Camera centre: centre of the largest group at sight range."""
    lab = fm.groups(X, R)
    return X[lab == np.bincount(lab).argmax()].mean(0)


def movie(name, kind, fr, ts, path, half=45.0):
    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_axes([0.01, 0.02, 0.55, 0.92])
    axs = [fig.add_axes([0.63, 0.70 - 0.29 * i, 0.35, 0.22]) for i in range(3)]
    d = ts[(ts["schedule"] == name) & (ts["seed"] == ts["seed"].min())]
    for a, (k, lab) in zip(axs, [("w", "w (purple), δ (orange)"), ("density", "density"), ("M", "mixing M")]):
        a.plot(d["t"], d[k], color="#4a3aa7", lw=1)
        if k == "w":
            a.plot(d["t"], d["delta0"], color="#eb6834", lw=1)
        a.set_title(lab, fontsize=8, loc="left"); a.tick_params(labelsize=7); fm.style(a)
    cursors = [a.axvline(0, color=fm.INK, lw=0.8) for a in axs]
    cols = np.array(fm.KIND_COLORS)[kind]
    q = None
    title = ax.set_title("", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    def draw(i):
        nonlocal q
        t, X, th = fr[i]
        c = follow(X, fm.BASE["R"])
        Y = X - c
        if q is not None:
            q.remove()
        q = ax.quiver(Y[:, 0], Y[:, 1], np.cos(th), np.sin(th), color=cols, angles="xy", scale_units="xy",
                      scale=0.6, width=0.005, headwidth=3, headlength=3)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        row = d.iloc[np.searchsorted(d["t"].values, t).clip(0, len(d) - 1)]
        title.set_text(f"{name}   step {t}   w = {row['w']:.2f}   δ = {row['delta0']:.2f}")
        for cu in cursors:
            cu.set_xdata([t, t])
        return [q, title] + cursors

    anim = animation.FuncAnimation(fig, draw, frames=len(fr), blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=30, bitrate=2400), dpi=110)
    plt.close(fig)


def strip(name, kind, fr, path, n=8, half=45.0):
    """n snapshots spread over one period after the warm-up."""
    per = 2 * P if name == "deep" else P
    times = WARM + np.linspace(0, per, n, endpoint=False).astype(int) + per  # second period
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.5))
    cols = np.array(fm.KIND_COLORS)[kind]
    byt = {t: (X, th) for t, X, th in fr}
    for ax, t in zip(axes, times):
        t = min(byt, key=lambda u: abs(u - t))
        X, th = byt[t]
        Y = X - follow(X, fm.BASE["R"])
        ax.quiver(Y[:, 0], Y[:, 1], np.cos(th), np.sin(th), color=cols, angles="xy", scale_units="xy",
                  scale=0.6, width=0.008, headwidth=3, headlength=3)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        w, dl = schedule(name, SEEDS[0]); k = min(t, T - 1)
        ax.set_title(f"t={t}\nw={w[k]:.2f} δ={dl[k]:.2f}", fontsize=7)
    fig.suptitle(f"{name}: one period, seed {SEEDS[0]}, camera follows the main group; colour = kind", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def timeseries_plot(ts, path):
    keys = [("w", "w"), ("delta0", "δ (attention to other kinds)"), ("density", "density of largest contact cluster"), ("M", "mixing M"),
            ("clusters_5", "contact clusters ≥5 agents"), ("C", "largest group at sight range C"), ("Phi", "polarization Φ")]
    fig, axes = plt.subplots(len(keys), len(SCHEDULES), figsize=(4.2 * len(SCHEDULES), 1.45 * len(keys)), sharex=True, squeeze=False)
    shades = ["#4a3aa7", "#eb6834", "#1baf7a"]
    for j, name in enumerate(SCHEDULES):
        for i, (k, lab) in enumerate(keys):
            ax = axes[i, j]
            for c, (seed, d) in zip(shades, ts[ts["schedule"] == name].groupby("seed")):
                ax.plot(d["t"], d[k], lw=0.9, color=c, alpha=0.85)
            ax.axvspan(0, WARM, color=fm.GRID, alpha=0.35, lw=0)
            if i == 0:
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
            ax.tick_params(labelsize=7); fm.style(ax)
        axes[-1, j].set_xlabel("step")
    fig.suptitle(f"w and δ over time; one colour per seed (N = {N}); grey: warm-up at w = 0.8, δ = 0.3", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(DYN_OUT, exist_ok=True)
    t0 = time.time()
    jobs = [(n, s) for n in SCHEDULES for s in SEEDS]
    outs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one)(n, s, s == SEEDS[0]) for n, s in jobs)
    ts = pd.DataFrame([r for rows, _ in outs for r in rows])
    ts.to_csv(os.path.join(DYN_OUT, "dynamic_timeseries.csv"), index=False)
    after = ts[ts["t"] >= WARM]
    summ = after.groupby("schedule")[["w", "delta0", "density", "M", "clusters_5", "C", "Phi", "aspect"]].agg(["mean", "std", "min", "max"]).round(3)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(DYN_OUT, "dynamic_summary.csv"))
    timeseries_plot(ts, os.path.join(DYN_OUT, "dynamic_timeseries.png"))
    films = [f for _, f in outs if f[3]]
    for name, seed, kind, fr in films:
        strip(name, kind, fr, os.path.join(DYN_OUT, f"dynamic_{name}_strip.png"))
    Parallel(n_jobs=N_JOBS)(delayed(movie)(name, kind, fr, ts, os.path.join(DYN_OUT, f"dynamic_{name}.mp4"))
                            for name, seed, kind, fr in films)
    pd.set_option("display.width", 220)
    print(after.groupby("schedule")[["density", "M", "clusters_5", "C", "Phi"]].agg(["mean", "std"]).round(3))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {DYN_OUT}")
