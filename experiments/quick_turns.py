"""Quick turns: no memory (mu = 0) and an escape cue in the perceptual state.

Time scale: one step = 0.1 s; agents move v0 = 1 body length per step (10 body lengths/s).
A "predator" appears at a random edge of the flock (3 body lengths beyond the edge agent farthest from the centre in a
random direction) on average every TRIGGER steps and follows the flock. A bird that reacts adds one extra term to its
perceptual state: the direction away from the predator (DODGE=radial) or one dodge shared by all reacting birds
(DODGE=common), with amplitude A_ESC (ESC2 = A_ESC**2, default 6 = as strong as its six partners together). Modes:
  none      no predator (control for mu = 0)
  imposed   the reaction spreads as in local_waves.py: a resting bird reacts with prob. Q per step if a neighbour within
            R_WAVE reacts; reacting birds also take w = 0.96, delta0 = 0.55; react for TAU_E, then refractory TAU_R
  self      only the birds within R_WAVE of the edge agent react (for TAU_E); everyone else keeps w = 0.8, delta0 = 0.3,
            so any further spread of the turn comes from alignment alone

    python experiments/quick_turns.py
    QUICK=1 python experiments/quick_turns.py

Results go to TURN_OUT (default output/quick_turns).
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
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import flock_model as fm
from starling_ext import step_ext
from w_sweep import shape

DT = 0.1
V0, MU = float(os.environ.get("V0", 1.0)), float(os.environ.get("MU", 0.0))
A_ESC = np.sqrt(float(os.environ.get("ESC2", 6.0)))
W_REST, D_REST, W_EXC, D_EXC = 0.8, 0.3, 0.96, 0.55
TAU_E, TAU_R, Q, R_WAVE, RELAX = 10, 30, 0.6, 3.0, 0.3
TRIGGER = int(os.environ.get("TRIGGER", 50))      # mean steps between predator attacks: 5 s
MODES = os.environ.get("MODES", "none,imposed,self").split(",")
DODGE = os.environ.get("DODGE", "radial")          # radial: each bird flees the predator point;
                                                   # common: all reacting birds take the same dodge, the flock heading
                                                   # turned 90 degrees away from the predator's side
N, WARM, RUN, EVERY = 240, 3000, 1500, 2
SEEDS = [9000, 9001, 9002]
MOVIE_STEPS = 600
QUICK = os.environ.get("QUICK", "0") == "1"
if QUICK:
    N, WARM, RUN, SEEDS, MOVIE_STEPS = 60, 100, 200, [9000], 100
TURN_OUT = os.environ.get("TURN_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "quick_turns"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))
REST, EXC, REF = 0, 1, 2


def one(mode, seed, film):
    p = fm.job_params([("N", N), ("v0", V0), ("mu", MU)])
    dirs = np.linspace(0, 2 * np.pi, p["n_dir"], endpoint=False)
    grid = (dirs, np.cos(dirs), np.sin(dirs))
    X, theta, kind, rng = fm.start(p, seed)
    wrng = np.random.default_rng(seed + 99)
    same = kind[:, None] == kind[None, :]
    state, clock = np.zeros(N, int), np.zeros(N, int)
    w, dl = np.full(N, W_REST), np.full(N, D_REST)
    pred, attacks = None, 0
    T = WARM + RUN
    mem, rows, fr = None, [], []
    head_prev = None
    for t in range(T + 1):
        c = X.mean(0)
        if pred is not None:
            pred = pred + (c - c_prev)                                   # the predator follows the flock
        c_prev = c
        if t >= WARM and mode != "none":
            if wrng.random() < 1 / TRIGGER:
                a = wrng.uniform(0, 2 * np.pi)
                u = np.array([np.cos(a), np.sin(a)])
                i0 = int(np.argmax((X - c) @ u))
                pred = X[i0] + 3.0 * (X[i0] - c) / (np.linalg.norm(X[i0] - c) + 1e-9)
                go = (np.linalg.norm(X - X[i0], axis=1) <= R_WAVE) & (state == REST)
                state[go], clock[go] = EXC, 0
                attacks += 1
            if mode == "imposed" and (state == EXC).any():
                pairs = cKDTree(X).query_pairs(R_WAVE, output_type="ndarray")
                if len(pairs):
                    i, j = pairs.T
                    exc = state == EXC
                    near = np.zeros(N, bool)
                    near[i[exc[j]]] = True; near[j[exc[i]]] = True
                    go = near & (state == REST) & (wrng.random(N) < Q)
                    state[go], clock[go] = EXC, -1
            clock += 1
            done = (state == EXC) & (clock >= TAU_E)
            state[done], clock[done] = REF, 0
            back = (state == REF) & (clock >= TAU_R)
            state[back], clock[back] = REST, 0
        exc = state == EXC
        if mode == "imposed":
            w += RELAX * (np.where(exc, W_EXC, W_REST) - w); dl += RELAX * (np.where(exc, D_EXC, D_REST) - dl)
        head = np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())
        if t >= WARM and t % EVERY == 0:
            dev = np.abs(np.angle(np.exp(1j * (theta - head))))
            turn = 0.0 if head_prev is None else abs(np.angle(np.exp(1j * (head - head_prev)))) / (EVERY * DT)
            rows.append(dict(mode=mode, seed=seed, t=t, reacting=float(exc.mean()), attacks=attacks,
                             dissent=float((dev > np.pi / 6).mean()), turn_rate=float(np.degrees(turn)),
                             **shape(X, theta, kind, p)))
            head_prev = head
        if film and t >= T - MOVIE_STEPS:
            fr.append((t, X.astype(np.float32), theta.astype(np.float32), exc.copy(),
                       None if pred is None else pred.copy()))
        if t < T:
            p["w"] = w[:, None]
            S = np.where(same, 1.0, fm.similarity_s(p["G"], dl)[:, None])
            esc_dir = esc_amp = None
            if exc.any() and pred is not None:
                if DODGE == "common":
                    rel = pred - X.mean(0)
                    side = np.sign(np.cos(head) * rel[1] - np.sin(head) * rel[0]) or 1.0   # +1: predator on the left
                    esc_dir = np.full(N, head - side * np.pi / 2)
                else:
                    esc_dir = np.arctan2(X[:, 1] - pred[1], X[:, 0] - pred[0])
                esc_amp = np.where(exc, A_ESC, 0.0)
            X, theta, mem = step_ext(X, theta, mem, kind, S, rng, grid, p, esc_dir, esc_amp)
    return rows, (mode, seed, kind, fr)


def movie(mode, kind, fr, path, half=35.0):
    fig, ax = plt.subplots(figsize=(5.5, 5.8))
    cols = np.array(fm.KIND_COLORS)[kind]
    q, title, c0 = None, ax.set_title("", fontsize=9), None
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ring = ax.scatter([], [], s=28, facecolors="none", edgecolors=fm.INK, linewidths=0.8)
    hawk = ax.scatter([], [], s=60, marker="x", color="#d12f2f")

    def draw(i):
        nonlocal q, c0
        t, X, th, ex, pred = fr[i]
        lab = fm.groups(X, fm.BASE["R"])
        c = X[lab == np.bincount(lab).argmax()].mean(0)
        c0 = c if c0 is None else 0.8 * c0 + 0.2 * c
        Y = X - c0
        if q is not None:
            q.remove()
        q = ax.quiver(Y[:, 0], Y[:, 1], np.cos(th), np.sin(th), color=cols, angles="xy", scale_units="xy",
                      scale=0.6, width=0.005, headwidth=3, headlength=3)
        ring.set_offsets(Y[ex] if ex.any() else np.empty((0, 2)))
        hawk.set_offsets((pred - c0)[None] if (pred is not None and ex.any()) else np.empty((0, 2)))
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        title.set_text(f"{mode} ({DODGE} dodge, cue² = {A_ESC**2:g}): μ = {MU:g}, {V0 * 10:g} bl/s, attack every ~{TRIGGER * DT:g} s   time {(t - fr[0][0]) * DT:5.1f} s\n"
                       "ringed: reacting (escape cue on); x: predator")
        return [q, ring, hawk, title]

    anim = animation.FuncAnimation(fig, draw, frames=len(fr), blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=20, bitrate=2400), dpi=110)      # 2x real time
    plt.close(fig)


def traces_plot(ts, path):
    keys = [("reacting", "share reacting"), ("dissent", "share >30° off mean heading"), ("turn_rate", "flock turn rate (°/s)"),
            ("density", "density of main cluster"), ("nbr_like", None), ("M", "mixing M"), ("clusters_5", "contact clusters ≥5"),
            ("Phi", "polarization Φ")]
    keys = [k for k in keys if k[1]]
    fig, axes = plt.subplots(len(keys), len(MODES), figsize=(3.8 * len(MODES), 1.45 * len(keys)), sharex=True, sharey="row", squeeze=False)
    for j, m in enumerate(MODES):
        d = ts[(ts["mode"] == m) & (ts["seed"] == ts["seed"].min())]
        d = d[d["t"] >= d["t"].max() - 600]
        tt = (d["t"] - d["t"].iloc[0]) * DT
        for i, (k, lab) in enumerate(keys):
            ax = axes[i, j]
            ax.plot(tt, d[k], lw=0.9, color="#4a3aa7")
            if i == 0:
                ax.set_title(m, fontsize=10)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
            ax.tick_params(labelsize=7); fm.style(ax)
        axes[-1, j].set_xlabel("s")
    fig.suptitle(f"Last 60 s, seed {ts['seed'].min()}; μ = {MU:g}, {V0 * 10:g} body lengths/s", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(TURN_OUT, exist_ok=True)
    t0 = time.time()
    jobs = [(m, s) for m in MODES for s in SEEDS]
    outs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one)(m, s, s == SEEDS[0]) for m, s in jobs)
    ts = pd.DataFrame([r for rows, _ in outs for r in rows])
    ts.to_csv(os.path.join(TURN_OUT, "turns_timeseries.csv"), index=False)
    cols = ["reacting", "dissent", "turn_rate", "density", "M", "clusters_5", "C", "Phi", "aspect", "nn"]
    per = ts.groupby(["mode", "seed"])[cols].agg(["mean", "std", "max"])
    per.columns = [f"{a}_{b}" for a, b in per.columns]
    per["attacks"] = ts.groupby(["mode", "seed"])["attacks"].max()
    per = per.reset_index()
    per.to_csv(os.path.join(TURN_OUT, "turns_per_run.csv"), index=False)
    summ = per.drop(columns="seed").groupby("mode").mean().round(3)
    summ.to_csv(os.path.join(TURN_OUT, "turns_summary.csv"))
    traces_plot(ts, os.path.join(TURN_OUT, "turns_traces.png"))
    films = [f for _, f in outs if f[3]]
    Parallel(n_jobs=N_JOBS)(delayed(movie)(m, kind, fr, os.path.join(TURN_OUT, f"turns_{m}.mp4")) for m, s, kind, fr in films)
    pd.set_option("display.width", 250)
    print(summ[["reacting_max", "dissent_mean", "dissent_max", "turn_rate_mean", "turn_rate_max", "density_std",
                "M_mean", "M_max", "clusters_5_mean", "C_mean", "Phi_mean", "attacks"]])
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {TURN_OUT}")
