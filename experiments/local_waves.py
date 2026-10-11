"""Local waves of w and delta0: each agent carries its own w and delta0, and a change started by a few agents
spreads to their neighbours, as the turning waves of starling flocks do.

Time scale: one step = 0.1 s (as in experiments/fast_cycles.py).

Each agent is in one of three states (an excitable medium):
  rest        w = W_REST, delta0 = D_REST (the paper's values: tight, sorted subgroups)
  excited     w = W_EXC,  delta0 = D_EXC  (loose, attends to other kinds); lasts TAU_E steps
  refractory  back to rest values, cannot be excited again for TAU_R steps
A resting agent is excited, with probability Q per step, if any agent within R_WAVE body lengths is excited.
Waves are started at random: on average every TRIGGER steps an agent on the edge of the flock (the farthest from the
centre in a random direction) and its neighbours within R_WAVE are excited, as by a predator's approach.
An agent's actual w and delta0 relax toward its state's values at rate RELAX per step, so changes are smooth.

The model is unchanged: step() accepts w as a per-agent column (N, 1), and a per-agent delta0 enters through the
rows of the similarity matrix S (row i is what agent i uses to choose its partners).

    python experiments/local_waves.py
    QUICK=1 python experiments/local_waves.py

Settings: AGENTS = "slow" (v0 = 5 body lengths/s, mu = 0.95) and "fast" (10 body lengths/s, mu = 0.8);
trigger intervals 3 s and 10 s; 2 seeds. Results go to WAVE_OUT (default output/local_waves).
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
from w_sweep import shape

DT = 0.1
W_REST, D_REST, W_EXC, D_EXC = 0.8, 0.3, 0.96, 0.55
TAU_E, TAU_R, Q, R_WAVE, RELAX = 10, 30, 0.6, 3.0, 0.3
AGENTS = {"slow": dict(v0=0.5, mu=0.95), "fast": dict(v0=1.0, mu=0.8)}
TRIGGERS = [30, 100]                       # mean steps between waves: 3 s, 10 s
N, WARM, RUN, EVERY = 240, 3000, 1500, 2
SEEDS = [8000, 8001]
MOVIE_STEPS = 600                          # last 60 s
QUICK = os.environ.get("QUICK", "0") == "1"
if QUICK:
    N, WARM, RUN, SEEDS, MOVIE_STEPS, TRIGGERS = 60, 100, 200, [8000], 100, [30]
WAVE_OUT = os.environ.get("WAVE_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "local_waves"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))
REST, EXC, REF = 0, 1, 2


def one(agents, trigger, seed, film):
    p = fm.job_params([("N", N), *AGENTS[agents].items()])
    case = fm.CASES["full"]
    dirs = np.linspace(0, 2 * np.pi, p["n_dir"], endpoint=False)
    grid = (dirs, np.cos(dirs), np.sin(dirs))
    X, theta, kind, rng = fm.start(p, seed)
    wrng = np.random.default_rng(seed + 99)
    same = kind[:, None] == kind[None, :]
    state, clock = np.zeros(N, int), np.zeros(N, int)
    w, dl = np.full(N, W_REST), np.full(N, D_REST)
    T = WARM + RUN
    mem, rows, fr, waves = None, [], [], 0
    for t in range(T + 1):
        if t >= WARM:
            # start a wave at the edge
            if wrng.random() < 1 / trigger:
                a = wrng.uniform(0, 2 * np.pi)
                c = X.mean(0)
                i0 = int(np.argmax((X - c) @ np.array([np.cos(a), np.sin(a)])))
                hit = np.linalg.norm(X - X[i0], axis=1) <= R_WAVE
                go = hit & (state == REST)
                state[go], clock[go] = EXC, 0
                waves += 1
            # spread
            exc = state == EXC
            if exc.any():
                pairs = cKDTree(X).query_pairs(R_WAVE, output_type="ndarray")
                if len(pairs):
                    i, j = pairs.T
                    near = np.zeros(N, bool)
                    near[i[exc[j]]] = True; near[j[exc[i]]] = True
                    go = near & (state == REST) & (wrng.random(N) < Q)
                    state[go], clock[go] = EXC, -1                     # -1: counts from the next step
            clock += 1
            done = (state == EXC) & (clock >= TAU_E)
            state[done], clock[done] = REF, 0
            back = (state == REF) & (clock >= TAU_R)
            state[back], clock[back] = REST, 0
        tw = np.where(state == EXC, W_EXC, W_REST)
        td = np.where(state == EXC, D_EXC, D_REST)
        w += RELAX * (tw - w); dl += RELAX * (td - dl)
        if t >= WARM and t % EVERY == 0:
            cnt = np.array([len(x) for x in cKDTree(X).query_ball_point(X, 3.0)]) - 1
            rows.append(dict(agents=agents, trigger_s=trigger * DT, seed=seed, t=t, excited=float((state == EXC).mean()),
                             w_mean=float(w.mean()), delta_mean=float(dl.mean()), waves=waves,
                             nbr_mean=float(cnt.mean()), nbr_cv=float(cnt.std() / max(cnt.mean(), 1e-9)),
                             **shape(X, theta, kind, p)))
        if film and t >= T - MOVIE_STEPS:
            fr.append((t, X.astype(np.float32), theta.astype(np.float32), (state == EXC).copy()))
        if t < T:
            p["w"] = w[:, None]
            s = fm.similarity_s(p["G"], dl)
            S = np.where(same, 1.0, s[:, None])
            X, theta, mem = fm.step(X, theta, mem, kind, S, case, rng, grid, p)
    return rows, (agents, trigger, seed, kind, fr)


def movie(agents, trigger, kind, fr, path, half=35.0):
    fig, ax = plt.subplots(figsize=(5.5, 5.8))
    cols = np.array(fm.KIND_COLORS)[kind]
    q, title, c0 = None, ax.set_title("", fontsize=9), None
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ring = ax.scatter([], [], s=28, facecolors="none", edgecolors=fm.INK, linewidths=0.8)

    def draw(i):
        nonlocal q, c0
        t, X, th, ex = fr[i]
        lab = fm.groups(X, fm.BASE["R"])
        c = X[lab == np.bincount(lab).argmax()].mean(0)
        c0 = c if c0 is None else 0.8 * c0 + 0.2 * c
        Y = X - c0
        if q is not None:
            q.remove()
        q = ax.quiver(Y[:, 0], Y[:, 1], np.cos(th), np.sin(th), color=cols, angles="xy", scale_units="xy",
                      scale=0.6, width=0.005, headwidth=3, headlength=3)
        ring.set_offsets(Y[ex] if ex.any() else np.empty((0, 2)))
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        title.set_text(f"{agents} agents, a wave every ~{trigger * DT:g} s   time {(t - fr[0][0]) * DT:5.1f} s\n"
                       f"ringed: excited (w = {W_EXC}, δ = {D_EXC})")
        return [q, ring, title]

    anim = animation.FuncAnimation(fig, draw, frames=len(fr), blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=20, bitrate=2400), dpi=110)      # 2x real time
    plt.close(fig)


def traces_plot(ts, path):
    combos = list(ts.groupby(["agents", "trigger_s"]).groups)
    keys = [("excited", "share excited"), ("density", "density of main cluster"), ("nbr_cv", "spread of local density (CV)"),
            ("M", "mixing M"), ("clusters_5", "contact clusters ≥5"), ("Phi", "polarization Φ")]
    fig, axes = plt.subplots(len(keys), len(combos), figsize=(3.6 * len(combos), 1.5 * len(keys)), sharex=True, squeeze=False)
    for j, cmb in enumerate(combos):
        d = ts[(ts["agents"] == cmb[0]) & (ts["trigger_s"] == cmb[1]) & (ts["seed"] == ts["seed"].min())]
        d = d[d["t"] >= d["t"].max() - 600]                       # last 60 s
        tt = (d["t"] - d["t"].iloc[0]) * DT
        for i, (k, lab) in enumerate(keys):
            ax = axes[i, j]
            ax.plot(tt, d[k], lw=0.9, color="#4a3aa7")
            if i == 0:
                ax.set_title(f"{cmb[0]} agents, wave every ~{cmb[1]:g} s", fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
            ax.tick_params(labelsize=7); fm.style(ax)
        axes[-1, j].set_xlabel("s")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(WAVE_OUT, exist_ok=True)
    t0 = time.time()
    jobs = [(a, tr, s) for a in AGENTS for tr in TRIGGERS for s in SEEDS]
    outs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one)(a, tr, s, s == SEEDS[0]) for a, tr, s in jobs)
    ts = pd.DataFrame([r for rows, _ in outs for r in rows])
    ts.to_csv(os.path.join(WAVE_OUT, "waves_timeseries.csv"), index=False)
    cols = ["excited", "density", "nbr_cv", "M", "clusters_5", "C", "Phi", "aspect"]
    per = ts.groupby(["agents", "trigger_s", "seed"])[cols].agg(["mean", "std", "max"])
    per.columns = [f"{a}_{b}" for a, b in per.columns]
    per["waves"] = ts.groupby(["agents", "trigger_s", "seed"])["waves"].max()
    per = per.reset_index()
    per.to_csv(os.path.join(WAVE_OUT, "waves_per_run.csv"), index=False)
    summ = per.drop(columns="seed").groupby(["agents", "trigger_s"]).mean().round(3)
    summ.to_csv(os.path.join(WAVE_OUT, "waves_summary.csv"))
    traces_plot(ts, os.path.join(WAVE_OUT, "waves_traces.png"))
    films = [f for _, f in outs if f[4]]
    Parallel(n_jobs=N_JOBS)(delayed(movie)(a, tr, kind, fr, os.path.join(WAVE_OUT, f"waves_{a}_{tr * DT:g}s.mp4"))
                            for a, tr, s, kind, fr in films)
    pd.set_option("display.width", 250)
    print(summ[["excited_mean", "excited_max", "density_std", "nbr_cv_mean", "M_mean", "M_max", "clusters_5_mean", "C_mean", "Phi_mean", "waves"]])
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {WAVE_OUT}")
