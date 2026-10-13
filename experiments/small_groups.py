"""Small groups (10-50 birds) in 2D and 3D: neighbour switching and how steady the spacing is.

Parakeet-like groups keep a steady density but swap neighbours often; this measures both in the full model.
Runs: dim 2 and 3, N in SIZES, delta0 in DELTAS (w = 0.8), and in 3D gravity gamma in GAMMAS (0 = none).
Every second (10 steps) we record:
  switch     share of each bird's K nearest neighbours that changed since the previous second (per second)
  nn, nn3    median nearest-neighbour distance, mean distance to the K nearest
  rg         radius of gyration of the whole group;  Phi, M (over the 5 nearest), contact clusters, largest share
  thick      3D: sd of heights / sd of horizontal positions (flatness; 1 = round)
  pitch      3D: mean |angle of the heading to the horizontal|, degrees

    python experiments/small_groups.py
    QUICK=1 python experiments/small_groups.py

Results go to SG_OUT (default output/small_groups).  One step is 0.1 s.
"""
import os, sys, time
os.environ.setdefault("OUT", os.path.join(os.path.dirname(__file__), "..", "output", "model_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import flock_model as fm
import flock3d as f3

DT, W, K = 0.1, 0.8, 3
SIZES = [int(x) for x in os.environ.get("SIZES", "10,20,30,50").split(",")]
DELTAS = [float(x) for x in os.environ.get("DELTAS", "0.3,0.5,0.65").split(",")]
GAMMAS = [float(x) for x in os.environ.get("GAMMAS", "0,10").split(",")]
SEEDS = list(range(8000, 8005))
T, EVERY, BURN = int(os.environ.get("T", 3000)), 10, 1000         # measures averaged after BURN steps
FILM = [(3, 30, 0.3, 0.0), (3, 30, 0.3, 10.0), (3, 30, 0.65, 10.0)]
if os.environ.get("QUICK", "0") == "1":
    SIZES, DELTAS, GAMMAS, SEEDS, T, BURN, FILM = [10, 30], [0.3], [0, 10], [8000], 200, 50, [(3, 10, 0.3, 10.0)]
SG_OUT = os.environ.get("SG_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "small_groups"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))


def measures(X, U, kind, p, prev):
    k = min(K, len(X) - 1)
    dist, nb = cKDTree(X).query(X, k + 1)
    nb = np.sort(nb[:, 1:], 1)
    switch = np.nan if prev is None else float(np.mean([len(set(a) - set(b)) / k for a, b in zip(nb, prev)])) / (EVERY * DT)
    lab = fm.groups(X, fm.CONTACT * p["r0"])
    sizes = np.bincount(lab)
    Y = X - X.mean(0)
    out = dict(switch=switch, nn=float(np.median(dist[:, 1])), nn3=float(dist[:, 1:].mean()),
               rg=float(np.sqrt((Y ** 2).sum(1).mean())), Phi=float(np.linalg.norm(U.mean(0))),
               M=float(fm.mixing_entropy(X, kind, p["G"], m=min(5, len(X) - 1))), clusters=int(len(sizes)),
               largest=float(sizes.max() / len(X)), C=float(fm.cohesion(X, p["R"])))
    if X.shape[1] == 3:
        out["thick"] = float(Y[:, 2].std() / max(np.sqrt(0.5 * (Y[:, 0].var() + Y[:, 1].var())), 1e-9))
        out["pitch"] = float(np.degrees(np.abs(np.arcsin(np.clip(U[:, 2], -1, 1)))).mean())
    return out, nb


def one(dim, N, delta, gamma, seed, film):
    p = dict(f3.BASE3, N=N, w=W, delta0=delta, gamma=gamma)
    if dim == 2:
        dirs = np.linspace(0, 2 * np.pi, p["n_dir"], endpoint=False)
        grid = (dirs, np.cos(dirs), np.sin(dirs))
        X, theta, kind, rng = fm.start(p, seed)
        U = f3.angles_to_units(theta)
    else:
        grid = f3.sphere_grid(p["n_dir3"])
        X, U, kind, rng = f3.start3d(p, seed)
    S = np.where(kind[:, None] == kind[None, :], 1.0, fm.similarity_s(p["G"], delta))
    mem, rows, prev, fr = None, [], None, []
    for t in range(T + 1):
        if t % EVERY == 0:
            m, prev = measures(X, U, kind, p, prev)
            rows.append(dict(dim=dim, N=N, delta0=delta, gamma=gamma, seed=seed, t=t, **m))
        if film and t % 2 == 0 and BURN <= t < BURN + 600:            # 60 s of flight, every 0.2 s
            fr.append((t, X.astype(np.float32), U.astype(np.float32)))
        if t < T:
            if dim == 2:
                X, theta, mem = fm.step(X, theta, mem, kind, S, fm.CASES["full"], rng, grid, p)
                U = f3.angles_to_units(theta)
            else:
                X, U, mem = f3.step3d(X, U, mem, kind, S, rng, grid, p)
    return rows, ((dim, N, delta, gamma), kind, fr)


def movie(key, kind, fr, path, half=8.0, label=None):
    dim, N, delta, gamma = key
    fig = plt.figure(figsize=(9.6, 4.8))
    a3 = fig.add_axes([0.0, 0.0, 0.5, 0.92], projection="3d")
    side = fig.add_axes([0.56, 0.1, 0.4, 0.75])
    kc = np.array(fm.KIND_COLORS)[kind]
    title = fig.text(0.02, 0.95, "", fontsize=9)
    c0 = [None]

    def draw(i):
        t, X, U = fr[i]
        c = X.mean(0)
        c0[0] = c if c0[0] is None else 0.7 * c0[0] + 0.3 * c
        Y = X - c0[0]
        a3.cla()
        a3.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=kc, s=18, linewidths=0)
        a3.quiver(Y[:, 0], Y[:, 1], Y[:, 2], U[:, 0], U[:, 1], U[:, 2], length=1.2, color=kc, lw=0.8, arrow_length_ratio=0)
        for f in (a3.set_xlim, a3.set_ylim, a3.set_zlim):
            f(-half, half)
        a3.set_box_aspect((1, 1, 1)); a3.set_xticklabels([]); a3.set_yticklabels([]); a3.set_zticklabels([])
        a3.view_init(elev=20, azim=-60 + 0.3 * i)
        # side view: horizontal distance along the mean horizontal heading vs height
        h = U[:, :2].mean(0); h = h / (np.linalg.norm(h) + 1e-12)
        side.cla()
        side.scatter(Y[:, :2] @ h, Y[:, 2], c=kc, s=18, linewidths=0)
        side.set_xlim(-half, half); side.set_ylim(-half, half); side.set_aspect("equal")
        side.set_xlabel("along travel"); side.set_ylabel("height"); side.set_title("side view", fontsize=8, loc="left")
        fm.style(side)
        title.set_text(f"3D  N = {N}  δ = {delta:g}  " + (label or f"gravity γ = {gamma:g}") + f"   t = {t * DT:5.0f} s")
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(fr), blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=25, bitrate=2000), dpi=100)
    plt.close(fig)


def plot(summ, path):
    keys = [("switch", "neighbour switching (share of 3 nearest per s)"), ("nn3", "distance to 3 nearest"),
            ("rg_cv", "spacing fluctuation (CV of group radius)"), ("Phi", "polarization Φ"),
            ("M", "mixing M"), ("largest", "share in largest contact cluster"), ("thick", "3D flatness: height sd / horizontal sd"),
            ("pitch", "3D mean |pitch| (deg)")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    styles = {(2, 0.0): ("#8a8a86", "o", "2D"), (3, 0.0): ("#4a3aa7", "s", "3D"), (3, 10.0): ("#eb6834", "^", "3D + gravity")}
    lines = {0.3: "-", 0.5: "--", 0.65: ":"}
    for ax, (k, lab) in zip(axes.flat, keys):
        for (dim, g), (c, mk, name) in styles.items():
            for dl, ls in lines.items():
                d = summ[(summ.dim == dim) & (summ.gamma == g) & (summ.delta0 == dl)]
                if len(d) and k in d and d[k].notna().any():
                    ax.plot(d.N, d[k], ls=ls, marker=mk, color=c, ms=4, lw=1.1, label=f"{name}, δ={dl:g}")
        ax.set_title(lab, fontsize=8, loc="left"); ax.set_xlabel("group size N", fontsize=8); fm.style(ax)
    axes.flat[0].legend(fontsize=6, ncol=1)
    fig.suptitle(f"Small groups, w = {W}: means over {len(SEEDS)} seeds, {BURN * DT:.0f}–{T * DT:.0f} s", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(SG_OUT, exist_ok=True)
    t0 = time.time()
    jobs = [(2, N, d, 0.0, s) for N in SIZES for d in DELTAS for s in SEEDS]
    jobs += [(3, N, d, g, s) for N in SIZES for d in DELTAS for g in GAMMAS for s in SEEDS]
    jobs.sort(key=lambda j: -(j[0] * j[1]))
    outs = Parallel(n_jobs=N_JOBS, verbose=2)(delayed(one)(*j, (j[0], j[1], j[2], j[3]) in FILM and j[4] == SEEDS[0]) for j in jobs)
    ts = pd.DataFrame([r for rows, _ in outs for r in rows])
    ts.to_csv(os.path.join(SG_OUT, "small_groups_timeseries.csv"), index=False)
    late = ts[ts.t >= BURN]
    keys = ["dim", "N", "delta0", "gamma", "seed"]
    per = late.groupby(keys).agg(switch=("switch", "mean"), nn=("nn", "mean"), nn3=("nn3", "mean"), rg=("rg", "mean"),
                                 rg_sd=("rg", "std"), Phi=("Phi", "mean"), M=("M", "mean"), largest=("largest", "mean"),
                                 clusters=("clusters", "mean"), C=("C", "mean"),
                                 thick=("thick", "mean"), pitch=("pitch", "mean")).reset_index()
    per["rg_cv"] = per.rg_sd / per.rg
    per.round(4).to_csv(os.path.join(SG_OUT, "small_groups_per_run.csv"), index=False)
    summ = per.drop(columns="seed").groupby(keys[:-1]).mean().reset_index().round(3)
    summ.to_csv(os.path.join(SG_OUT, "small_groups_summary.csv"), index=False)
    plot(summ, os.path.join(SG_OUT, "small_groups.png"))
    films = [o for _, o in outs if o[2]]
    Parallel(n_jobs=N_JOBS)(delayed(movie)(k, kind, fr, os.path.join(SG_OUT, f"small_d{k[0]}_N{k[1]}_dl{k[2]:g}_g{k[3]:g}.mp4"))
                            for k, kind, fr in films)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summ)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {SG_OUT}")
