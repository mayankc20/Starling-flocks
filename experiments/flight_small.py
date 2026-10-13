"""Small 3D groups (10-50) with and without flight constraints (model/flight3d.py).

Conditions, all at cruise speed V0 body lengths per step (default 2, i.e. 5 m/s for a 0.25 m bird at 0.1 s per step):
  free      3D model, no gravity, constant speed
  gravity   3D model with the gravity weight gamma on the perceptual states, constant speed
  flight    gravity weight + banked turns + speed that changes with climb (FLIGHT defaults)
  quad      gravity weight + speed with climb, turns without banking (a quadrotor), same pitch cap
Start: mixed ball, headings within about 20 deg of +x and nearly level (a group already in flight).
Measures (every second, after BURN): those of small_groups.py plus turn rate (deg/s), |bank| (deg), speed spread.

    python experiments/flight_small.py
    QUICK=1 python experiments/flight_small.py

Results go to FS_OUT (default output/flight_small).
"""
import os, sys, time
os.environ.setdefault("OUT", os.path.join(os.path.dirname(__file__), "..", "output", "model_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import flock_model as fm
import flock3d as f3
import flight3d as fl
import small_groups as sg

DT, W, V0, GAMMA = 0.1, 0.8, float(os.environ.get("V0", 2.0)), 10.0
R_SIGHT = float(os.environ.get("R_SIGHT", 100.0))   # body lengths; at 25 (the base value) groups at this speed break up
SIZES = [int(x) for x in os.environ.get("SIZES", "10,20,30,50").split(",")]
DELTAS = [float(x) for x in os.environ.get("DELTAS", "0.3,0.65").split(",")]
CONDS = os.environ.get("CONDS", "free,gravity,flight,quad").split(",")
SEEDS = list(range(8100, 8105))
T, EVERY, BURN = int(os.environ.get("T", 3000)), sg.EVERY, 1000
FILM = [("flight", 30, 0.3), ("free", 30, 0.3), ("flight", 30, 0.65)]
if os.environ.get("QUICK", "0") == "1":
    SIZES, DELTAS, SEEDS, T, BURN, FILM = [10, 30], [0.3], [8100], 200, 50, [("flight", 10, 0.3)]
    sg.BURN = BURN
FS_OUT = os.environ.get("FS_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "flight_small"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))


def one(cond, N, delta, seed, film):
    p = dict(f3.BASE3, **fl.FLIGHT, N=N, w=W, delta0=delta, v0=V0, R=R_SIGHT, gamma=0.0 if cond == "free" else GAMMA)
    if cond == "quad":
        p.update(bank_max_deg=89.9, roll_rate_deg_s=1e6)
    grid = f3.sphere_grid(p["n_dir3"])
    X, U, kind, rng = f3.start3d(p, seed)
    U = np.array([1.0, 0.0, 0.0]) + 0.3 * U                                 # start mixed but flying roughly one way, level
    U[:, 2] *= 0.3; U /= np.linalg.norm(U, axis=1, keepdims=True)
    S = np.where(kind[:, None] == kind[None, :], 1.0, fm.similarity_s(p["G"], delta))
    v, bank, yaw = np.full(N, V0), np.zeros(N), np.zeros(N)
    mem, rows, prev, fr = None, [], None, []
    for t in range(T + 1):
        if t % EVERY == 0:
            m, prev = sg.measures(X, U, kind, p, prev)
            rows.append(dict(cond=cond, N=N, delta0=delta, seed=seed, t=t, **m,
                             turn=float(np.degrees(np.abs(yaw)).mean() / DT), bank=float(np.degrees(np.abs(bank)).mean()),
                             v_cv=float(v.std() / v.mean()), v_mean=float(v.mean())))
        if film and t % 2 == 0 and BURN <= t < BURN + 600:
            fr.append((t, X.astype(np.float32), U.astype(np.float32)))
        if t < T:
            if cond in ("flight", "quad"):
                X, U, v, bank, mem, yaw = fl.step_flight(X, U, v, bank, mem, kind, S, rng, grid, p)
            else:
                U0 = U
                X, U, mem = f3.step3d(X, U, mem, kind, S, rng, grid, p)
                side, _ = fl.frame(U0)
                yaw = np.arctan2(np.einsum("ij,ij->i", U, side), np.einsum("ij,ij->i", U, U0))
    return rows, ((cond, N, delta), kind, fr)


def plot(summ, path):
    keys = [("switch", "neighbour switching (share of 3 nearest per s)"), ("nn3", "distance to 3 nearest (body lengths)"),
            ("rg_cv", "spacing fluctuation (CV of group radius)"), ("Phi", "polarization Φ"),
            ("thick", "flatness: height sd / horizontal sd"), ("turn", "mean turn rate (deg/s)"),
            ("M", "mixing M (5 nearest)"), ("largest", "share in largest contact cluster")]
    cols = {"free": "#8a8a86", "gravity": "#2a78d6", "flight": "#eb6834", "quad": "#1baf7a"}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, (k, lab) in zip(axes.flat, keys):
        for c in CONDS:
            for dl, ls in zip(DELTAS, ["-", ":"]):
                d = summ[(summ.cond == c) & (summ.delta0 == dl)]
                ax.plot(d.N, d[k], ls=ls, marker="o", ms=4, color=cols[c], lw=1.1, label=f"{c}, δ={dl:g}")
        ax.set_title(lab, fontsize=8, loc="left"); ax.set_xlabel("group size N", fontsize=8); fm.style(ax)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle(f"3D small groups, w = {W}, cruise {V0:g} body lengths per step; means over {len(SEEDS)} seeds, "
                 f"{BURN * DT:.0f}–{T * DT:.0f} s, sight {R_SIGHT:g} body lengths", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FS_OUT, exist_ok=True)
    t0 = time.time()
    jobs = [(c, N, d, s) for c in CONDS for N in SIZES for d in DELTAS for s in SEEDS]
    jobs.sort(key=lambda j: -j[1])
    outs = Parallel(n_jobs=N_JOBS, verbose=2)(delayed(one)(*j, (j[0], j[1], j[2]) in FILM and j[3] == SEEDS[0]) for j in jobs)
    ts = pd.DataFrame([r for rows, _ in outs for r in rows])
    ts.to_csv(os.path.join(FS_OUT, "flight_timeseries.csv"), index=False)
    late = ts[ts.t >= BURN]
    keys = ["cond", "N", "delta0", "seed"]
    agg = {k: (k, "mean") for k in ["switch", "nn", "nn3", "rg", "Phi", "M", "largest", "clusters", "thick", "pitch",
                                    "turn", "bank", "v_cv", "v_mean"]}
    per = late.groupby(keys).agg(**agg, rg_sd=("rg", "std"), turn_p95=("turn", lambda x: np.percentile(x, 95))).reset_index()
    per["rg_cv"] = per.rg_sd / per.rg
    per.round(4).to_csv(os.path.join(FS_OUT, "flight_per_run.csv"), index=False)
    summ = per.drop(columns="seed").groupby(keys[:-1]).mean().reset_index().round(3)
    summ.to_csv(os.path.join(FS_OUT, "flight_summary.csv"), index=False)
    plot(summ, os.path.join(FS_OUT, "flight_small.png"))
    films = [o for _, o in outs if o[2]]
    Parallel(n_jobs=N_JOBS)(delayed(sg.movie)((3, k[1], k[2], GAMMA if k[0] != "free" else 0.0), kind, fr,
                                              os.path.join(FS_OUT, f"{k[0]}_N{k[1]}_dl{k[2]:g}.mp4"), half=16.0,
                                              label=k[0])
                            for k, kind, fr in films)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summ)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {FS_OUT}")
