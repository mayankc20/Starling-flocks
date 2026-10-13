"""The full model in 3D against 2D, at the regimes found in 2D (w = 0.8, delta0 from 0.3 to 0.65).

Runs (mixed, disordered start; same N, speed, sight range, partners and noise in both):
  d{2,3}_w{W}_dl{X}      w and delta0 fixed
  d{2,3}_w0.8_cycle{P}   delta0(t) = mid + amp cos(2 pi t / P)  (CYCLE_LO..CYCLE_HI, w = 0.8)
  d3_..._grid4000        the 3D run of delta0 = 0.3 with a finer direction grid (4000 instead of 2000 points)

    python experiments/flock3d_compare.py                 # all runs, figures and videos
    QUICK=1 python experiments/flock3d_compare.py         # tiny check

Results go to F3_OUT (default output/flock3d).  One step is 0.1 s.
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
from joblib import Parallel, delayed
import flock_model as fm
import flock3d as f3

DT = 0.1
DELTAS = [float(x) for x in os.environ.get("DELTAS", "0.3,0.4,0.5,0.6,0.65").split(",") if x]
W_EXTRA = [float(x) for x in os.environ.get("W_EXTRA", "0.6,0.9,1.0").split(",") if x]   # at delta0 = 0.3
CYCLE_LO, CYCLE_HI = float(os.environ.get("CYCLE_LO", 0.4)), float(os.environ.get("CYCLE_HI", 0.5))
PERIOD = int(os.environ.get("PERIOD", 2000))
N, T, EVERY, FRAME_EVERY = 240, int(os.environ.get("T", 5000)), 20, 10
T_CYCLE = int(os.environ.get("T_CYCLE", 8000))
SEEDS = [7000, 7001, 7002]
FILM = {0.3, 0.5, 0.65}                       # delta0 of the 3D runs filmed (first seed); the cycle run is filmed too
STAGES = os.environ.get("STAGES", "const,w,cycle,grid").split(",")
if os.environ.get("QUICK", "0") == "1":
    DELTAS, W_EXTRA, N, T, T_CYCLE, EVERY, SEEDS, PERIOD, FILM = [0.3, 0.6], [1.0], 60, 200, 200, 10, [7000], 100, {0.3}
F3_OUT = os.environ.get("F3_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "flock3d"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))


def delta_of(spec, t):
    if spec[0] == "const":
        return spec[1]
    mid, amp = 0.5 * (CYCLE_LO + CYCLE_HI), 0.5 * (CYCLE_HI - CYCLE_LO)
    return mid + amp * np.cos(2 * np.pi * t / spec[1])


def one(label, dim, w, spec, seed, steps, film, n_dir3=None):
    p = dict(f3.BASE3, N=N, w=w)
    if n_dir3:
        p["n_dir3"] = n_dir3
    same = None
    if dim == 2:
        dirs = np.linspace(0, 2 * np.pi, p["n_dir"], endpoint=False)
        grid = (dirs, np.cos(dirs), np.sin(dirs))
        X, theta, kind, rng = fm.start(p, seed)
        U = f3.angles_to_units(theta)
    else:
        grid = f3.sphere_grid(p["n_dir3"])
        X, U, kind, rng = f3.start3d(p, seed)
    same = kind[:, None] == kind[None, :]
    mem, rows, fr = None, [], []
    for t in range(steps + 1):
        d = delta_of(spec, t)
        if t % EVERY == 0:
            rows.append(dict(run=label, dim=dim, w=w, seed=seed, t=t, delta0=d, **f3.shape_nd(X, U, kind, p)))
        if film and t % FRAME_EVERY == 0:
            fr.append((t, X.astype(np.float32), U.astype(np.float32), d))
        if t < steps:
            p["delta0"] = d
            S = np.where(same, 1.0, fm.similarity_s(p["G"], d))
            if dim == 2:
                X, theta, mem = fm.step(X, theta, mem, kind, S, fm.CASES["full"], rng, grid, p)
                U = f3.angles_to_units(theta)
            else:
                X, U, mem = f3.step3d(X, U, mem, kind, S, rng, grid, p)
    return rows, (label, kind, fr, X, U)


def jobs():
    js = []
    for dim in (2, 3):
        if "const" in STAGES:
            js += [(f"d{dim}_w0.8_dl{x:g}", dim, 0.8, ("const", x), s, T, dim == 3 and x in FILM and s == SEEDS[0], None)
                   for x in DELTAS for s in SEEDS]
        if "w" in STAGES:
            js += [(f"d{dim}_w{w:g}_dl0.3", dim, w, ("const", 0.3), s, T, False, None) for w in W_EXTRA for s in SEEDS[:2]]
        if "cycle" in STAGES:
            js += [(f"d{dim}_w0.8_cycle{PERIOD}", dim, 0.8, ("cycle", PERIOD), s, T_CYCLE, dim == 3 and s == SEEDS[0], None)
                   for s in SEEDS[:2]]
    if "grid" in STAGES:
        js += [("d3_w0.8_dl0.3_grid4000", 3, 0.8, ("const", 0.3), SEEDS[0], T, False, 4000)]
    return js


# ---------------------------------------------------------------- figures

KC = np.array(fm.KIND_COLORS)


def centre(X, R=fm.BASE["R"]):
    lab = fm.groups(X, R)
    return X[lab == np.bincount(lab).argmax()].mean(0)


def frame_axes(X, U, kind, ax, half=14.0, c=None):
    """Draw a 3D flock around c (default: centre of its largest group); short lines show the headings."""
    c = centre(X) if c is None else c
    Y = X - c
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=KC[kind], s=7, depthshade=True, linewidths=0)
    ax.quiver(Y[:, 0], Y[:, 1], Y[:, 2], U[:, 0], U[:, 1], U[:, 2], length=1.3, color=KC[kind], lw=0.6,
              arrow_length_ratio=0.0)
    for f in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        f(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(length=0)


def rotate_to_heading(X, U):
    """Rotate so that the mean heading is +x (keeps the views comparable across runs)."""
    h = U.mean(0)
    if np.linalg.norm(h) < 1e-6:
        return X, U
    h = h / np.linalg.norm(h)
    a = np.cross(h, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.array([0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(h, a)
    Rm = np.stack([h, a, b])                                   # rows: new x, y, z
    return X @ Rm.T, U @ Rm.T


def snapshots(finals, ts, path):
    labs = [f"d3_w0.8_dl{x:g}" for x in DELTAS]
    fig = plt.figure(figsize=(3.0 * len(labs), 6.3))
    for j, lab in enumerate(labs):
        f = finals.get(lab)
        if f is None:
            continue
        kind, X, U = f
        if X.shape[1] == 3:
            Xr, Ur = rotate_to_heading(X - centre(X), U)
            for i, (el, az, ttl) in enumerate([(15, -60, "oblique"), (0, 0, "seen from ahead")]):
                ax = fig.add_subplot(2, len(labs), 1 + j + i * len(labs), projection="3d")
                frame_axes(Xr, Ur, kind, ax)
                ax.view_init(elev=el, azim=az)
                r = ts[(ts.run == lab) & (ts.seed == SEEDS[0])].iloc[-1]
                ax.set_title(f"δ = {lab.split('dl')[1]}  ({ttl})\nΦ={r.Phi:.2f}  M={r.M:.2f}  clusters≥5: {r.clusters_5}",
                             fontsize=8)
    fig.suptitle(f"3D, w = 0.8, t = {T * DT:.0f} s, seed {SEEDS[0]}; mean heading along +x (arrow = heading)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def snapshots2d(finals, ts, path):
    labs = [f"d2_w0.8_dl{x:g}" for x in DELTAS]
    fig, axes = plt.subplots(1, len(labs), figsize=(3.0 * len(labs), 3.2))
    for ax, lab in zip(np.atleast_1d(axes), labs):
        kind, X, U = finals[lab]
        Y = X - centre(X)
        ax.quiver(Y[:, 0], Y[:, 1], U[:, 0], U[:, 1], color=KC[kind], angles="xy", scale_units="xy", scale=0.6,
                  width=0.006, headwidth=3, headlength=3)
        ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        r = ts[(ts.run == lab) & (ts.seed == SEEDS[0])].iloc[-1]
        ax.set_title(f"2D δ = {lab.split('dl')[1]}\nΦ={r.Phi:.2f}  M={r.M:.2f}  clusters≥5: {r.clusters_5}", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


METRICS = [("Phi", "polarization Φ"), ("M", "mixing M"), ("clusters_5", "contact clusters ≥5"),
           ("largest", "share in largest cluster"), ("density", "density (per area / volume)"), ("aspect", "longest / thinnest axis"),
           ("kinds_vs_heading", "|cos| kinds' axis vs heading")]


def compare_plot(summ, path):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    sub = summ[summ.run.str.contains("_w0.8_dl") & ~summ.run.str.contains("grid")]
    for ax, (k, lab) in zip(axes.flat, METRICS):
        for dim, c in [(2, "#8a8a86"), (3, "#4a3aa7")]:
            d = sub[sub.dim == dim].groupby("delta")[k].agg(["mean", "std"]).reset_index()
            ax.errorbar(d["delta"], d["mean"], d["std"], color=c, marker="o", ms=4, lw=1.2, capsize=2, label=f"{dim}D")
        ax.set_title(lab, fontsize=9, loc="left"); ax.set_xlabel("δ (w = 0.8)", fontsize=8); fm.style(ax)
    axes.flat[0].legend(fontsize=8)
    ax = axes.flat[-1]
    w = summ[summ.run.str.endswith("_dl0.3") & ~summ.run.str.contains("grid")]
    for dim, c in [(2, "#8a8a86"), (3, "#4a3aa7")]:
        d = w[w.dim == dim].groupby("w")["Phi"].mean()
        ax.plot(d.index, d.values, "o-", color=c, ms=4, label=f"{dim}D Φ")
        d = w[w.dim == dim].groupby("w")["M"].mean()
        ax.plot(d.index, d.values, "s--", color=c, ms=4, label=f"{dim}D M")
    ax.set_title("w at δ = 0.3", fontsize=9, loc="left"); ax.set_xlabel("w", fontsize=8); ax.legend(fontsize=7); fm.style(ax)
    fig.suptitle(f"2D vs 3D, N = {N}, mixed start; mean ± sd over seeds of the second half of each run ({T * DT / 2:.0f}–{T * DT:.0f} s)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def traces_plot(ts, runs, path, keys=("delta0", "Phi", "M", "clusters_5", "density", "aspect")):
    fig, axes = plt.subplots(len(keys), len(runs), figsize=(2.9 * len(runs), 1.4 * len(keys)), sharex=True, sharey="row", squeeze=False)
    names = dict(METRICS, delta0="δ")
    for j, r in enumerate(runs):
        for i, k in enumerate(keys):
            ax = axes[i, j]
            for c, (s, d) in zip(["#4a3aa7", "#eb6834", "#1baf7a"], ts[ts.run == r].groupby("seed")):
                ax.plot(d.t * DT, d[k], lw=0.8, color=c)
            if i == 0:
                ax.set_title(r, fontsize=8)
            if j == 0:
                ax.set_ylabel(names.get(k, k), fontsize=7)
            ax.tick_params(labelsize=7); fm.style(ax)
        axes[-1, j].set_xlabel("s")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def movie(label, kind, fr, ts, path, half=14.0):
    fig = plt.figure(figsize=(9.6, 5.4))
    ax = fig.add_axes([0.0, 0.0, 0.6, 0.95], projection="3d")
    axs = [fig.add_axes([0.66, 0.72 - 0.3 * i, 0.32, 0.2]) for i in range(3)]
    d = ts[(ts.run == label) & (ts.seed == SEEDS[0])]
    for a, (k, lab, c) in zip(axs, [("Phi", "polarization Φ", "#4a3aa7"), ("M", "mixing M", "#1baf7a"),
                                    ("delta0" if "cycle" in label else "clusters_5",
                                     "δ" if "cycle" in label else "contact clusters ≥5", "#eb6834")]):
        a.plot(d.t * DT, d[k], color=c, lw=1); a.set_title(lab, fontsize=8, loc="left"); a.tick_params(labelsize=7); fm.style(a)
    axs[-1].set_xlabel("s (0.1 s per step)", fontsize=7)
    cursors = [a.axvline(0, color=fm.INK, lw=0.8) for a in axs]
    title = fig.text(0.02, 0.95, "", fontsize=9)
    c0 = [None]

    def draw(i):
        t, X, U, dl = fr[i]
        c = centre(X)
        c0[0] = c if c0[0] is None else 0.8 * c0[0] + 0.2 * c
        ax.cla()
        frame_axes(X, U, kind, ax, half=half, c=c0[0])
        ax.view_init(elev=20, azim=-60 + 0.25 * i)
        title.set_text(f"3D  {label}   t = {t * DT:5.0f} s   δ = {dl:.3f}   (camera follows the flock, slowly turning)")
        for cu in cursors:
            cu.set_xdata([t * DT, t * DT])
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(fr), blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=30, bitrate=2400), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(F3_OUT, exist_ok=True)
    t0 = time.time()
    js = jobs()
    js.sort(key=lambda j: -(j[1] == 3) * j[5])                 # slow 3D runs first
    outs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one)(*j) for j in js)
    ts = pd.DataFrame([r for rows, _ in outs for r in rows])
    tag = os.environ.get("TAG", "")
    ts.to_csv(os.path.join(F3_OUT, f"flock3d_timeseries{tag}.csv"), index=False)
    late = ts[ts.t >= ts.groupby("run").t.transform("max") / 2]
    num = [k for k, _ in METRICS] + ["C", "nn", "mid_thin", "long_vs_heading", "thin_vs_heading", "gap_max", "extent"]
    per_run = late.groupby(["run", "dim", "w", "seed"])[num].mean().reset_index()
    per_run["delta"] = per_run.run.str.extract(r"_dl([0-9.]+)")[0].str.rstrip(".").astype(float)
    per_run.round(4).to_csv(os.path.join(F3_OUT, f"flock3d_per_run{tag}.csv"), index=False)
    summ = per_run.groupby("run")[num].agg(["mean", "std"]).round(3)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(F3_OUT, f"flock3d_summary{tag}.csv"))
    finals = {}
    for (lab, kind, fr, X, U), j in zip([o for _, o in outs], js):
        if j[4] == SEEDS[0]:
            finals[lab] = (kind, X, U)
    if "const" in STAGES:
        snapshots(finals, ts, os.path.join(F3_OUT, "flock3d_snapshots.png"))
        snapshots2d(finals, ts, os.path.join(F3_OUT, "flock2d_snapshots.png"))
        compare_plot(per_run, os.path.join(F3_OUT, "flock3d_vs_2d.png"))
        traces_plot(ts, [f"d{d}_w0.8_dl{x:g}" for d in (2, 3) for x in DELTAS[::2]], os.path.join(F3_OUT, "flock3d_traces.png"))
    if "cycle" in STAGES:
        traces_plot(ts, [f"d{d}_w0.8_cycle{PERIOD}" for d in (2, 3)], os.path.join(F3_OUT, f"flock3d_cycle_traces{tag}.png"))
    films = [o for _, o in outs if o[2]]
    Parallel(n_jobs=N_JOBS)(delayed(movie)(lab, kind, fr, ts, os.path.join(F3_OUT, f"{lab}.mp4")) for lab, kind, fr, X, U in films)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(summ[[c for c in summ.columns if c.endswith("_mean")]])
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {F3_OUT}")
