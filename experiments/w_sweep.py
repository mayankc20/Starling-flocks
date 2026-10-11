"""Sweep of w, the weight of partners' headings against partners' bearings, and its effect on flock shape.

In the model each partner enters the perceptual state twice: its heading with amplitude sqrt(w) (alignment)
and its bearing with amplitude sqrt(1 - w) (attraction). w = 0 is attraction only, w = 1 alignment only;
the paper uses w = 0.8. Everything else is the full model of the paper (BASE, case "full").

    python experiments/w_sweep.py                  # 8 values of w x 5 seeds, N = 240, 5000 steps
    QUICK=1 python experiments/w_sweep.py          # tiny check run

Results (CSV, figures) go to W_OUT (default output/w_sweep).
"""
import os, sys, time
os.environ.setdefault("OUT", os.path.join(os.path.dirname(__file__), "..", "output", "model_cache"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, ConvexHull
from joblib import Parallel, delayed
import flock_model as fm

QUICK = os.environ.get("QUICK", "0") == "1"
W_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
SEEDS = list(range(5000, 5005))
N, T, TAIL, EVERY = 240, 5000, 1000, 50       # measure shape every EVERY steps over the last TAIL steps
if QUICK:
    W_VALUES, SEEDS, N, T, TAIL, EVERY = [0.2, 0.8, 1.0], [5000], 60, 200, 100, 50
W_OUT = os.environ.get("W_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "w_sweep"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))
CONTACT = fm.CONTACT                            # body lengths: link distance of a "contact" cluster


def shape(X, theta, kind, p):
    """Shape of the flock: clusters, and elongation / density / orientation of the largest contact cluster."""
    lab = fm.groups(X, CONTACT * p["r0"])
    sizes = np.bincount(lab)
    big = lab == sizes.argmax()
    Y = X[big] - X[big].mean(0)
    ev, evec = np.linalg.eigh(np.cov(Y.T))                     # ascending eigenvalues
    major = evec[:, 1]
    head = np.array([np.cos(theta[big]).mean(), np.sin(theta[big]).mean()])
    head_norm = np.linalg.norm(head)
    area = ConvexHull(Y).volume if big.sum() >= 3 else np.nan
    return dict(
        Phi=float(np.hypot(np.cos(theta).mean(), np.sin(theta).mean())),
        C=float(fm.cohesion(X, p["R"])),                       # largest group at sight range, as in the paper
        M=float(fm.mixing_entropy(X, kind, p["G"])),
        clusters=int(len(sizes)),                              # contact clusters
        clusters_5=int((sizes >= 5).sum()),                    # contact clusters of 5 or more agents
        largest=float(big.mean()),                             # share of agents in the largest contact cluster
        aspect=float(np.sqrt(ev[1] / max(ev[0], 1e-12))),      # length / width of the largest cluster
        align_axis=float(abs(major @ head) / head_norm) if head_norm > 1e-9 else np.nan,  # |cos| major axis vs heading
        nn=float(np.median(cKDTree(X).query(X, 2)[0][:, 1])),  # median nearest-neighbour distance
        density=float(big.sum() / area) if area and area > 0 else np.nan,  # agents per unit area (convex hull)
        extent=float(2 * np.sqrt(ev[1])),                      # 2 sd along the major axis
    )


def one(w, seed):
    p = fm.job_params([("N", N), ("w", w)])
    case = fm.CASES["full"]
    dirs = np.linspace(0, 2 * np.pi, p["n_dir"], endpoint=False)
    grid = (dirs, np.cos(dirs), np.sin(dirs))
    X, theta, kind, rng = fm.start(p, seed)
    s = fm.similarity_s(p["G"], p["delta0"])
    S = np.where(kind[:, None] == kind[None, :], 1.0, s)
    mem, rows = None, []
    for t in range(T + 1):
        if t >= T - TAIL and t % EVERY == 0:
            rows.append(dict(w=w, seed=seed, t=t, **shape(X, theta, kind, p)))
        if t < T:
            X, theta, mem = fm.step(X, theta, mem, kind, S, case, rng, grid, p)
    return rows, (w, seed, X, theta, kind)


def snapshots(finals, path):
    seed0 = SEEDS[0]
    pick = {w: f for (w, s, *f) in finals if s == seed0}
    cols = 4
    nrow = int(np.ceil(len(W_VALUES) / cols))
    fig, axes = plt.subplots(nrow, cols, figsize=(3.2 * cols, 3.2 * nrow))
    for ax, w in zip(np.ravel(axes), W_VALUES):
        X, theta, kind = pick[w]
        lab = fm.groups(X, CONTACT)
        big = lab == np.bincount(lab).argmax()
        c = X[big].mean(0)
        Y = X - c
        half = max(12.0, np.percentile(np.abs(Y[big]), 98) * 1.15)
        for g in range(int(kind.max()) + 1):
            m = kind == g
            ax.quiver(Y[m, 0], Y[m, 1], np.cos(theta[m]), np.sin(theta[m]), color=fm.KIND_COLORS[g],
                      angles="xy", scale_units="xy", scale=0.7, width=0.006, headwidth=3, headlength=3)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
        ax.set_title(f"w = {w}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        out = (~big).sum()
        if out:
            ax.text(0.02, 0.02, f"{out} agents outside largest cluster", transform=ax.transAxes, fontsize=7, color=fm.INK2)
    for ax in np.ravel(axes)[len(W_VALUES):]:
        ax.axis("off")
    fig.suptitle(f"Final state (step {T}), seed {seed0}, centred on the largest contact cluster; colour = kind", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def one_group_table(ts):
    """Per w: runs that end as one group (paper's rule: largest group at sight range C > C_ONE at the final step),
    and runs whose largest contact cluster holds 90 % or more of the agents."""
    last = ts[ts["t"] == ts["t"].max()]
    t = last.groupby("w").agg(runs=("seed", "size"),
                              one_group_sight=("C", lambda c: int((c > fm.C_ONE).sum())),
                              one_group_contact=("largest", lambda c: int((c >= 0.9).sum())),
                              C_mean=("C", "mean"), largest_contact_mean=("largest", "mean"),
                              contact_clusters_5_median=("clusters_5", "median"))
    return t.round(3)


def cohesion_plot(per_run, table, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    g = per_run.groupby("w")
    for ax, (k, lab) in zip(axes[:2], [("C", "largest group at sight range, C"), ("largest", "largest contact cluster (share)")]):
        ax.plot(per_run["w"], per_run[k], "o", ms=3, color="#9a9994", alpha=0.7)
        ax.errorbar(g[k].mean().index, g[k].mean(), yerr=g[k].std(), color="#4a3aa7", lw=1.6, capsize=2)
        ax.axhline(fm.C_ONE if k == "C" else 0.9, color=fm.GRID, ls="--", lw=1, zorder=0)
        ax.set_xlabel("w"); ax.set_title(lab, fontsize=9); ax.set_ylim(0, 1.05); fm.style(ax)
    ax = axes[2]
    ax.plot(table.index, table["one_group_sight"] / table["runs"], "o-", color="#4a3aa7", label="C > 0.9 (paper)")
    ax.plot(table.index, table["one_group_contact"] / table["runs"], "s--", color="#eb6834", label="contact cluster ≥ 90 %")
    ax.set_xlabel("w"); ax.set_title("share of runs ending as one group", fontsize=9); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7); fm.style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def metric_plot(per_run, path):
    panels = [("Phi", "polarization Φ"), ("aspect", "elongation (length / width)"),
              ("align_axis", "|cos| long axis vs heading\n(1 = long along motion)"), ("density", "density, agents / area"),
              ("nn", "median nearest-neighbour dist."), ("clusters_5", "contact clusters (≥5 agents)"),
              ("largest", "share in largest contact cluster"), ("M", "mixing entropy M")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    g = per_run.groupby("w")
    for ax, (k, lab) in zip(np.ravel(axes), panels):
        ax.plot(per_run["w"], per_run[k], "o", ms=3, color="#9a9994", alpha=0.7)
        ax.errorbar(g[k].mean().index, g[k].mean(), yerr=g[k].std(), color="#4a3aa7", lw=1.6, capsize=2)
        ax.axvline(0.8, color=fm.GRID, ls="--", lw=1, zorder=0)
        ax.set_xlabel("w (heading weight)"); ax.set_title(lab, fontsize=9); fm.style(ax)
    fig.suptitle(f"w sweep: N = {N}, {len(SEEDS)} seeds, mean over last {TAIL} of {T} steps (dashed: paper value w = 0.8)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(W_OUT, exist_ok=True)
    t0 = time.time()
    jobs = [(w, s) for w in W_VALUES for s in SEEDS]
    outs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one)(w, s) for w, s in jobs)
    ts = pd.DataFrame([r for rows, _ in outs for r in rows])
    ts.to_csv(os.path.join(W_OUT, "w_sweep_timeseries.csv"), index=False)
    per_run = ts.drop(columns="t").groupby(["w", "seed"]).mean().reset_index()
    per_run.to_csv(os.path.join(W_OUT, "w_sweep_per_run.csv"), index=False)
    summ = per_run.drop(columns="seed").groupby("w").agg(["mean", "std"]).round(3)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(W_OUT, "w_sweep_summary.csv"))
    metric_plot(per_run, os.path.join(W_OUT, "w_sweep_metrics.png"))
    table = one_group_table(ts)
    table.to_csv(os.path.join(W_OUT, "w_sweep_one_group.csv"))
    cohesion_plot(per_run, table, os.path.join(W_OUT, "w_sweep_one_group.png"))
    print(table)
    snapshots([f for _, f in outs], os.path.join(W_OUT, "w_sweep_snapshots.png"))
    pd.set_option("display.width", 200)
    print(per_run.drop(columns="seed").groupby("w").mean().round(3))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {W_OUT}")
