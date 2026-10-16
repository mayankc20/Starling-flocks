"""Label-free flock statistics, computed the same way for real trajectories and for the 2D model.

Real data (jackdaws, pigeons) are reduced to the horizontal plane, because the model is 2D.
Distances are in body lengths, times in seconds, so data and model can be put side by side.

Usage:
  python experiments/flock_stats.py model             # single-kind 2D model runs -> OUT/model_stats.csv
  python experiments/flock_stats.py table FILE BL DT  # a table with columns t,id,x,y[,z] (metres), body length BL (m), frame step DT (s)
Settings: OUT (folder, default output/flock-stats), QUICK_TEST=1 (short runs).
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

OUT = os.environ.get("OUT", "output/flock-stats")
K_NEIGH = 6                     # neighbours used for persistence
TAUS = [0.5, 1.0, 2.0, 5.0]     # seconds, for neighbour persistence
MIN_BIRDS = 5


def frame_stats(X, V):
    """Statistics of one frame. X: (N, 2) positions in body lengths, V: (N, 2) velocities."""
    N = len(X)
    nn = cKDTree(X).query(X, 2)[0][:, 1]
    speed = np.linalg.norm(V, axis=1)
    u = V / np.maximum(speed[:, None], 1e-12)
    m = u.mean(0)
    phi = float(np.linalg.norm(m))
    # group shape: principal axes of the positions, and the long axis against the mean heading
    Xc = X - X.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(Xc.T))
    aspect = float(np.sqrt(evals[1] / max(evals[0], 1e-12)))
    heading = m / max(phi, 1e-12)
    cos_long = float(abs(evecs[:, 1] @ heading))
    # velocity correlation length: first zero crossing of the connected correlation of velocity fluctuations
    dv = V - V.mean(0)
    dv = dv / max(np.sqrt((dv ** 2).sum(1).mean()), 1e-12)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    C = dv @ dv.T
    iu = np.triu_indices(N, 1)
    d, c = D[iu], C[iu]
    xi = np.nan
    if N >= 10:
        edges = np.linspace(0, d.max(), 21)
        idx = np.digitize(d, edges) - 1
        cr = np.array([c[idx == b].mean() if np.any(idx == b) else np.nan for b in range(20)])
        mid = 0.5 * (edges[1:] + edges[:-1])
        neg = np.where(cr < 0)[0]
        if len(neg) and neg[0] > 0:
            b = neg[0]
            xi = float(mid[b - 1] + (mid[b] - mid[b - 1]) * cr[b - 1] / (cr[b - 1] - cr[b]))
    extent = float(np.sqrt(evals[1]) * 2)
    return dict(N=N, nn=float(np.median(nn)), phi=phi, speed=float(np.median(speed)), aspect=aspect,
                cos_long=cos_long, xi=xi, xi_rel=xi / extent if extent > 0 else np.nan, extent=extent)


def neighbour_sets(X, ids, k=K_NEIGH):
    k = min(k, len(X) - 1)
    nb = cKDTree(X).query(X, k + 1)[1][:, 1:]
    return {ids[i]: set(ids[nb[i]]) for i in range(len(ids))}


def persistence(frames, dt):
    """Share of the k nearest neighbours still among the k nearest after tau seconds (only birds present at both times)."""
    out = {}
    for tau in TAUS:
        lag = int(round(tau / dt))
        if lag < 1 or lag >= len(frames):
            out[f"persist_{tau}s"] = np.nan; continue
        vals = []
        for a in range(0, len(frames) - lag, max(1, lag // 2)):
            (ida, Xa), (idb, Xb) = frames[a], frames[a + lag]
            common = np.intersect1d(ida, idb)
            if len(common) < MIN_BIRDS + 1:
                continue
            sa = neighbour_sets(Xa[np.isin(ida, common)], ida[np.isin(ida, common)])
            sb = neighbour_sets(Xb[np.isin(idb, common)], idb[np.isin(idb, common)])
            vals += [len(sa[i] & sb[i]) / len(sa[i]) for i in common if len(sa[i])]
        out[f"persist_{tau}s"] = float(np.mean(vals)) if vals else np.nan
    return out


def table_stats(df, body_len, dt, label, every=1):
    """df: columns t, id, x, y (metres, horizontal plane). Velocities from finite differences per bird."""
    df = df.sort_values(["id", "t"]).copy()
    for c in ("x", "y"):
        df["v" + c] = df.groupby("id")[c].diff() / df.groupby("id")["t"].diff()
    df = df.dropna(subset=["vx", "vy"])
    times = np.sort(df["t"].unique())
    rows, frames = [], []
    for t in times:
        f = df[df["t"] == t]
        if len(f) < MIN_BIRDS:
            continue
        X = f[["x", "y"]].to_numpy() / body_len
        V = f[["vx", "vy"]].to_numpy() / body_len
        frames.append((f["id"].to_numpy(), X))
        if len(frames) % every == 0:
            rows.append(dict(t=t, **frame_stats(X, V)))
    s = pd.DataFrame(rows).median(numeric_only=True).to_dict()
    s.update(persistence(frames, dt))
    s["label"] = label
    s["frames"] = len(rows)
    return s


def model_stats(w, seed, T, record_every=5, label=None):
    """Single-kind 2D model. The "no_identity" case ignores kind, so this is one species."""
    import flock_model as fm
    p = fm.job_params(dict(G=2, w=w))   # kinds are labels only: the no_identity case treats everyone alike
    r = fm.run("no_identity", seed, p, T, frame_every=record_every)
    dt = 0.1 * record_every
    frames, rows = [], []
    burn = len(r["frames"]) // 2
    ids = np.arange(p["N"])
    for X, theta in r["frames"][burn:]:
        V = p["v0"] / 0.1 * np.c_[np.cos(theta), np.sin(theta)]      # body lengths per second
        # analyse the largest cluster only, like a filmed flock
        lab = fm.groups(X, fm.CONTACT * p["r0"])
        big = lab == np.bincount(lab).argmax()
        frames.append((ids[big], X[big]))
        rows.append(frame_stats(X[big], V[big]))
    s = pd.DataFrame(rows).median(numeric_only=True).to_dict()
    s.update(persistence(frames, dt))
    s["label"] = label or f"model w={w} seed={seed}"
    s["frames"] = len(rows)
    return s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "model"
    if mode == "model":
        quick = os.environ.get("QUICK_TEST", "0") == "1"
        T = 400 if quick else 3000
        seeds = [1] if quick else [1, 2, 3]
        res = [model_stats(w, s, T) for w in (0.6, 0.8, 0.9) for s in seeds]
        out = pd.DataFrame(res)
        out.to_csv(os.path.join(OUT, "model_stats.csv"), index=False)
        print(out.to_string())
    elif mode == "table":
        f, bl, dt = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
        df = pd.read_csv(f)
        res = table_stats(df, bl, dt, os.path.basename(f))
        print(pd.Series(res).to_string())
