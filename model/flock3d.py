"""The full model of the paper (flock_model.py, case "full") in three dimensions.

What changes from 2D:
  * headings are unit vectors U (N, 3) instead of angles; positions X are (N, 3);
  * the direction grid is a near-uniform grid of n_dir3 points on the sphere (Fibonacci grid) instead of a circle,
    and a perceived direction d enters the state as <u|d> = exp[(lam/2)(u.d - 1)] (von Mises-Fisher instead of von Mises);
  * the field of view is a cone of full angle fov_deg around the heading (270 deg: everything but a 45 deg cone behind);
  * the start is a ball of number density density3 (0.125 per unit volume: about the same spacing as 0.25 per unit area
    in 2D), headings uniform on the sphere, kinds shuffled.
Everything else is unchanged: n partners sampled from the view in proportion to similarity (delta0), own heading with
amplitude 1, partners' headings with sqrt(w), partners' bearings with sqrt(1 - w) and a minus sign for other kinds,
Born rule with memory mu, speed v0, overlap removal at r0.  w may be per agent (shape (N, 1)) and mu = 0 gives no memory.
"""
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
import flock_model as fm

BASE3 = dict(fm.BASE, n_dir3=2000, density3=0.125)


def sphere_grid(n):
    # n near-uniform unit vectors (Fibonacci grid); the mean spacing is about sqrt(4 pi / n) rad
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (1 + np.sqrt(5)) * i
    D = np.c_[r * np.cos(a), r * np.sin(a), z]
    return D, float(np.sqrt(4 * np.pi / n))


def random_unit(rng, N):
    v = rng.standard_normal((N, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def start3d(p, seed):
    # disordered, fully mixed start: positions in a ball, headings uniform on the sphere, kinds shuffled
    rng = np.random.default_rng(seed)
    N = p["N"]
    radius = (3 * N / (4 * np.pi * p["density3"])) ** (1 / 3)
    X = random_unit(rng, N) * radius * rng.random((N, 1)) ** (1 / 3)
    U = random_unit(rng, N)
    kind = rng.permutation(np.arange(N) % p["G"])
    return X, U, kind, rng


def step3d(X, U, mem, kind, S, rng, grid, p):
    D, spacing = grid
    N, n = len(U), p["n"]

    # 1. See: within range R and inside the cone of view around the heading.
    dX = X[None, :, :] - X[:, None, :]
    dist = np.linalg.norm(dX, axis=2)
    np.fill_diagonal(dist, np.inf)
    cos_half = np.cos(np.deg2rad(p["fov_deg"]) / 2)
    with np.errstate(invalid="ignore"):
        visible = (dist <= p["R"]) & (np.einsum("ijk,ik->ij", dX, U) >= cos_half * dist)

    # 2. n partners sampled from the view in proportion to similarity.
    rows = np.arange(N)[:, None]
    with np.errstate(divide="ignore"):
        score = np.where(visible, -rng.standard_exponential((N, N)) / S, -np.inf)
    partners = np.argpartition(-score, n - 1, axis=1)[:, :n]
    found = np.isfinite(np.take_along_axis(score, partners, 1))

    # 3. Directions (unit vectors) and amplitudes: own heading, partners' headings, partners' bearings
    #    (minus sign for other kinds).
    with np.errstate(invalid="ignore"):
        bearing = dX[rows, partners] / dist[rows, partners][:, :, None]
    bearing = np.where(found[:, :, None], bearing, 0.0)
    direction = np.concatenate([U[:, None, :], U[partners], bearing], axis=1)       # (N, 1 + 2n, 3)
    w = p["w"]
    amplitude = np.c_[np.ones(N), np.sqrt(w) * found, np.sqrt(1 - w) * found]
    other = kind[partners] != kind[:, None]
    amplitude[:, 1 + n:] *= np.where(other, -1.0, 1.0)
    amplitude /= np.linalg.norm(amplitude, axis=1, keepdims=True)

    # 4. Born rule: amplitudes added, squared; heading drawn from the grid, then spread within its cell.
    half = 0.5 * np.broadcast_to(np.asarray(p["lam"], float), (N,))[:, None, None]
    states = (direction.astype(np.float32) @ D.T.astype(np.float32))                # (N, 1 + 2n, n_dir)
    states = np.exp((half * (states - 1)).astype(np.float32))
    percept = np.matmul(amplitude.astype(np.float32)[:, None, :], states)[:, 0, :].astype(float)
    percept /= np.linalg.norm(percept, axis=1, keepdims=True)
    mu = p["mu"]
    mem = percept if (mem is None or mu == 0) else np.sqrt(mu) * mem + np.sqrt(1 - mu) * percept
    mem /= np.linalg.norm(mem, axis=1, keepdims=True)
    prob = mem * mem
    cdf = np.cumsum(prob, axis=1)
    pick = np.minimum((cdf < rng.random((N, 1)) * cdf[:, -1:]).sum(1), len(D) - 1)
    U = D[pick] + spacing * (rng.random((N, 3)) - 0.5)
    U /= np.linalg.norm(U, axis=1, keepdims=True)

    # 5. Move, then remove overlaps.
    X = X + p["v0"] * U
    for _ in range(p["passes"]):
        pairs = cKDTree(X).query_pairs(p["r0"], output_type="ndarray")
        if not len(pairs):
            break
        i, j = pairs.T
        d = X[j] - X[i]
        L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        push = 0.5 * (p["r0"] - L) * d / L
        np.add.at(X, i, -push)
        np.add.at(X, j, push)
    return X, U, mem


def shape_nd(X, U, kind, p):
    """Flock shape in 2D or 3D (X, U of shape (N, d)): the measures of experiments/w_sweep.shape(), plus
    the axis lengths of the largest contact cluster and where the kinds sit relative to the heading."""
    d = X.shape[1]
    lab = fm.groups(X, fm.CONTACT * p["r0"])
    sizes = np.bincount(lab)
    big = lab == sizes.argmax()
    Y = X[big] - X[big].mean(0)
    if big.sum() > d:
        ev, evec = np.linalg.eigh(np.cov(Y.T))                 # ascending
    else:
        ev, evec = np.full(d, np.nan), np.eye(d)
    ev = np.clip(ev, 1e-12, None)
    head = U[big].mean(0)
    hn = np.linalg.norm(head)
    try:
        vol = ConvexHull(Y).volume if big.sum() > d + 1 else np.nan   # area in 2D, volume in 3D
    except Exception:
        vol = np.nan
    # centroids of the kinds (whole flock): spread, and the direction of their main axis against the mean heading
    cen = np.array([X[kind == g].mean(0) for g in range(p["G"])])
    gaps = np.linalg.norm(cen[:, None] - cen[None, :], axis=2)[np.triu_indices(p["G"], 1)]
    ce = cen - cen.mean(0)
    kaxis = np.linalg.svd(ce)[2][0]
    hall = U.mean(0)
    out = dict(
        Phi=float(np.linalg.norm(hall)),
        C=float(fm.cohesion(X, p["R"])),
        M=float(fm.mixing_entropy(X, kind, p["G"])),
        clusters=int(len(sizes)),
        clusters_5=int((sizes >= 5).sum()),
        largest=float(big.mean()),
        aspect=float(np.sqrt(ev[-1] / ev[0])),                 # longest / thinnest axis of the largest cluster
        mid_thin=float(np.sqrt(ev[1] / ev[0])) if d == 3 else np.nan,   # middle / thinnest axis (3D)
        long_vs_heading=float(abs(evec[:, -1] @ head) / hn) if hn > 1e-9 else np.nan,   # |cos| longest axis vs heading
        thin_vs_heading=float(abs(evec[:, 0] @ head) / hn) if hn > 1e-9 else np.nan,    # |cos| thinnest axis vs heading
        nn=float(np.median(cKDTree(X).query(X, 2)[0][:, 1])),
        density=float(big.sum() / vol) if vol and vol > 0 else np.nan,   # agents per unit area (2D) / volume (3D)
        extent=float(2 * np.sqrt(ev[-1])),
        gap_min=float(gaps.min()), gap_max=float(gaps.max()),
        kinds_vs_heading=float(abs(kaxis @ hall) / np.linalg.norm(hall)) if np.linalg.norm(hall) > 1e-9 else np.nan,
    )
    return out


def angles_to_units(theta):
    return np.c_[np.cos(theta), np.sin(theta)]
