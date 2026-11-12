"""MIND-FLOCK in 3D for drone swarms, with drone limits and classical baselines.

One simulator, several decision rules, the same sensing and the same vehicle for all of them.

Units: lengths in drone diameters (body lengths, BL; r_coll = 1 BL, e.g. 0.3 m), one step = dt = 0.1 s.
Speeds are in BL per step (0.5 BL/step = 5 BL/s = 1.5 m/s for a 0.3 m drone).

Pipeline of one step, for every drone i
  1. Sense (same for all controllers): neighbours within range R inside a cone of full angle fov_deg around the
     heading; each link is lost with probability p_drop (packet loss / missed detection); what is seen is the state of
     the neighbour `latency` steps ago (no prediction), with noise: bearing perturbed by a von Mises-Fisher draw of
     concentration kappa_b, range by a factor (1 + sigma_r * N(0,1)), velocity by sigma_v * v0 * N(0,1)^3.
  2. Decide (the controller): a desired velocity.
  3. Fly (same for all): the velocity turns toward the desired one by at most omega_max * dt, its speed changes by at
     most a_max * dt and stays within [0, v_max]; the drone moves. Nothing removes overlaps: two drones whose centres
     come closer than r_coll have collided, and that is counted.

Controllers
  mindflock       the full model of the paper in 3D. Own heading (amplitude 1), n partners' headings (sqrt(w)) and
                  bearings (sqrt(1-w), minus sign for another kind), partners sampled from the view in proportion to
                  similarity (delta0). Each direction d enters as the angular bump <u|d> = exp[(lam/2)(u.d - 1)] on a
                  sphere grid, i.e. a von Mises-Fisher amplitude, so |sum|^2 is a vMF mixture with interference.
                  Collision avoidance: every neighbour closer than r_safe adds a bump at the escape direction -b, of
                  amplitude a_avoid * (r_safe - d) / (r_safe - r_coll) (clipped to [0, 1]). Heading drawn by the Born
                  rule; desired speed v0.
  mindflock_plus  the same without the minus sign (phase 0), the paper's "relational partners, no phase" control.
  boids           Reynolds (1987): separation inside r_safe, alignment and cohesion to the n nearest seen, constant
                  cruise speed v0.
  boids_signed    boids with the cohesion to other kinds negated (the classical analogue of the minus sign).
  vicsek          3D Vicsek: mean velocity of self + n nearest seen, plus vMF noise of concentration lam, speed v0.
  olfati          Olfati-Saber (2006) Algorithm 1: sigma-norm gradient term toward lattice spacing d_os and velocity
                  consensus among neighbours within r_os = 1.5 d_os (1.2 in the paper fragments here), plus a cruise term c_v (v0 vhat - v) instead of
                  a shared gamma-agent (no leader, no shared target, as for the other rules).
  olfati_signed   Olfati-Saber with the attractive part of the gradient toward other kinds reversed.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CONTROLLERS = ["mindflock", "mindflock_plus", "boids", "boids_signed", "vicsek", "olfati", "olfati_signed"]
LABEL = {"mindflock": "MIND-FLOCK", "mindflock_plus": "MIND-FLOCK, no minus sign", "boids": "Reynolds boids",
         "boids_signed": "boids, signed cohesion", "vicsek": "Vicsek", "olfati": "Olfati-Saber",
         "olfati_signed": "Olfati-Saber, signed"}
KIND_AWARE = {"mindflock", "boids_signed", "olfati_signed"}   # rules that use the kind of a partner in their sign

BASE = dict(
    N=30, G=3, dt=0.1,
    # sensing
    R=25.0, fov_deg=360.0, kappa_b=np.inf, sigma_r=0.0, sigma_v=0.0, latency=0, p_drop=0.0,
    # vehicle
    v0=0.5, v_max=1.0, a_max=0.2, omega_max_deg_s=180.0,
    # safety
    r_coll=1.0, r_safe=5.0,
    # MIND-FLOCK
    n=6, w=0.5, lam=50.0, mu=0.8, delta0=0.3, a_avoid=3.0, n_dir3=2000, gamma=0.0,
    # boids
    k_sep=0.6, k_ali=0.3, k_coh=0.1,
    # olfati-saber
    d_os=5.0, r_os_ratio=1.5, eps_os=0.1, h_os=0.2, a_os=5.0, b_os=5.0, k_os=0.02, c_v=0.05,
    # start: mixed ball of number density rho0 with spacing >= start_gap, headings within ~1/sqrt(kappa0) of +x
    rho0=0.03, start_gap=2.0, kappa0=10.0,
    # failures: share of drones that drop out (land) at step fail_at
    fail_frac=0.0, fail_at=None,
)


# ---------------------------------------------------------------- geometry

def sphere_grid(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (1 + np.sqrt(5)) * i
    return np.c_[r * np.cos(a), r * np.sin(a), z], float(np.sqrt(4 * np.pi / n))


def unit(v, eps=1e-12):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def vmf_sample(mean, kappa, rng):
    """One von Mises-Fisher draw on S^2 around each row of mean (unit vectors), concentration kappa (scalar)."""
    mean = np.asarray(mean, float)
    shape = mean.shape
    m = mean.reshape(-1, 3)
    if not np.isfinite(kappa):
        return mean
    u = rng.random(len(m))
    wz = 1 + np.log(u + (1 - u) * np.exp(-2 * kappa)) / kappa           # cos of the angle to the mean (Wood 1994)
    phi = 2 * np.pi * rng.random(len(m))
    a = np.where(np.abs(m[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = unit(np.cross(m, a)); e2 = np.cross(m, e1)
    s = np.sqrt(np.clip(1 - wz * wz, 0, None))
    out = wz[:, None] * m + s[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
    return out.reshape(shape)


def start(p, seed):
    rng = np.random.default_rng(seed)
    N = p["N"]
    radius = (3 * N / (4 * np.pi * p["rho0"])) ** (1 / 3)
    X = np.empty((0, 3))
    while len(X) < N:                                                    # rejection: no two closer than start_gap
        c = unit(rng.standard_normal(3)) * radius * rng.random() ** (1 / 3)
        if not len(X) or np.min(np.linalg.norm(X - c, axis=1)) >= p["start_gap"]:
            X = np.vstack([X, c])
    U = vmf_sample(np.tile([1.0, 0, 0], (N, 1)), p["kappa0"], rng)
    kind = rng.permutation(np.arange(N) % p["G"])
    return X, p["v0"] * U, kind, rng


# ---------------------------------------------------------------- sensing

def sense(Xnow, Vnow, Xold, Vold, alive, p, rng):
    """What each drone perceives: relative positions (N, N, 3), velocities (N, N, 3), distances, a visibility mask.
    Row i is drone i's view; positions of others are delayed (Xold), its own is current."""
    N = len(Xnow)
    rel = Xold[None, :, :] - Xnow[:, None, :]
    dist = np.linalg.norm(rel, axis=2)
    np.fill_diagonal(dist, np.inf)
    vis = (dist <= p["R"]) & alive[None, :] & alive[:, None]
    if p["fov_deg"] < 360:
        h = unit(Vnow)
        cos_half = np.cos(np.deg2rad(p["fov_deg"]) / 2)
        with np.errstate(invalid="ignore"):
            vis &= np.einsum("ijk,ik->ij", rel, h) >= cos_half * dist
    if p["p_drop"] > 0:
        vis &= rng.random((N, N)) >= p["p_drop"]
    np.fill_diagonal(vis, False)
    Vobs = np.broadcast_to(Vold[None, :, :], (N, N, 3)).copy()
    if np.isfinite(p["kappa_b"]) or p["sigma_r"] > 0:
        b = vmf_sample(unit(rel), p["kappa_b"], rng)
        d = dist * np.clip(1 + p["sigma_r"] * rng.standard_normal((N, N)), 0.05, None)
        d[~np.isfinite(dist)] = np.inf
        rel = b * np.where(np.isfinite(d), d, 0)[:, :, None]
        dist = np.where(np.isfinite(dist), d, np.inf)
    if p["sigma_v"] > 0:
        Vobs += p["sigma_v"] * p["v0"] * rng.standard_normal((N, N, 3))
    dist = np.where(vis, dist, np.inf)
    return rel, Vobs, dist, vis


def nearest_k(dist, k):
    """Indices of the k nearest per row and whether each exists (finite)."""
    k = min(k, dist.shape[1] - 1)
    idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
    return idx, np.isfinite(np.take_along_axis(dist, idx, 1))


# ---------------------------------------------------------------- controllers

def ctrl_mindflock(i_state, obs, kind, S, rng, grid, p, minus=True):
    X, V, mem = i_state
    rel, Vobs, dist, vis = obs
    D, spacing = grid
    N, n = len(X), p["n"]
    rows = np.arange(N)[:, None]
    U = unit(V)
    # partners: n sampled from the view in proportion to similarity
    with np.errstate(divide="ignore"):
        score = np.where(vis, -rng.standard_exponential((N, N)) / S, -np.inf)
    n_eff = min(n, N - 1)
    partners = np.argpartition(-score, n_eff - 1, axis=1)[:, :n_eff]
    found = np.isfinite(np.take_along_axis(score, partners, 1))
    bearing = np.where(found[:, :, None], unit(rel[rows, partners]), 0.0)
    heads = np.where(found[:, :, None], unit(Vobs[rows, partners]), 0.0)
    w = p["w"]
    amp = np.c_[np.ones(N), np.sqrt(w) * found, np.sqrt(1 - w) * found]
    if minus:
        amp[:, 1 + n_eff:] *= np.where(kind[partners] != kind[:, None], -1.0, 1.0)
    amp /= np.linalg.norm(amp, axis=1, keepdims=True)
    # avoidance: every neighbour inside r_safe adds a bump at the escape direction
    k_av = min(8, N - 1)
    near, nfound = nearest_k(dist, k_av)
    dn = np.take_along_axis(dist, near, 1)
    g = np.clip((p["r_safe"] - dn) / (p["r_safe"] - p["r_coll"]), 0, 1) * nfound
    esc = np.where(nfound[:, :, None], -unit(rel[rows, near]), 0.0)
    amp = np.c_[amp, p["a_avoid"] * g]
    direction = np.concatenate([U[:, None, :], heads, bearing, esc], axis=1)
    half = 0.5 * p["lam"]
    states = direction.astype(np.float32) @ D.T.astype(np.float32)
    states = np.exp((half * (states - 1)).astype(np.float32))
    percept = np.matmul(amp.astype(np.float32)[:, None, :], states)[:, 0, :].astype(float)
    if p["gamma"]:
        percept *= np.exp(-0.5 * p["gamma"] * D[:, 2] ** 2)[None, :]
    percept /= np.maximum(np.linalg.norm(percept, axis=1, keepdims=True), 1e-300)
    mu = p["mu"]
    mem = percept if (mem is None or mu == 0) else np.sqrt(mu) * mem + np.sqrt(1 - mu) * percept
    mem /= np.maximum(np.linalg.norm(mem, axis=1, keepdims=True), 1e-300)
    prob = mem * mem
    cdf = np.cumsum(prob, axis=1)
    pick = np.minimum((cdf < rng.random((N, 1)) * cdf[:, -1:]).sum(1), len(D) - 1)
    Ud = unit(D[pick] + spacing * (rng.random((N, 3)) - 0.5))
    return p["v0"] * Ud, mem


def ctrl_boids(i_state, obs, kind, S, rng, grid, p, signed=False):
    X, V, mem = i_state
    rel, Vobs, dist, vis = obs
    N = len(X)
    rows = np.arange(N)[:, None]
    near, found = nearest_k(dist, p["n"])
    f = found[:, :, None].astype(float)
    cnt = np.maximum(f.sum(1), 1)
    r = rel[rows, near]
    ali = (Vobs[rows, near] * f).sum(1) / cnt - V * (found.any(1)[:, None])
    sgn = np.where(kind[near] != kind[:, None], -1.0, 1.0)[:, :, None] if signed else 1.0
    coh = (r * f * sgn).sum(1) / cnt
    dn = np.take_along_axis(dist, near, 1)
    g = np.clip((p["r_safe"] - dn) / (p["r_safe"] - p["r_coll"]), 0, 1) * found
    sep = -(unit(r) * g[:, :, None]).sum(1)
    Vd = V + p["k_sep"] * p["v0"] * sep + p["k_ali"] * ali + p["k_coh"] * coh
    return p["v0"] * unit(Vd), mem


def ctrl_vicsek(i_state, obs, kind, S, rng, grid, p):
    X, V, mem = i_state
    rel, Vobs, dist, vis = obs
    N = len(X)
    rows = np.arange(N)[:, None]
    near, found = nearest_k(dist, p["n"])
    s = unit(V) + (unit(Vobs[rows, near]) * found[:, :, None]).sum(1)
    return p["v0"] * vmf_sample(unit(s), p["lam"], rng), mem


def _sig(z, eps):
    return (np.sqrt(1 + eps * z * z) - 1) / eps


def _bump(z, h):
    return np.where(z < h, 1.0, np.where(z <= 1, 0.5 * (1 + np.cos(np.pi * (z - h) / (1 - h))), 0.0))


def ctrl_olfati(i_state, obs, kind, S, rng, grid, p, signed=False):
    X, V, mem = i_state
    rel, Vobs, dist, vis = obs
    eps, h, a, b = p["eps_os"], p["h_os"], p["a_os"], p["b_os"]
    r_os = p["r_os_ratio"] * p["d_os"]
    ra, da = _sig(r_os, eps), _sig(p["d_os"], eps)
    c = abs(a - b) / np.sqrt(4 * a * b)
    d = np.where(np.isfinite(dist), dist, 1e9)
    zs = _sig(d, eps)
    phi = 0.5 * ((a + b) * ((zs - da + c) / np.sqrt(1 + (zs - da + c) ** 2)) + (a - b))
    phi_a = _bump(zs / ra, h) * phi
    if signed:
        phi_a = np.where((kind[None, :] != kind[:, None]) & (phi_a > 0), -phi_a, phi_a)
    phi_a = np.where(vis, phi_a, 0.0)
    nij = rel / np.sqrt(1 + eps * d * d)[:, :, None]
    aij = np.where(vis, _bump(zs / ra, h), 0.0)
    u = (phi_a[:, :, None] * nij).sum(1) + (aij[:, :, None] * (Vobs - V[:, None, :])).sum(1)
    u += p["c_v"] * (p["v0"] * unit(V) - V)
    return V + p["k_os"] * u, mem


def decide(name, state, obs, kind, S, rng, grid, p):
    if name == "mindflock":
        return ctrl_mindflock(state, obs, kind, S, rng, grid, p, minus=True)
    if name == "mindflock_plus":
        return ctrl_mindflock(state, obs, kind, S, rng, grid, p, minus=False)
    if name == "boids":
        return ctrl_boids(state, obs, kind, S, rng, grid, p)
    if name == "boids_signed":
        return ctrl_boids(state, obs, kind, S, rng, grid, p, signed=True)
    if name == "vicsek":
        return ctrl_vicsek(state, obs, kind, S, rng, grid, p)
    if name == "olfati":
        return ctrl_olfati(state, obs, kind, S, rng, grid, p)
    if name == "olfati_signed":
        return ctrl_olfati(state, obs, kind, S, rng, grid, p, signed=True)
    raise KeyError(name)


# ---------------------------------------------------------------- vehicle

def fly(V, Vd, p):
    """Turn V toward Vd by at most omega_max dt, change speed by at most a_max dt, speed in [0, v_max]."""
    sp, spd = np.linalg.norm(V, axis=1), np.linalg.norm(Vd, axis=1)
    u, ud = unit(V), unit(Vd)
    ang = np.arccos(np.clip((u * ud).sum(1), -1, 1))
    lim = np.deg2rad(p["omega_max_deg_s"]) * p["dt"]
    t = np.where(ang > lim, lim / np.maximum(ang, 1e-12), 1.0)
    perp = unit(ud - (u * ud).sum(1, keepdims=True) * u)
    turned = np.cos(t * ang)[:, None] * u + np.sin(t * ang)[:, None] * perp
    turned = np.where((ang < 1e-9)[:, None] | (sp < 1e-9)[:, None], ud, turned)
    s = np.clip(np.clip(spd, sp - p["a_max"], sp + p["a_max"]), 0, p["v_max"])
    return s[:, None] * turned


# ---------------------------------------------------------------- measures

def components(X, link):
    if len(X) < 2:
        return np.zeros(len(X), int)
    pr = cKDTree(X).query_pairs(link, output_type="ndarray")
    A = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(X),) * 2)
    return connected_components(A, directed=False)[1]


def mixing(X, kind, G, m=6):
    m = min(m, len(X) - 1)
    _, nb = cKDTree(X).query(X, m + 1)
    frac = np.stack([(kind[nb[:, 1:]] == g).mean(1) for g in range(G)], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(frac > 0, frac * np.log(frac), 0).sum(1)
    return float(H.mean() / np.log(G))


def measures(X, V, kind, p):
    N = len(X)
    G = p["G"]
    labR = components(X, p["R"])
    labC = components(X, 2 * p["r_safe"])
    # same-kind cohesion: share of each kind in its largest same-kind contact cluster
    kc = []
    for g in range(G):
        sel = kind == g
        if sel.sum() > 1:
            kc.append(np.bincount(components(X[sel], 2 * p["r_safe"])).max() / sel.sum())
    dd, _ = cKDTree(X).query(X, 2)
    M = mixing(X, kind, G)
    Mr = np.mean([mixing(X, np.random.default_rng(k).permutation(kind), G) for k in range(3)])
    return dict(M=M, M_rand=float(Mr), seg=float(1 - M / max(Mr, 1e-9)),
                C_R=float(np.bincount(labR).max() / N), C_contact=float(np.bincount(labC).max() / N),
                kind_cohesion=float(np.mean(kc)) if kc else np.nan,
                Phi=float(np.linalg.norm(unit(V).mean(0))), nn=float(np.median(dd[:, 1])),
                radius=float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean())),
                speed=float(np.linalg.norm(V, axis=1).mean()))


# ---------------------------------------------------------------- run

def run(name, seed, p, T, every=10, burn=None, frames_every=0):
    """One run. Returns a dict: time series 'rec' (every `every` steps), collision counts, final state."""
    burn = T // 2 if burn is None else burn
    X, V, kind, rng = start(p, seed)
    N = len(X)
    grid = sphere_grid(p["n_dir3"]) if name.startswith("mindflock") else None
    delta = p["delta0"]
    s = delta / ((p["G"] - 1) * (1 - delta)) if delta < 1 else 1e9
    S = np.where(kind[:, None] == kind[None, :], 1.0, s)
    alive = np.ones(N, bool)
    hist = [(X.copy(), V.copy())] * (p["latency"] + 1)
    mem = None
    rec, frames = [], []
    in_contact = np.zeros((N, N), bool)
    new_coll, coll_steps, dmin = 0, 0, np.inf
    for t in range(T + 1):
        if p["fail_frac"] > 0 and t == (p["fail_at"] if p["fail_at"] is not None else T // 2):
            k = int(round(p["fail_frac"] * N))
            alive[rng.choice(N, k, replace=False)] = False
        Xa, Va, ka = X[alive], V[alive], kind[alive]
        if t % every == 0:
            rec.append(dict(t=t, **measures(Xa, Va, ka, p)))
        if frames_every and t % frames_every == 0:
            frames.append((X.copy(), V.copy(), alive.copy()))
        if t == T:
            break
        # collisions (centre distance < r_coll among drones still flying), counted after 2 s
        Dm = np.linalg.norm(X[:, None] - X[None, :], axis=2)
        np.fill_diagonal(Dm, np.inf)
        Dm[~alive, :] = np.inf; Dm[:, ~alive] = np.inf
        c = Dm < p["r_coll"]
        if t >= 20:
            new_coll += int(np.triu(c & ~in_contact).sum())
            coll_steps += int(c.any())
            dmin = min(dmin, float(Dm.min()))
        in_contact = c
        Xo, Vo = hist[0]
        obs = sense(X, V, Xo, Vo, alive, p, rng)
        Vd, mem = decide(name, (X, V, mem), obs, kind, S, rng, grid, p)
        Vn = fly(V, Vd, p)
        V = np.where(alive[:, None], Vn, 0.0)
        X = X + V
        hist = hist[1:] + [(X.copy(), V.copy())]
    import pandas as pd
    df = pd.DataFrame(rec)
    tail = df[df.t >= burn]
    flying_min = (T - 20) * p["dt"] / 60
    out = dict(rec=df, final=(X, V, kind, alive),
               summary=dict(**{k: float(tail[k].mean()) for k in df.columns if k != "t"},
                            C_R_min=float(tail.C_R.min()),
                            collisions_per_drone_min=new_coll / (alive.sum() * flying_min),
                            collision_steps_frac=coll_steps / max(T - 20, 1), d_min=dmin))
    if frames_every:
        out["frames"] = frames
    return out
