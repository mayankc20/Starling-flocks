"""Concise 3D MIND-FLOCK in open space, with a hard safe distance and one moving spherical obstacle.

Units: lengths in body lengths (BL, one drone/bird diameter), one step = 0.1 s, speeds in BL per step.
No walls and no periodic box: the flock flies in unbounded 3D space.

One step, for every agent i
  1. See the neighbours within range R (all directions).
  2. Build the percept on a grid of directions on the sphere as a sum of angular bumps
     <u|d> = exp[(lam/2)(u.d - 1)]:
        own heading                     amplitude 1
        n partners' headings            amplitude sqrt(w)
        n partners' bearings            amplitude sqrt(1-w), with a minus sign for a partner of another kind
        escape from close neighbours    amplitude a_avoid * closeness, at the direction away from the neighbour
        steer around the obstacle       amplitude a_obs * closeness, at the direction around the obstacle
     Partners are drawn from those in view in proportion to similarity (delta = attention to other kinds).
  3. Pick the desired heading by the Born rule (probability = |percept|^2); desired velocity = v0 * heading.
  4. Safety filter at the velocity level (a discrete control barrier function, CBF): the desired velocities
     are changed as little as possible so that, for every pair, the closing speed is at most
     gamma * (d - d_safe), and toward the obstacle at most gamma * (gap - d_safe / 2). Then the distance can
     shrink only geometrically toward d_safe and never cross it.
  5. Fly: the velocity turns toward the filtered one by at most omega_max * dt and its speed changes by at
     most a_max per step, within [0, v_max]. Steps 4-5 are repeated so the filtered velocity is one the
     vehicle can reach. Positions change only by X += V (no jumps).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PARAMS = dict(
    N=30, G=3,                 # agents, kinds
    R=40.0,                    # sight range (BL)
    n=6, w=0.5, delta=0.3,     # partners, heading-vs-bearing weight, attention to other kinds
    lam=50.0, mu=0.0,          # bump sharpness, percept memory (0 = none)
    v0=0.5, v_max=1.0,         # cruise and top speed (BL/step)
    a_max=0.1,                 # max speed change per step (BL/step^2 = 10 BL/s^2, ~3 m/s^2 for a 0.3 m drone)
    omega_max_deg_s=180.0, dt=0.1,
    d_safe=3.0,                # minimum centre-to-centre distance (BL) kept by the CBF; change freely
    gamma=0.2,                 # CBF rate: closing speed <= gamma * (d - d_safe) per step
    r_avoid=6.0,               # soft escape starts inside this distance (kept > d_safe)
    a_avoid=3.0,
    obs_radius=9.0,            # obstacle radius (BL)
    obs_speed=0.3,             # obstacle speed (BL/step)
    obs_sense=8.0,             # agents react when within this distance of the obstacle surface
    a_obs=4.0,
    n_dir=2000,                # directions on the sphere grid
)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def sphere_grid(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (1 + np.sqrt(5)) * i
    return np.c_[r * np.cos(a), r * np.sin(a), z], float(np.sqrt(4 * np.pi / n))


class Flock:
    def __init__(self, p=None, seed=0):
        self.p = dict(PARAMS, **(p or {}))
        p = self.p
        self.rng = np.random.default_rng(seed)
        N = p["N"]
        # start: a ball, no two closer than 1.3 d_safe, headings near +x
        gap = 1.3 * p["d_safe"]
        radius = gap * (N / 0.4) ** (1 / 3) / 1.6
        X = np.empty((0, 3))
        while len(X) < N:
            c = unit(self.rng.standard_normal(3)) * radius * self.rng.random() ** (1 / 3)
            if not len(X) or np.linalg.norm(X - c, axis=1).min() >= gap:
                X = np.vstack([X, c])
        self.X = X
        self.V = p["v0"] * unit(np.array([1.0, 0, 0]) + 0.2 * self.rng.standard_normal((N, 3)))
        self.kind = self.rng.permutation(np.arange(N) % p["G"])
        s = p["delta"] / ((p["G"] - 1) * (1 - p["delta"]))
        self.S = np.where(self.kind[:, None] == self.kind[None, :], 1.0, s)
        self.D, self.spacing = sphere_grid(p["n_dir"])
        self.mem = None
        self.obs_c = None          # obstacle centre (None = no obstacle)
        self.obs_v = None

    def launch_obstacle(self, ahead=60.0):
        """Place the obstacle `ahead` BL in front of the flock, flying straight at it (head-on)."""
        h = unit(self.V.mean(0))
        self.obs_c = self.X.mean(0) + ahead * h
        self.obs_v = -self.p["obs_speed"] * h

    # ------------------------------------------------------------ one step
    def step(self):
        p, rng, X, V = self.p, self.rng, self.X, self.V
        N, n = len(X), p["n"]
        rows = np.arange(N)[:, None]
        rel = X[None, :, :] - X[:, None, :]
        dist = np.linalg.norm(rel, axis=2)
        np.fill_diagonal(dist, np.inf)
        vis = dist <= p["R"]
        U = unit(V)

        # partners: n drawn from those in view, in proportion to similarity
        score = np.where(vis, -rng.standard_exponential((N, N)) / self.S, -np.inf)
        k = min(n, N - 1)
        part = np.argpartition(-score, k - 1, axis=1)[:, :k]
        found = np.isfinite(np.take_along_axis(score, part, 1))
        bear = np.where(found[:, :, None], unit(rel[rows, part]), 0.0)
        head = np.where(found[:, :, None], unit(V[part]), 0.0)
        sign = np.where(self.kind[part] != self.kind[:, None], -1.0, 1.0)
        amp = np.c_[np.ones(N), np.sqrt(p["w"]) * found, np.sqrt(1 - p["w"]) * found * sign]
        amp /= np.linalg.norm(amp, axis=1, keepdims=True)
        dirs = [U[:, None, :], head, bear]

        # soft escape from the 8 nearest neighbours inside r_avoid
        m = min(8, N - 1)
        near = np.argpartition(dist, m - 1, axis=1)[:, :m]
        dn = np.take_along_axis(dist, near, 1)
        g = np.clip((p["r_avoid"] - dn) / (p["r_avoid"] - p["d_safe"]), 0, 1)
        amp = np.c_[amp, p["a_avoid"] * g]
        dirs.append(-unit(rel[rows, near]))

        # steer around the obstacle: direction = outward normal + own heading's tangential part
        if self.obs_c is not None:
            out = X - self.obs_c
            gap = np.linalg.norm(out, axis=1) - p["obs_radius"]
            nrm = unit(out)
            tang = U - (U * nrm).sum(1, keepdims=True) * nrm
            around = unit(nrm + unit(tang))
            go = np.clip((p["obs_sense"] - gap) / (p["obs_sense"] - p["d_safe"] / 2), 0, 1)
            amp = np.c_[amp, p["a_obs"] * go]
            dirs.append(around[:, None, :])

        # Born rule on the direction grid
        dirs = np.concatenate(dirs, axis=1)
        states = np.exp(0.5 * p["lam"] * (dirs.astype(np.float32) @ self.D.T.astype(np.float32) - 1))
        percept = np.matmul(amp.astype(np.float32)[:, None, :], states)[:, 0, :].astype(float)
        percept /= np.maximum(np.linalg.norm(percept, axis=1, keepdims=True), 1e-300)
        mem = percept if self.mem is None else np.sqrt(p["mu"]) * self.mem + np.sqrt(1 - p["mu"]) * percept
        self.mem = mem / np.maximum(np.linalg.norm(mem, axis=1, keepdims=True), 1e-300)
        cdf = np.cumsum(self.mem ** 2, axis=1)
        pick = np.minimum((cdf < rng.random((N, 1)) * cdf[:, -1:]).sum(1), len(self.D) - 1)
        Ud = unit(self.D[pick] + self.spacing * (rng.random((N, 3)) - 0.5))

        # safety filter (velocity level) + vehicle limits, repeated so both hold together
        if self.obs_c is not None:
            self.obs_c = self.obs_c + self.obs_v
        Vn = p["v0"] * Ud
        for _ in range(3):
            Vn = self.fly(V, self.cbf(X, Vn))
        self.V = Vn
        self.X = X + Vn                              # positions only ever change by integrating velocity

    def cbf(self, X, Vd, iters=30):
        """Smallest change of Vd such that every pair closes at most gamma*(d - d_safe) per step
        (and every agent approaches the obstacle surface at most gamma*(gap - d_safe/2))."""
        p, V = self.p, Vd.copy()
        ds, gm = p["d_safe"], p["gamma"]
        reach = ds + 2 * p["v_max"] / gm            # pairs further apart cannot violate the constraint
        pr = cKDTree(X).query_pairs(reach, output_type="ndarray")
        if len(pr):
            a, b = pr[:, 0], pr[:, 1]
            e = unit(X[b] - X[a])
            slack = -gm * (np.linalg.norm(X[b] - X[a], axis=1) - ds)
        for _ in range(iters):
            done = True
            if len(pr):
                short = slack - ((V[b] - V[a]) * e).sum(1)      # > 0: closing too fast
                bad = short > 1e-9
                if bad.any():
                    c = 0.5 * short[bad, None] * e[bad]
                    np.add.at(V, a[bad], -c)
                    np.add.at(V, b[bad], c)
                    done = False
            if self.obs_c is not None:
                out = X - self.obs_c
                r = np.linalg.norm(out, axis=1)
                nrm = out / r[:, None]
                short = -gm * (r - p["obs_radius"] - ds / 2) - ((V - self.obs_v) * nrm).sum(1)
                bad = short > 1e-9
                if bad.any():
                    V[bad] += short[bad, None] * nrm[bad]
                    done = False
            if done:
                break
        return V

    def fly(self, V, Vd):
        """Turn V toward Vd by at most omega_max*dt, change speed by at most a_max, speed in [0, v_max]."""
        p = self.p
        sp, spd = np.linalg.norm(V, axis=1), np.linalg.norm(Vd, axis=1)
        U, Ud = unit(V), unit(Vd)
        ang = np.arccos(np.clip((U * Ud).sum(1), -1, 1))
        lim = np.deg2rad(p["omega_max_deg_s"]) * p["dt"]
        t = np.minimum(1.0, lim / np.maximum(ang, 1e-12))
        perp = unit(Ud - (U * Ud).sum(1, keepdims=True) * U)
        Un = np.cos(t * ang)[:, None] * U + np.sin(t * ang)[:, None] * perp
        Un = np.where(((ang < 1e-9) | (sp < 1e-9))[:, None], Ud, Un)
        # speed: the part of the desired velocity along the new heading (brake when the turn is not done yet)
        target = spd * np.maximum(np.cos(ang * (1 - t)), 0.0)
        s = np.clip(np.clip(target, sp - p["a_max"], sp + p["a_max"]), 0, p["v_max"])
        return s[:, None] * Un

    # ------------------------------------------------------------ measures
    def groups(self, link):
        """Connected groups: agents closer than `link` are in the same group. Returns labels."""
        pr = cKDTree(self.X).query_pairs(link, output_type="ndarray")
        A = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(self.X),) * 2)
        return connected_components(A, directed=False)[1]

    def min_distance(self):
        return float(cKDTree(self.X).query(self.X, 2)[0][:, 1].min())

    def obstacle_clearance(self):
        if self.obs_c is None:
            return np.inf
        return float((np.linalg.norm(self.X - self.obs_c, axis=1) - self.p["obs_radius"]).min())
