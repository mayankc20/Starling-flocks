"""Concise 3D MIND-FLOCK with variable speed, camera-only sensing, and one moving spherical obstacle.

Units: lengths in body lengths (BL, one drone diameter), one step = 0.1 s, speeds in BL per step.
No walls and no periodic box: the flock flies in unbounded 3D space. Fully decentralised: no GPS, no messages.

State. Each agent's percept is a wave function on direction AND speed, psi(u, s), with u on the unit sphere and
s in [s_min, s_max]. A cue (direction d, speed sigma) is the bump
    <u, s | d, sigma> = exp[(lam/2)(u.d - 1)] * exp[-(s - sigma)^2 / (2 sig_s^2)].

Percept of agent i (own heading U_i, own speed s_i):
    psi_i = 1 <U_i, s_i| + c0 <U_i, s0|                               inertia, weak pull back to cruise s0
          + sqrt(w)   sum_k      <h_k, s_k|                            n partners' headings at their speeds
          + sqrt(1-w) sum_k (+-1) <b_k, s_k|                           bearings, minus sign for another kind
          + sum_j a_avoid g_j <-b_j, s_esc,j|                          collision: turn away and brake
          + a_obs g_obs <around, s_obs|                                obstacle: steer around and slow
New heading and speed are drawn together from |psi_i|^2 (Born rule). The drone then flies there within its
turn-rate and acceleration limits; positions change only by X += V.

Camera-only measurements of agent i (own velocity v_i from its IMU / optic flow):
    bearing b_k     where the neighbour appears in the image (von Mises-Fisher noise, kappa_b)
    range r_k       from apparent size (all drones have the same known size), relative noise sigma_r
    velocity        v_k = v_i + [r_k b_k (now) - r_k b_k (m frames ago)] / m       -> heading h_k, speed s_k
    time to contact tau_j = (r_j - d_safe) / (-dr_j/dt) for approaching neighbours (looming)
Collision weight g_j = max(closeness inside r_avoid, (tau_safe - tau_j) / tau_safe), clipped to [0, 1].
Escape speed: s_min if the neighbour is ahead (brake), own speed if it is behind.

Vehicle. State: position, yaw, pitch, speed; it flies along its nose. Commands each step: yaw rate r, pitch rate
q, acceleration a, all limited (r_max, q_max, a_max; pitch within +-pitch_max; speed within [s_min, s_max]).
A proportional controller turns the Born-rule target (heading u*, speed s*) into these commands:
    r = clip(wrap(yaw* - yaw) / tau_c), q = clip((pitch* - pitch) / tau_c), a = clip((s* - s) / tau_c).

Backup safety layer (look-ahead on commands): if holding the command for `horizon` steps would bring the robot
closer than d_safe (+ buffer) to a neighbour predicted from its estimated velocity, the robot takes the safest of
the 27 extreme commands instead. How often this happens is counted (overrides).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PARAMS = dict(
    N=30, G=3,                 # agents, kinds
    R=40.0,                    # camera range (BL)
    n=6, w=0.5, delta=0.3,     # partners, heading-vs-bearing weight, attention to other kinds
    lam=50.0,                  # direction bump sharpness
    s_min=0.2, s0=0.5, s_max=1.0, n_speed=17, sig_s=0.06, c0=2.0,   # speed (BL/step): min, cruise, max
    a_max=0.1,                 # max speed change per step (BL/step^2 = 10 BL/s^2, ~3 m/s^2 for a 0.3 m drone)
    r_max_deg_s=180.0, q_max_deg_s=90.0, pitch_max_deg=60.0, dt=0.1,   # yaw-rate, pitch-rate, pitch limits
    tau_c=3.0,                 # controller time constant in steps (rate command = error / (tau_c dt))
    # camera
    kappa_b=3000.0, sigma_r=0.02, m_frames=5, beta=0.2,   # beta: tracker smoothing per frame
    # safety
    d_safe=5.0, r_avoid=10.0, a_avoid=3.0, tau_safe=20.0,     # tau_safe in steps (2 s)
    horizon=10, buffer=0.5,    # backup look-ahead (steps) and extra distance it keeps (BL)
    # obstacle
    obs_radius=9.0, obs_speed=0.3, obs_sense=12.0, a_obs=4.0,
    n_dir=2000,
)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def sphere_grid(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (1 + np.sqrt(5)) * i
    return np.c_[r * np.cos(a), r * np.sin(a), z], float(np.sqrt(4 * np.pi / n))


def vmf(mean, kappa, rng):
    """von Mises-Fisher noise around unit vectors (any leading shape)."""
    m = mean.reshape(-1, 3)
    u = rng.random(len(m))
    wz = 1 + np.log(u + (1 - u) * np.exp(-2 * kappa)) / kappa
    phi = 2 * np.pi * rng.random(len(m))
    a = np.where(np.abs(m[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    e1 = unit(np.cross(m, a)); e2 = np.cross(m, e1)
    s = np.sqrt(np.clip(1 - wz * wz, 0, None))[:, None]
    return (wz[:, None] * m + s * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)).reshape(mean.shape)


class Flock:
    def __init__(self, p=None, seed=0):
        self.p = dict(PARAMS, **(p or {}))
        p = self.p
        self.rng = np.random.default_rng(seed)
        N = p["N"]
        gap = 1.3 * p["d_safe"]
        radius = gap * (N / 0.4) ** (1 / 3) / 1.6
        X = np.empty((0, 3))
        while len(X) < N:
            c = unit(self.rng.standard_normal(3)) * radius * self.rng.random() ** (1 / 3)
            if not len(X) or np.linalg.norm(X - c, axis=1).min() >= gap:
                X = np.vstack([X, c])
        self.X = X
        self.V = p["s0"] * unit(np.array([1.0, 0, 0]) + 0.2 * self.rng.standard_normal((N, 3)))
        sp0 = np.linalg.norm(self.V, axis=1)
        self.speed = sp0
        self.yaw = np.arctan2(self.V[:, 1], self.V[:, 0])
        self.pitch = np.arcsin(self.V[:, 2] / sp0)
        self.cmd = np.zeros((N, 3))
        self.kind = self.rng.permutation(np.arange(N) % p["G"])
        s = p["delta"] / ((p["G"] - 1) * (1 - p["delta"]))
        self.S = np.where(self.kind[:, None] == self.kind[None, :], 1.0, s)
        self.D, self.spacing = sphere_grid(p["n_dir"])
        self.sg = np.linspace(p["s_min"], p["s_max"], p["n_speed"])
        self.obs_c = None
        self.obs_v = None
        self.vtrack = None         # each agent's smoothed estimate of every neighbour's velocity
        self.frames = []           # past camera frames (relative position estimates) for velocity / looming
        self.overrides = 0         # robot-steps in which the safety backup replaced the model's command
        self.agent_steps = 0

    def launch_obstacle(self, ahead=60.0):
        """Place the obstacle `ahead` BL in front of the flock, flying straight at it (head-on)."""
        h = unit(self.V.mean(0))
        self.obs_c = self.X.mean(0) + ahead * h
        self.obs_v = -self.p["obs_speed"] * h

    # ------------------------------------------------------------ camera
    def camera(self):
        """Each agent's noisy estimate of every neighbour's relative position: (N, N, 3)."""
        p, X = self.p, self.X
        rel = X[None, :, :] - X[:, None, :]
        r = np.linalg.norm(rel, axis=2)
        np.fill_diagonal(r, 1.0)
        b = vmf(rel / r[:, :, None], p["kappa_b"], self.rng)
        rh = r * (1 + p["sigma_r"] * self.rng.standard_normal(r.shape))
        return b * rh[:, :, None]

    # ------------------------------------------------------------ one step
    def step(self):
        p, rng, X, V = self.p, self.rng, self.X, self.V
        N, n = len(X), p["n"]
        rows = np.arange(N)[:, None]
        sp = np.linalg.norm(V, axis=1)
        U = unit(V)

        relh = self.camera()                              # this frame
        self.frames = (self.frames + [(relh, V.copy())])[-(p["m_frames"] + 1):]
        old = self.frames[0][0]
        m = len(self.frames) - 1
        dist = np.linalg.norm(relh, axis=2)
        np.fill_diagonal(dist, np.inf)
        vis = dist <= p["R"]
        bear = unit(relh)
        if m:
            # neighbour velocity = own velocity (IMU) + change of its relative position over m frames;
            # range rate from the change of range (looming)
            vown = np.mean([f[1] for f in self.frames[1:]], axis=0)
            vraw = vown[:, None, :] + (relh - old) / m
            self.vtrack = vraw if self.vtrack is None else (1 - p["beta"]) * self.vtrack + p["beta"] * vraw
            vhat = self.vtrack
            old_r = np.linalg.norm(old, axis=2)
            np.fill_diagonal(old_r, np.inf)
            rdot = np.nan_to_num((dist - old_r) / m, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            vhat = np.broadcast_to(V[None], (N, N, 3)).copy()
            rdot = np.zeros((N, N))
        shat = np.clip(np.linalg.norm(vhat, axis=2), p["s_min"], p["s_max"])
        hhat = unit(vhat)

        # partners: n drawn from those in view, in proportion to similarity
        score = np.where(vis, -rng.standard_exponential((N, N)) / self.S, -np.inf)
        k = min(n, N - 1)
        part = np.argpartition(-score, k - 1, axis=1)[:, :k]
        found = np.isfinite(np.take_along_axis(score, part, 1))
        sign = np.where(self.kind[part] != self.kind[:, None], -1.0, 1.0)
        dirs = [U[:, None], U[:, None], hhat[rows, part], bear[rows, part]]
        spds = [sp[:, None], np.full((N, 1), p["s0"]), shat[rows, part], shat[rows, part]]
        amp = [np.ones((N, 1)), np.full((N, 1), p["c0"]), np.sqrt(p["w"]) * found,
               np.sqrt(1 - p["w"]) * found * sign]

        # collision: nearest 8, weight from distance and from time to contact; brake if the neighbour is ahead
        mm = min(8, N - 1)
        near = np.argpartition(dist, mm - 1, axis=1)[:, :mm]
        dn = np.take_along_axis(dist, near, 1)
        rd = rdot[rows, near]
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = np.where(rd < 0, np.maximum(dn - p["d_safe"], 0) / -rd, np.inf)
        g = np.maximum(np.clip((p["r_avoid"] - dn) / (p["r_avoid"] - p["d_safe"]), 0, 1),
                       np.clip((p["tau_safe"] - tau) / p["tau_safe"], 0, 1) * (dn < p["R"]))
        bn = bear[rows, near]
        ahead = (bn * U[:, None]).sum(2) > 0
        dirs.append(-bn); spds.append(np.where(ahead, p["s_min"], sp[:, None])); amp.append(p["a_avoid"] * g)

        # obstacle (seen by the camera like a big neighbour): steer around, slow down if it is ahead
        if self.obs_c is not None:
            out = X - self.obs_c
            gap = np.linalg.norm(out, axis=1) - p["obs_radius"]
            nrm = unit(out)
            tang = U - (U * nrm).sum(1, keepdims=True) * nrm
            go = np.clip((p["obs_sense"] - gap) / (p["obs_sense"] - p["d_safe"] / 2), 0, 1)
            front = np.clip(-(U * nrm).sum(1), 0, 1)
            dirs.append(unit(nrm + unit(tang))[:, None])
            spds.append((sp - go * front * (sp - p["s_min"]))[:, None])
            amp.append(p["a_obs"] * go[:, None])

        dirs = np.concatenate(dirs, 1)
        spds = np.concatenate(spds, 1)
        amp = np.concatenate(amp, 1)
        amp = amp / np.linalg.norm(amp[:, :2 + 2 * k], axis=1, keepdims=True)
        bd = np.exp(0.5 * p["lam"] * (dirs.astype(np.float32) @ self.D.T.astype(np.float32) - 1))     # (N,K,D)
        bs = np.exp(-(self.sg[None, None, :] - spds[:, :, None]) ** 2 / (2 * p["sig_s"] ** 2))      # (N,K,S)
        psi = np.einsum("nk,nkd,nks->nds", amp.astype(np.float32), bd, bs.astype(np.float32))
        prob = (psi.astype(float) ** 2).reshape(N, -1)
        cdf = np.cumsum(prob, axis=1)
        pick = np.minimum((cdf < rng.random((N, 1)) * cdf[:, -1:]).sum(1), prob.shape[1] - 1)
        di, si = np.divmod(pick, len(self.sg))
        Ud = unit(self.D[di] + self.spacing * (rng.random((N, 3)) - 0.5))
        dsg = self.sg[1] - self.sg[0]
        sd = np.clip(self.sg[si] + dsg * (rng.random(N) - 0.5), p["s_min"], p["s_max"])

        # controller: target heading/speed -> commands (yaw rate, pitch rate, acceleration)
        if self.obs_c is not None:
            self.obs_c = self.obs_c + self.obs_v
        cmd = self.controller(Ud, sd)
        cmd, over = self.safety(X, cmd, vhat, dist)
        self.overrides += int(over.sum())
        self.agent_steps += N
        self.cmd = cmd
        self.apply(cmd)

    # ------------------------------------------------------------ vehicle
    @staticmethod
    def heading(yaw, pitch):
        return np.stack([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], -1)

    def controller(self, Ud, sd):
        """Commands per step: yaw rate r, pitch rate q (rad/s), acceleration a (BL/step^2), all clipped."""
        p = self.p
        yaw_t = np.arctan2(Ud[:, 1], Ud[:, 0])
        pmax = np.deg2rad(p["pitch_max_deg"])
        pitch_t = np.clip(np.arcsin(np.clip(Ud[:, 2], -1, 1)), -pmax, pmax)
        dyaw = (yaw_t - self.yaw + np.pi) % (2 * np.pi) - np.pi
        rmax, qmax = np.deg2rad(p["r_max_deg_s"]), np.deg2rad(p["q_max_deg_s"])
        r = np.clip(dyaw / (p["tau_c"] * p["dt"]), -rmax, rmax)
        q = np.clip((pitch_t - self.pitch) / (p["tau_c"] * p["dt"]), -qmax, qmax)
        a = np.clip((sd - self.speed) / p["tau_c"], -p["a_max"], p["a_max"])
        return np.c_[r, q, a]

    def next_state(self, cmd, idx=slice(None)):
        p = self.p
        pmax = np.deg2rad(p["pitch_max_deg"])
        yaw = self.yaw[idx] + cmd[..., 0] * p["dt"]
        pitch = np.clip(self.pitch[idx] + cmd[..., 1] * p["dt"], -pmax, pmax)
        speed = np.clip(self.speed[idx] + cmd[..., 2], p["s_min"], p["s_max"])
        return yaw, pitch, speed

    def apply(self, cmd):
        self.yaw, self.pitch, self.speed = self.next_state(cmd)
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi
        self.V = self.speed[:, None] * self.heading(self.yaw, self.pitch)
        self.X = self.X + self.V                     # positions only ever change by flying

    def safety(self, X, cmd, vhat, dist):
        """Backup, per robot and decentralised (a short look-ahead check). Each robot predicts its own path for
        H steps holding the command, and each of its 8 nearest neighbours (and the obstacle) moving straight
        with its estimated velocity. If the path would come closer than min(d_safe + buffer, now) to any of
        them, it tries the 27 extreme commands (yaw rate, pitch rate, acceleration each in {-max, 0, +max}) and
        keeps the one with the largest worst-case margin. Returns the commands and which robots were overridden."""
        p = self.p
        N, H, dt = len(X), p["horizon"], p["dt"]
        ds = p["d_safe"]
        rows = np.arange(N)[:, None]
        k = min(8, N - 1)
        near = np.argpartition(dist, k - 1, axis=1)[:, :k]
        hs = np.arange(1, H + 1)
        d_now = np.linalg.norm(X[near] - X[:, None], axis=2)                             # (N, k)
        nb_path = X[near][:, :, None] + hs[None, None, :, None] * vhat[rows, near][:, :, None]   # (N, k, H, 3)
        need = np.minimum(ds + p["buffer"], d_now)
        pmax = np.deg2rad(p["pitch_max_deg"])
        if self.obs_c is not None:
            ob_path = self.obs_c + hs[:, None] * self.obs_v                                # (H, 3)
            g_now = np.linalg.norm(X - self.obs_c, axis=1) - p["obs_radius"]
            ob_need = np.minimum(ds / 2 + p["buffer"], g_now)

        def margin(cand, idx):
            """cand (M, C, 3) -> worst margin over horizon, neighbours and obstacle: (M, C)."""
            yaw = self.yaw[idx][:, None, None] + cand[..., 0, None] * dt * hs
            pit = np.clip(self.pitch[idx][:, None, None] + cand[..., 1, None] * dt * hs, -pmax, pmax)
            spd = np.clip(self.speed[idx][:, None, None] + cand[..., 2, None] * hs, p["s_min"], p["s_max"])
            path = X[idx][:, None, None] + np.cumsum(spd[..., None] * self.heading(yaw, pit), axis=2)  # (M,C,H,3)
            dd = np.linalg.norm(nb_path[idx][:, None] - path[:, :, None], axis=-1)                    # (M,C,k,H)
            mg = (dd.min(-1) - need[idx][:, None]).min(-1)
            if self.obs_c is not None:
                go = (np.linalg.norm(path - ob_path, axis=-1) - p["obs_radius"]).min(-1)
                mg = np.minimum(mg, go - ob_need[idx][:, None])
            return mg

        base = margin(cmd[:, None], np.arange(N))[:, 0]
        bad = np.where(base < 0)[0]
        if not len(bad):
            return cmd, np.zeros(N, bool)
        rmax, qmax = np.deg2rad(p["r_max_deg_s"]), np.deg2rad(p["q_max_deg_s"])
        grid = np.array([[r, q, a] for r in (-rmax, 0, rmax) for q in (-qmax, 0, qmax)
                         for a in (-p["a_max"], 0, p["a_max"])])
        mg = margin(np.broadcast_to(grid, (len(bad),) + grid.shape), bad)
        best = mg.argmax(1)
        new = cmd.copy()
        better = mg[np.arange(len(bad)), best] > base[bad]
        new[bad[better]] = grid[best[better]]
        over = np.zeros(N, bool)
        over[bad[better]] = True
        return new, over

    # ------------------------------------------------------------ measures
    def groups(self, link):
        pr = cKDTree(self.X).query_pairs(link, output_type="ndarray")
        A = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(self.X),) * 2)
        return connected_components(A, directed=False)[1]

    def same_kind_share(self, m=6):
        """Share of each agent's m nearest neighbours that are of its own kind (well mixed: (N/G - 1)/(N - 1))."""
        m = min(m, len(self.X) - 1)
        _, nb = cKDTree(self.X).query(self.X, m + 1)
        return float((self.kind[nb[:, 1:]] == self.kind[:, None]).mean())

    def min_distance(self):
        return float(cKDTree(self.X).query(self.X, 2)[0][:, 1].min())

    def obstacle_clearance(self):
        if self.obs_c is None:
            return np.inf
        return float((np.linalg.norm(self.X - self.obs_c, axis=1) - self.p["obs_radius"]).min())
