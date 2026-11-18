"""MIND-FLOCK 3D with variable speed and sharp turns (extends model/mindflock3d.py, which is left as it was).

Two things the drone model did not have:

1. Variable speed. In mindflock3d every drone wants the same cruise speed v0 all the time. Here each drone has a
   preferred speed v_pref_i(t) set by a speed regime, and a speed rule decides the desired speed from it, from the
   partners' speeds and from where the partners are (ahead or behind).

   Speed regimes (what the drones are asked to do)
     fixed   v_pref = v0 for everyone (the old model).
     steps   a shared command that jumps between levels (slow, cruise, fast) every step_s seconds, given to a share
             frac_speed of the drones (the informed ones); the others keep v0 as a weak preference (c_uninf).
     ou      every drone has its own preferred speed, an Ornstein-Uhlenbeck process around v0 (sd sd_ou, time
             constant tau_ou): drones that wander in speed on their own, as birds and imperfect drones do.

   Speed rules (how a drone picks its speed)
     const   desired speed = v_pref (no speed coordination; what mindflock3d does with v_pref = v0).
     match   deterministic relaxation, per step:
               v_des = v + k_pref c_i (v_pref - v) + k_match (mean partner speed - v)
                         + k_gap * mean partner offset along the own heading - braking,
             braking = k_brake * v * sum over neighbours ahead inside r_safe of g * cos(angle to heading) * c,
             g = (r_safe - d)/(r_safe - r_coll) clipped to [0, 1], c = closing speed / close_ref clipped to
             [0, 1] (no braking behind a neighbour that is not getting closer); c_i = 1 if informed, c_uninf
             otherwise.
     born    the model's own logic on a 1D speed grid (0 .. v_max): a speed wave function built from Gaussian bumps
             <s|c> = exp(-(s - c)^2 / (4 sig_s^2)) (so |.|^2 has width sig_s) at
               own current speed (amplitude 1), own preference (c_i), each partner's speed (sqrt(w)),
               each partner's along-heading position as a speed v + k_gap * offset (sqrt(1 - w)),
               a braking speed v (1 - g cos) for each neighbour ahead inside r_safe (a_brake * g);
             optionally with the minus sign for other kinds (speed_minus), memory mu, and a Born-rule draw.

2. Sharp turns. A share frac_turn of the drones (the informed ones, all kinds) get a goal direction: one more bump in
   their percept at the goal, amplitude a_goal. Every turn_every seconds the goal is rotated in the horizontal plane by
   turn_deg (alternating left and right, so the flock does not just spiral). The goal amplitude can carry a pulse
   that decays after each command, a_goal + a_pulse exp(-time since the command / tau_pulse): a strong, short
   "turn now" cue on top of a weak standing preference. frac_turn = 1 is an operator command to
   all drones; frac_turn = 0.2 means the rest must follow the informed ones, as in a bird flock.

Turn measures (per event, averaged): lag until the flock's mean horizontal heading is within 20 deg of the new goal,
lowest share in the largest group within R during the 15 s after the turn, collisions in that window.
"""
import numpy as np
import pandas as pd
import mindflock3d as mf

SPEED = dict(
    speed_rule="const", regime="fixed",
    # steps regime
    levels=(0.5, 0.8, 0.25), step_s=30.0, frac_speed=1.0, c_uninf=0.1,
    # ou regime
    sd_ou=0.15, tau_ou=10.0,
    # match / born rule
    k_pref=0.2, k_match=0.3, k_gap=0.003, k_brake=0.5, close_ref=0.05, v_min=0.1,
    n_speed=121, sig_s=0.03, a_brake=3.0, speed_minus=False,
    # turns
    turn_every=None, turn_deg=90.0, frac_turn=0.2, a_goal=1.0, turn_first=None, a_pulse=0.0, tau_pulse=2.0,
)


def _rotz(v, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def _partner_terms(X, V, obs, partners, found, p):
    """Partner speeds, along-heading offsets, and braking weights for neighbours ahead inside r_safe."""
    rel, Vobs, dist, vis = obs
    N = len(X)
    rows = np.arange(N)[:, None]
    U = mf.unit(V)
    ps = np.linalg.norm(Vobs[rows, partners], axis=2)                       # partner speeds (N, n)
    off = np.einsum("ijk,ik->ij", rel[rows, partners], U)                   # offsets along own heading
    near, nf = mf.nearest_k(dist, min(8, N - 1))
    dn = np.take_along_axis(dist, near, 1)
    g = np.clip((p["r_safe"] - dn) / (p["r_safe"] - p["r_coll"]), 0, 1) * nf
    cosf = np.einsum("ijk,ik->ij", mf.unit(rel[rows, near]), U)
    Vrel = Vobs[rows, near] - V[:, None, :]
    closing = -np.einsum("ijk,ijk->ij", mf.unit(rel[rows, near]), Vrel)    # > 0: the gap is shrinking
    brake = g * np.clip(cosf, 0, None) * np.clip(closing / p["close_ref"], 0, 1) * nf   # ahead and closing in
    return ps, off, brake


def speed_match(X, V, vpref, cinf, obs, partners, found, p):
    v = np.linalg.norm(V, axis=1)
    ps, off, brake = _partner_terms(X, V, obs, partners, found, p)
    cnt = np.maximum(found.sum(1), 1)
    mean_ps = np.where(found.any(1), (ps * found).sum(1) / cnt, v)
    mean_off = (np.nan_to_num(off) * found).sum(1) / cnt
    vd = v + p["k_pref"] * cinf * (vpref - v) + p["k_match"] * (mean_ps - v) + p["k_gap"] * mean_off \
        - p["k_brake"] * v * brake.sum(1)
    return np.clip(vd, p["v_min"], p["v_max"])


def speed_born(X, V, vpref, cinf, obs, partners, found, kind, smem, rng, p):
    N = len(X)
    v = np.linalg.norm(V, axis=1)
    ps, off, brake = _partner_terms(X, V, obs, partners, found, p)
    s = np.linspace(0, p["v_max"], p["n_speed"])
    w = p["w"]
    sgn = np.where(kind[partners] != kind[:, None], -1.0, 1.0) if p["speed_minus"] else 1.0
    gap_c = np.clip(v[:, None] + p["k_gap"] * np.nan_to_num(off), 0, p["v_max"])
    centres = np.c_[v, vpref, ps, gap_c, v[:, None] * (1 - brake)]
    amp = np.c_[np.ones(N), cinf, np.sqrt(w) * found * sgn, np.sqrt(1 - w) * found * sgn,
                p["a_brake"] * brake]
    psi = (amp[:, :, None] * np.exp(-(s[None, None, :] - centres[:, :, None]) ** 2 / (4 * p["sig_s"] ** 2))).sum(1)
    psi[:, s < p["v_min"]] = 0.0
    psi /= np.maximum(np.linalg.norm(psi, axis=1, keepdims=True), 1e-300)
    mu = p["mu"]
    smem = psi if (smem is None or mu == 0) else np.sqrt(mu) * smem + np.sqrt(1 - mu) * psi
    smem /= np.maximum(np.linalg.norm(smem, axis=1, keepdims=True), 1e-300)
    prob = smem * smem
    cdf = np.cumsum(prob, axis=1)
    pick = np.minimum((cdf < rng.random((N, 1)) * cdf[:, -1:]).sum(1), len(s) - 1)
    ds = s[1] - s[0]
    return np.clip(s[pick] + ds * (rng.random(N) - 0.5), p["v_min"], p["v_max"]), smem


def run(seed, p, T, every=10, burn=None, frames_every=0, name="mindflock"):
    """One run of MIND-FLOCK (name = 'mindflock' or 'mindflock_plus') with a speed regime, a speed rule and turns."""
    burn = T // 2 if burn is None else burn
    X, V, kind, rng = mf.start(p, seed)
    N = len(X)
    grid = mf.sphere_grid(p["n_dir3"])
    delta = p["delta0"]
    s_ = delta / ((p["G"] - 1) * (1 - delta)) if delta < 1 else 1e9
    S = np.where(kind[:, None] == kind[None, :], 1.0, s_)
    alive = np.ones(N, bool)
    hist = [(X.copy(), V.copy())] * (p["latency"] + 1)
    mem = smem = None
    dt = p["dt"]
    # who is informed (balanced over kinds)
    def pick_inf(frac):
        m = np.zeros(N, bool)
        for g in range(p["G"]):
            ids = np.where(kind == g)[0]
            m[rng.permutation(ids)[:int(round(frac * len(ids)))]] = True
        return m
    inf_speed = pick_inf(p["frac_speed"]) if p["regime"] == "steps" else np.ones(N, bool)
    inf_turn = pick_inf(p["frac_turn"]) if p["turn_every"] else np.zeros(N, bool)
    vpref = np.full(N, p["v0"])
    ou = np.zeros(N)
    goal = np.array([1.0, 0, 0])
    turn_steps = int(round(p["turn_every"] / dt)) if p["turn_every"] else 0
    first = int(round((p["turn_first"] if p["turn_first"] is not None else p["turn_every"] or 0) / dt))
    step_steps = int(round(p["step_s"] / dt))
    rec, frames, turns = [], [], []
    in_contact = np.zeros((N, N), bool)
    new_coll, coll_steps, dmin = 0, 0, np.inf
    coll_t = []
    k_turn = 0
    for t in range(T + 1):
        # speed command
        if p["regime"] == "steps":
            lev = p["levels"][(t // step_steps) % len(p["levels"])]
            vpref = np.where(inf_speed, lev, p["v0"])
            cmd = lev
        elif p["regime"] == "ou":
            a = dt / p["tau_ou"]
            ou = ou * (1 - a) + p["sd_ou"] * np.sqrt(2 * a) * rng.standard_normal(N)
            vpref = np.clip(p["v0"] + ou, p["v_min"], p["v_max"])
            cmd = p["v0"]
        else:
            cmd = p["v0"]
        # turn command
        if turn_steps and t >= first and (t - first) % turn_steps == 0:
            goal = _rotz(goal, p["turn_deg"] * (1 if k_turn % 2 == 0 else -1))
            turns.append(dict(t=t, goal=goal.copy()))
            k_turn += 1
        Xa, Va, ka = X[alive], V[alive], kind[alive]
        if t % every == 0:
            m = mf.measures(Xa, Va, ka, p)
            h = Va[:, :2].mean(0)
            m["goal_err_deg"] = float(np.degrees(np.arccos(np.clip(mf.unit(h) @ goal[:2], -1, 1)))) \
                if turn_steps else np.nan
            sp = np.linalg.norm(Va, axis=1)
            m.update(cmd=cmd, speed_sd=float(sp.std()), vpref_mean=float(vpref.mean()))
            rec.append(dict(t=t, **m))
        if frames_every and t % frames_every == 0:
            frames.append((X.copy(), V.copy(), alive.copy()))
        if t == T:
            break
        Dm = np.linalg.norm(X[:, None] - X[None, :], axis=2)
        np.fill_diagonal(Dm, np.inf)
        c = Dm < p["r_coll"]
        if t >= 20:
            k_new = int(np.triu(c & ~in_contact).sum())
            new_coll += k_new
            if k_new:
                coll_t += [t] * k_new
            coll_steps += int(c.any())
            dmin = min(dmin, float(Dm.min()))
        in_contact = c
        Xo, Vo = hist[0]
        obs = mf.sense(X, V, Xo, Vo, alive, p, rng)
        extra = None
        if turn_steps:
            since = (t - turns[-1]["t"]) * dt if turns else np.inf
            a_t = p["a_goal"] + p["a_pulse"] * np.exp(-since / p["tau_pulse"])
            extra = (np.broadcast_to(goal, (N, 1, 3)).copy(), (a_t * inf_turn)[:, None])
        Vd, mem, (partners, found) = mf.ctrl_mindflock((X, V, mem), obs, kind, S, rng, grid, p,
                                                       minus=(name == "mindflock"), extra=extra,
                                                       return_partners=True)
        if p["speed_rule"] == "const":
            vd = vpref
        else:
            cinf = np.where(inf_speed, 1.0, p["c_uninf"])
            if p["speed_rule"] == "match":
                vd = speed_match(X, V, vpref, cinf, obs, partners, found, p)
            else:
                vd, smem = speed_born(X, V, vpref, cinf, obs, partners, found, kind, smem, rng, p)
        Vd = mf.unit(Vd) * vd[:, None]
        V = mf.fly(V, Vd, p)
        X = X + V
        hist = hist[1:] + [(X.copy(), V.copy())]
    df = pd.DataFrame(rec)
    tail = df[df.t >= burn]
    flying_min = (T - 20) * dt / 60
    summ = {k: float(tail[k].mean()) for k in df.columns if k not in ("t",)}
    summ.update(C_R_min=float(tail.C_R.min()), collisions_per_drone_min=new_coll / (N * flying_min),
                d_min=dmin)
    # speed tracking: how well the flock's mean speed follows the command (regime steps)
    if p["regime"] == "steps":
        summ["speed_track_err"] = float(np.abs(tail.speed - tail.cmd).mean())
        # response time: after each level change, time until mean speed within 10 % of the new level
        lags = []
        for k in range(1, T // step_steps + 1):
            t0 = k * step_steps
            seg_ = df[(df.t >= t0) & (df.t < t0 + step_steps)]
            if len(seg_) == 0 or t0 < burn:
                continue
            hit = seg_[np.abs(seg_.speed - seg_.cmd.iloc[0]) < 0.1 * seg_.cmd.iloc[0]]
            lags.append((hit.t.iloc[0] - t0) * dt if len(hit) else step_steps * dt)
        summ["speed_lag_s"] = float(np.mean(lags)) if lags else np.nan
    if turns:
        ev = []
        coll_t = np.array(coll_t)
        win = int(round(15 / dt))
        for tr in turns:
            t0 = tr["t"]
            if t0 < burn or t0 + win > T:
                continue
            seg_ = df[(df.t >= t0) & (df.t <= t0 + win)]
            hit = seg_[seg_.goal_err_deg < 20]
            ev.append(dict(lag=(hit.t.iloc[0] - t0) * dt if len(hit) else np.inf,
                           C_R_min=float(seg_.C_R.min()),
                           coll=int(((coll_t >= t0) & (coll_t < t0 + win)).sum()) if len(coll_t) else 0))
        if ev:
            e = pd.DataFrame(ev)
            summ.update(turn_lag_s=float(e.lag.replace(np.inf, 15.0).mean()),
                        turn_followed=float(np.isfinite(e.lag).mean()),
                        turn_C_R_min=float(e.C_R_min.min()), turn_coll=float(e.coll.mean()))
    out = dict(rec=df, summary=summ, final=(X, V, kind, alive), turns=turns, inf_turn=inf_turn,
               inf_speed=inf_speed)
    if frames_every:
        out["frames"] = frames
    return out


def params(**kw):
    p = dict(mf.BASE)
    p.update(SPEED)
    p.update(kw)
    return p
