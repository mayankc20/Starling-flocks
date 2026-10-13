"""Flight constraints on the 3D model: gravity, banked turns and speed that changes with climb.

The perception and choice of heading are those of flock3d (choose3d, with the gravity weight gamma on the states).
What the bird or drone can then do with the chosen heading U_want is limited by flight:
  * turning sideways needs a bank angle phi with tan(phi) = v * omega / g (coordinated turn); |phi| <= bank_max and phi
    can change by at most roll_rate per second, so turns are capped and cannot start or stop instantly;
  * pitching up or down is capped at pitch_rate per second;
  * speed follows energy: dv/dt = -g u_z + (v0 - v) / tau  (slows when climbing, speeds up when diving, relaxes to cruise);
  * optional speed control from the K_SPEED nearest neighbours:  + k_match (mean neighbour speed - v)
    + k_keep (mean distance of those neighbours ahead along the own heading) * dt   [k_match, k_keep per second].
Units: lengths in body lengths (r0 = 1, of L_m metres), one step = dt seconds.  g in body lengths per step^2 is
g_ms2 * dt^2 / L_m.  A quadrotor (turns without banking) is bank_max_deg = 89.9, roll_rate_deg_s = 1e6.
"""
import numpy as np
from scipy.spatial import cKDTree
import flock3d as f3

FLIGHT = dict(L_m=0.25, dt=0.1, g_ms2=9.81, bank_max_deg=60.0, roll_rate_deg_s=360.0, pitch_rate_deg_s=90.0,
              tau_s=1.0, v_min=0.4, v_max=2.0,      # v_min, v_max: speed limits as multiples of v0
              k_match=0.0, k_keep=0.0)
K_SPEED = 6


def g_model(p):
    return p["g_ms2"] * p["dt"] ** 2 / p["L_m"]


def frame(U):
    """Side (left, horizontal) and up (perpendicular to U, in the vertical plane through U) unit vectors."""
    side = np.cross(np.array([0.0, 0.0, 1.0]), U)
    n = np.linalg.norm(side, axis=1, keepdims=True)
    side = np.where(n > 1e-6, side / np.maximum(n, 1e-12), np.array([0.0, 1.0, 0.0]))
    up = np.cross(U, side)
    return side, up


def step_flight(X, U, v, bank, mem, kind, S, rng, grid, p):
    """One step with flight constraints. v: speeds (N,), bank: signed bank angles in rad (N,; > 0 = turning left)."""
    U_want, mem = f3.choose3d(X, U, mem, kind, S, rng, grid, p)
    g = g_model(p)
    side, up = frame(U)
    fwd = np.einsum("ij,ij->i", U_want, U)
    yaw = np.arctan2(np.einsum("ij,ij->i", U_want, side), fwd)      # sideways turn wanted this step (rad)
    pitch = np.arctan2(np.einsum("ij,ij->i", U_want, up), fwd)      # turn up / down wanted this step (rad)

    # banked sideways turn
    bmax = np.deg2rad(p["bank_max_deg"])
    want_bank = np.clip(np.arctan(v * yaw / g), -bmax, bmax)
    droll = np.deg2rad(p["roll_rate_deg_s"]) * p["dt"]
    bank = bank + np.clip(want_bank - bank, -droll, droll)
    yaw = g * np.tan(bank) / v
    # pitch rate cap
    dp = np.deg2rad(p["pitch_rate_deg_s"]) * p["dt"]
    pitch = np.clip(pitch, -dp, dp)

    U = (U * np.cos(yaw)[:, None] + side * np.sin(yaw)[:, None])
    U = U * np.cos(pitch)[:, None] + up * np.sin(pitch)[:, None]
    U /= np.linalg.norm(U, axis=1, keepdims=True)

    # speed: climbing costs speed, diving gives it, drag brings it back to cruise
    v_new = v - g * U[:, 2] + (p["v0"] - v) * p["dt"] / p["tau_s"]
    if p.get("k_match", 0) or p.get("k_keep", 0):
        k = min(K_SPEED, len(X) - 1)
        nb = cKDTree(X).query(X, k + 1)[1][:, 1:]
        ahead = np.einsum("ikj,ij->ik", X[nb] - X[:, None, :], U).mean(1)
        v_new += p["dt"] * (p["k_match"] * (v[nb].mean(1) - v) + p["k_keep"] * ahead * p["dt"])
    v = v_new
    v = np.clip(v, p["v_min"] * p["v0"], p["v_max"] * p["v0"])
    X = f3.move3d(X, U, v, p)
    return X, U, v, bank, mem, yaw
