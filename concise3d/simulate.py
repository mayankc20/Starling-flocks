"""Run one flock through a moving obstacle and make the video.

    python simulate.py                     # defaults: N=30, d_safe=3 BL, seed 1 -> out/
    python simulate.py --d_safe 4 --N 20 --seed 2 --out out_d4

Outputs (in --out):  flock_obstacle.mp4   the 3D video, camera follows the flock
                     split_rejoin.png     groups over time and minimum distance over time
                     timeseries.csv       the same numbers, one row per step
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from model import Flock, unit

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=30)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--d_safe", type=float, default=3.0, help="minimum distance between agents kept by the velocity-level safety filter (BL)")
ap.add_argument("--T", type=int, default=1200, help="steps (0.1 s each)")
ap.add_argument("--launch", type=int, default=300, help="step at which the obstacle appears ahead")
ap.add_argument("--every", type=int, default=2, help="video frame every k steps")
ap.add_argument("--out", default="out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# drawn arrow length: shorter than d_safe, so two arrows (each inside a ball of radius L/2) can never touch
ARROW = 2.0 * a.d_safe / 3.0
f = Flock(dict(N=a.N, d_safe=a.d_safe, r_avoid=2 * a.d_safe), seed=a.seed)
LINK = 1.7 * f.p["r_avoid"]       # agents closer than this belong to the same group (10 BL at d_safe = 3)


def n_groups(lab, min_size=2):
    return int((np.bincount(lab) >= min_size).sum())


rows, frames = [], []
for t in range(a.T + 1):
    if t == a.launch:
        f.launch_obstacle(ahead=60.0)
    lab = f.groups(LINK)
    rows.append((t * f.p["dt"], n_groups(lab), np.bincount(lab).max() / a.N, f.min_distance(), f.obstacle_clearance(),
                 np.linalg.norm(f.V, axis=1).mean(), np.linalg.norm(f.V, axis=1).min()))
    if t % a.every == 0:
        frames.append((f.X.copy(), f.V.copy(), None if f.obs_c is None else f.obs_c.copy(), lab, t))
    if t < a.T:
        f.step()
R = np.array(rows)
np.savetxt(os.path.join(a.out, "timeseries.csv"), R, delimiter=",", fmt="%.4f",
           header="time_s,groups,largest_group_share,min_pair_distance_BL,min_obstacle_gap_BL,mean_speed,min_speed", comments="")
print(f"min pair distance over run: {R[:, 3].min():.3f} BL (d_safe {a.d_safe}, arrow length {ARROW:.2f})")
print(f"min gap to obstacle surface: {R[:, 4].min():.3f} BL (limit {a.d_safe / 2})")

# ------------------------------------------------------------------ static plot
t_launch = a.launch * f.p["dt"]
fig, ax = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
ax[0].step(R[:, 0], R[:, 1], where="post", color="k", lw=1)
ax[0].set_ylabel("groups (size >= 2)"); ax[0].yaxis.get_major_locator().set_params(integer=True)
ax0b = ax[0].twinx()
ax0b.plot(R[:, 0], R[:, 2], color="tab:blue", lw=1)
ax0b.set_ylabel("largest group share", color="tab:blue")
ax0b.set_ylim(0, 1.05)
ax[1].plot(R[:, 0], R[:, 3], color="tab:red", lw=1, label="closest pair")
ax[1].axhline(a.d_safe, color="tab:red", ls="--", lw=1, label=f"d_safe = {a.d_safe:g}")
ax[1].axhline(ARROW, color="grey", ls=":", lw=1, label=f"drawn arrow = {ARROW:.2f}")
ax[1].set_ylabel("distance (BL)"); ax[1].legend(loc="upper right", fontsize=8)
ax[2].plot(R[:, 0], 10 * R[:, 5], color="tab:green", lw=1, label="mean")
ax[2].plot(R[:, 0], 10 * R[:, 6], color="tab:green", lw=0.8, ls="--", label="slowest agent")
ax[2].set_ylabel("speed (BL/s)"); ax[2].set_xlabel("time (s)"); ax[2].legend(loc="lower right", fontsize=8)
for x in ax:
    x.axvline(t_launch, color="orange", lw=1)
ax[0].set_title(f"N = {a.N}, obstacle appears 60 BL ahead at t = {t_launch:g} s (orange)")
fig.tight_layout()
fig.savefig(os.path.join(a.out, "split_rejoin.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ video
COL = np.array(["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"])
HALF = 32.0
fig = plt.figure(figsize=(9, 8.2))
ax3 = fig.add_axes([0.0, 0.25, 1.0, 0.75], projection="3d")
axg = fig.add_axes([0.1, 0.06, 0.82, 0.16])
axg.step(R[:, 0], R[:, 1], where="post", color="k", lw=1)
axg.set_xlim(0, R[-1, 0]); axg.set_ylim(0, max(R[:, 1].max(), 2) + 0.5)
axg.set_xlabel("time (s)"); axg.set_ylabel("groups")
axg.axvline(t_launch, color="orange", lw=1)
cursor = axg.axvline(0, color="tab:red", lw=1.5)
u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
sph = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
cen_s = frames[0][0].mean(0)


def draw(k):
    global cen_s
    X, V, oc, lab, t = frames[k]
    cen_s = 0.9 * cen_s + 0.1 * X.mean(0)          # smoothed camera centre
    ax3.cla()
    U = unit(V) * ARROW
    P = X - U / 2                                   # arrows centred on the agent
    ax3.quiver(P[:, 0], P[:, 1], P[:, 2], U[:, 0], U[:, 1], U[:, 2], colors=COL[f.kind % 5],
               arrow_length_ratio=0.4, linewidth=1.6)
    ax3.scatter(X[:, 0], X[:, 1], X[:, 2], s=6, c=COL[f.kind % 5], depthshade=False)
    if oc is not None:
        r = f.p["obs_radius"]
        ax3.plot_surface(oc[0] + r * sph[0], oc[1] + r * sph[1], oc[2] + r * sph[2],
                         color="grey", alpha=0.35, linewidth=0)
    for i, c in enumerate("xyz"):
        getattr(ax3, f"set_{c}lim")(cen_s[i] - HALF, cen_s[i] + HALF)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel("x (BL)"); ax3.set_ylabel("y (BL)"); ax3.set_zlabel("z (BL)")
    ax3.view_init(elev=25, azim=-60 + 0.05 * t)
    i = t
    ax3.set_title(f"t = {R[i, 0]:5.1f} s   groups = {int(R[i, 1])}   closest pair = {R[i, 3]:.2f} BL "
                  f"(d_safe {a.d_safe:g})", fontsize=10)
    cursor.set_xdata([R[i, 0]] * 2)


anim = FuncAnimation(fig, draw, frames=len(frames))
fps = 25
anim.save(os.path.join(a.out, "flock_obstacle.mp4"), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=100)
print(f"video: {len(frames)} frames at {fps} fps = {len(frames) / fps:.0f} s, "
      f"{a.every * f.p['dt'] * fps:.0f}x real time")
