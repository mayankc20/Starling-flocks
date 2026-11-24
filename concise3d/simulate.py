"""Run one flock through a moving obstacle and make the video.

    python simulate.py                     # defaults: N=30, d_safe=5 BL, arrows 2 BL, seed 1 -> out/
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
ap.add_argument("--d_safe", type=float, default=5.0, help="minimum distance between agents kept by the velocity-level safety filter (BL)")
ap.add_argument("--arrow", type=float, default=2.0, help="drawn arrow length (BL)")
ap.add_argument("--T", type=int, default=1200, help="steps (0.1 s each)")
ap.add_argument("--launch", type=int, default=300, help="step at which the obstacle appears ahead")
ap.add_argument("--every", type=int, default=2, help="video frame every k steps")
ap.add_argument("--out", default="out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# drawn arrow length (fixed, BL); as long as it is shorter than d_safe two arrows can never touch
ARROW = a.arrow
f = Flock(dict(N=a.N, d_safe=a.d_safe, r_avoid=2 * a.d_safe), seed=a.seed)
LINK = 1.7 * f.p["r_avoid"]       # agents closer than this belong to the same group (17 BL at d_safe = 5)


def n_groups(lab, min_size=2):
    return int((np.bincount(lab) >= min_size).sum())


rows, frames = [], []
for t in range(a.T + 1):
    if t == a.launch:
        f.launch_obstacle(ahead=60.0)
    lab = f.groups(LINK)
    rows.append((t * f.p["dt"], n_groups(lab), np.bincount(lab).max() / a.N, f.min_distance(), f.obstacle_clearance(),
                 np.linalg.norm(f.V, axis=1).mean(), np.linalg.norm(f.V, axis=1).min(),
                 np.rad2deg(np.abs(f.cmd[:, 0])).mean(), np.rad2deg(np.abs(f.cmd[:, 1])).mean(),
                 10 * np.abs(f.cmd[:, 2]).mean() / f.p["dt"], f.overrides / max(f.agent_steps, 1),
                 f.same_kind_share()))
    if t % a.every == 0:
        frames.append((f.X.copy(), f.V.copy(), None if f.obs_c is None else f.obs_c.copy(), lab, t))
    if t < a.T:
        f.step()
R = np.array(rows)
np.savetxt(os.path.join(a.out, "timeseries.csv"), R, delimiter=",", fmt="%.4f",
           header="time_s,groups,largest_group_share,min_pair_distance_BL,min_obstacle_gap_BL,mean_speed,min_speed,mean_abs_yaw_rate_deg_s,mean_abs_pitch_rate_deg_s,mean_abs_accel_BL_s2,safety_override_share_so_far,same_kind_share_of_6_nearest", comments="")
print(f"min pair distance over run: {R[:, 3].min():.3f} BL (d_safe {a.d_safe}, arrow length {ARROW:.2f})")
print(f"min gap to obstacle surface: {R[:, 4].min():.3f} BL (limit {a.d_safe / 2})")
print(f"kinds: {np.bincount(f.kind).tolist()}; same-kind share of 6 nearest: start {R[0, 11]:.2f}, "
      f"mean over run {R[:, 11].mean():.2f}, end {R[-1, 11]:.2f} (well mixed {(a.N / f.p['G'] - 1) / (a.N - 1):.2f})")
print(f"safety backup replaced the model's command in {100 * R[-1, 10]:.2f}% of robot-steps")
print(f"speed range over run: {10 * R[:, 6].min():.2f} to {10 * R[:, 5].max():.2f} BL/s (mean of means {10 * R[:, 5].mean():.2f})")

# ------------------------------------------------------------------ static plot
t_launch = a.launch * f.p["dt"]
fig, ax = plt.subplots(4, 1, figsize=(8, 9.5), sharex=True)
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
ax[2].axhline(10 * f.p["s_min"], color="grey", ls=":", lw=1); ax[2].axhline(10 * f.p["s_max"], color="grey", ls=":", lw=1)
ax[2].set_ylabel("speed (BL/s)"); ax[2].legend(loc="lower right", fontsize=8)
ax[3].plot(R[:, 0], R[:, 7], lw=0.8, label="|yaw rate| (deg/s)")
ax[3].plot(R[:, 0], R[:, 8], lw=0.8, label="|pitch rate| (deg/s)")
ax[3].set_ylabel("mean command"); ax[3].set_xlabel("time (s)"); ax[3].legend(loc="upper right", fontsize=8)
for x in ax:
    x.axvline(t_launch, color="orange", lw=1)
ax[0].set_title(f"N = {a.N}, obstacle appears 60 BL ahead at t = {t_launch:g} s (orange)")
fig.tight_layout()
fig.savefig(os.path.join(a.out, "split_rejoin.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ video
COL = np.array(["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"])
HALF = 40.0
fig = plt.figure(figsize=(9, 10))
ax3 = fig.add_axes([0.0, 0.36, 1.0, 0.64], projection="3d")
for g in range(f.p["G"]):
    fig.text(0.04, 0.96 - 0.025 * g, f"kind {'ABCDE'[g]}: {int((f.kind == g).sum())} agents", color=COL[g], fontsize=9)
# live panel 1: distances (drawn up to the current time)
axd = fig.add_axes([0.1, 0.17, 0.82, 0.17])
gap = np.where(np.isfinite(R[:, 4]), R[:, 4], np.nan)
axd.set_xlim(0, R[-1, 0])
axd.set_ylim(0, max(np.nanmax(R[:, 3]), 2 * a.d_safe) * 1.1)
axd.axhline(a.d_safe, color="tab:red", ls="--", lw=1, label=f"safe distance d_safe = {a.d_safe:g}")
axd.axhline(a.d_safe / 2, color="tab:purple", ls="--", lw=1, label=f"obstacle margin d_safe/2 = {a.d_safe / 2:g}")
axd.axhline(ARROW, color="grey", ls=":", lw=1, label=f"arrow length {ARROW:g}")
ln_pair, = axd.plot([], [], color="tab:red", lw=1.2, label="closest pair of agents")
ln_obs, = axd.plot([], [], color="tab:purple", lw=1.2, label="closest agent to obstacle surface")
axd.set_ylabel("distance (BL)"); axd.tick_params(labelbottom=False)
axd.axvline(t_launch, color="orange", lw=1)
axd.legend(loc="upper right", fontsize=7, ncol=2)
# live panel 2: groups
axg = fig.add_axes([0.1, 0.05, 0.82, 0.1])
axg.set_xlim(0, R[-1, 0]); axg.set_ylim(0.5, max(R[:, 1].max(), 2) + 0.5)
ln_g, = axg.step([], [], where="post", color="k", lw=1)
axg.set_xlabel("time (s)"); axg.set_ylabel("groups")
axs = axg.twinx()
axs.set_ylim(0, 1)
MIXED = (a.N / f.p["G"] - 1) / (a.N - 1)
axs.axhline(MIXED, color="tab:green", ls=":", lw=1)
ln_s, = axs.plot([], [], color="tab:green", lw=1)
axs.set_ylabel("same-kind\nneighbours", color="tab:green", fontsize=8)
axg.axvline(t_launch, color="orange", lw=1)
u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
sph = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
cen_s = frames[0][0].mean(0)


def draw(k):
    global cen_s
    X, V, oc, lab, t = frames[k]
    cen_s = 0.9 * cen_s + 0.1 * X.mean(0)          # smoothed camera centre
    ax3.cla()
    U = unit(V) * ARROW
    P = X - U / 2                                   # arrows centred on the agent, length ARROW in BL (data units)
    # depth cue: a 2D picture of a 3D flock always shows agents at different depths on top of each other.
    # Nearer agents are drawn darker and thicker, and each has a drop line + shadow on the floor, so overlap on
    # screen can be told apart from contact (which the safety filter rules out).
    el, az = np.deg2rad(25), np.deg2rad(-60 + 0.05 * t)
    toward_cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    depth = (X - cen_s) @ toward_cam
    near = np.clip(0.5 + depth / (2 * HALF), 0, 1)  # 1 = nearest to the camera
    rgba = matplotlib.colors.to_rgba_array(COL[f.kind % 5])
    rgba[:, 3] = 0.3 + 0.7 * near
    order = np.argsort(near)                        # far first, near drawn on top
    floor = cen_s[2] - HALF
    for i in order:
        ax3.plot([X[i, 0]] * 2, [X[i, 1]] * 2, [floor, X[i, 2]], color="0.75", lw=0.5)
    ax3.scatter(X[:, 0], X[:, 1], np.full(len(X), floor), s=4, c="0.6", depthshade=False)
    for i in order:
        ax3.quiver(P[i, 0], P[i, 1], P[i, 2], U[i, 0], U[i, 1], U[i, 2], colors=[rgba[i]],
                   arrow_length_ratio=0.4, linewidth=0.8 + 1.8 * near[i])
    if oc is not None:
        r = f.p["obs_radius"]
        ax3.plot_surface(oc[0] + r * sph[0], oc[1] + r * sph[1], oc[2] + r * sph[2],
                         color="grey", alpha=0.35, linewidth=0)
        tr = np.array([fr[2] for fr in frames[:k + 1] if fr[2] is not None])     # obstacle's path so far
        ax3.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="0.4", ls="--", lw=1)
    for i, c in enumerate("xyz"):
        getattr(ax3, f"set_{c}lim")(cen_s[i] - HALF, cen_s[i] + HALF)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel("x (BL)"); ax3.set_ylabel("y (BL)"); ax3.set_zlabel("z (BL)")
    ax3.view_init(elev=25, azim=-60 + 0.05 * t)
    ax3.set_proj_type("persp", focal_length=0.5)
    i = t
    obs_txt = (f"   obstacle moving {10 * f.p['obs_speed']:g} BL/s, gap {R[i, 4]:.1f} BL"
               if np.isfinite(R[i, 4]) else "")
    ax3.set_title(f"t = {R[i, 0]:5.1f} s   closest pair {R[i, 3]:.2f} BL (d_safe {a.d_safe:g}){obs_txt}",
                  fontsize=10, y=0.98)
    ln_pair.set_data(R[:i + 1, 0], R[:i + 1, 3])
    ln_obs.set_data(R[:i + 1, 0], gap[:i + 1])
    ln_g.set_data(R[:i + 1, 0], R[:i + 1, 1])
    ln_s.set_data(R[:i + 1, 0], R[:i + 1, 11])


anim = FuncAnimation(fig, draw, frames=len(frames))
fps = 25
anim.save(os.path.join(a.out, "flock_obstacle.mp4"), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=100)
print(f"video: {len(frames)} frames at {fps} fps = {len(frames) / fps:.0f} s, "
      f"{a.every * f.p['dt'] * fps:.0f}x real time")
