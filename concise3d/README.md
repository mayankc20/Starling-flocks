# Concise 3D MIND-FLOCK for drones: variable speed, robot commands, camera-only, moving obstacle

Two files, nothing else needed besides numpy, scipy, matplotlib and ffmpeg.

| File | What it is |
|---|---|
| `model.py` | The model: MIND-FLOCK on direction × speed, camera-only sensing, a vehicle driven by yaw-rate / pitch-rate / acceleration commands, a look-ahead safety backup, one moving obstacle. |
| `simulate.py` | Runs one flock through the obstacle; writes the video, a plot (groups, closest pair, speeds, commands) and a CSV. |

```
python simulate.py --seed 2                     # N=30, d_safe=5 BL, arrows 2 BL: the video in the project folder
python simulate.py --d_safe 3 --N 20 --seed 1 --out out_d3
```

Units: body lengths (BL, one drone diameter) and steps of 0.1 s. Open 3D space (no walls, no periodic box).
Fully decentralised: no GPS, no messages between robots.

## 1. Robot

State: position x, yaw ψ, pitch θ, speed s; it flies along its nose e(ψ, θ) = (cos θ cos ψ, cos θ sin ψ, sin θ).
Commands each step: yaw rate r (|r| ≤ 180°/s), pitch rate q (|q| ≤ 90°/s), acceleration a (|a| ≤ 0.1 BL/step²,
about 3 m/s² for a 0.3 m drone). Pitch stays within ±60°, speed within [s_min, s_max] = [2, 10] BL/s, cruise 5 BL/s.

    ψ ← ψ + r dt,   θ ← θ + q dt,   s ← s + a,   x ← x + s e(ψ, θ)

Positions only change by flying; nothing is moved or set directly.

## 2. Percept: a wave function on direction and speed

A cue with direction d and speed σ is the bump ⟨u, s | d, σ⟩ = exp[(λ/2)(u·d − 1)] · exp[−(s − σ)² / (2 σ_s²)].

    ψ_i(u, s) = 1·⟨U_i, s_i| + c0·⟨U_i, s0|                       inertia, pull back to cruise speed
              + √w   Σ_k        ⟨ĥ_k, ŝ_k|                          partners' headings at their speeds
              + √(1−w) Σ_k (±1) ⟨b_k, ŝ_k|                          bearings, minus sign for another kind
              + Σ_j a_avoid g_j ⟨−b_j, s_esc|                       turn away; brake if the neighbour is ahead
              + a_obs g_obs ⟨around, s_obs|                         steer around the obstacle, slow if it is ahead

n = 6 partners drawn in proportion to similarity (δ = 0.3), w = 0.5, λ = 50, σ_s = 0.6 BL/s, c0 = 2.
g_j = max(closeness inside r_avoid = 2 d_safe, (τ_safe − τ_j)/τ_safe), with time to contact τ_j and τ_safe = 2 s.
The Born rule draws a target heading u* and speed s* from |ψ_i|².

## 3. Controller: target → commands

    r = clip(wrap(ψ* − ψ) / τ_c),   q = clip((θ* − θ) / τ_c),   a = clip((s* − s) / τ_c),   τ_c = 0.3 s

## 4. Camera-only sensing (plus own IMU)

- bearing b_k: where the neighbour is in the image (von Mises–Fisher noise, κ = 3000, about 1°)
- range r_k: from apparent size (all drones the same size), 2 % noise
- neighbour velocity: v̂_k = v_i + Δ(r_k b_k)/Δt over 5 frames (0.5 s), smoothed by a simple tracker;
  gives heading ĥ_k and speed ŝ_k
- time to contact τ_j = (r_j − d_safe)/(−ṙ_j), from looming

## 5. Safety inside the wave function (soft projector)

Safety is not a separate layer: it is a soft projector P_safe applied to ψ before the Born rule,

    ψ'_i(u, s) = M_i(u, s) ψ_i(u, s),   M_i(u, s) = Π_j σ( (closest_j(u, s) − need_j) / ε ),   ε = 0.1 BL

closest_j(u, s) is the closest robot i would get to neighbour j (8 nearest) within 1 s if it flew at speed s in
direction u while j kept its camera-estimated velocity (3 s for the obstacle, which is big and closes fast).
need_j = min(d_safe + 1 BL, current distance); for the obstacle, its radius + d_safe/2 + 1 BL.
Written as ψ' = ψ − (1 − M) ψ, it is destructive interference that cancels ψ's own component in unsafe states.
A fixed negative bump would not do this: where it is larger than ψ, the Born rule squares it into a lobe
toward the neighbour. The draw stays stochastic among the safe states.

Outside ψ there are only the vehicle's turn and acceleration limits, and a fallback for the case where every
state is masked (the least unsafe state of the same grid is taken). The earlier look-ahead override on commands
is still in the code (`use_override=True`) but off.

## Results (unified model, N = 30 as 3 kinds of 10, d_safe 5 BL)

8 runs of 120 s with the obstacle (seeds 0 to 7):
- closest pair 5.61 to 6.47 BL in every run (d_safe 5), with no separate override
- closest approach to the obstacle surface 3.01 BL (margin 2.5)
- the fallback (every state masked) was never needed: 0 of 288,000 robot-decisions
- split at the obstacle in 4 of 8 runs; all 4 rejoined, 13 to 87 s after the obstacle appeared
- same-kind share of the 6 nearest neighbours over the last 10 s: 0.32 to 0.78 (fully mixed 0.31)

Video (seed 2): closest pair 5.71 BL, closest to the obstacle surface 4.73 BL, no decision outside ψ; the flock
splits into 2 groups at t ≈ 39 s and is one again at t ≈ 77 s; same-kind share rises from 0.34 to 0.77.

**Arrows overlapping on screen** is depth, not contact: the video is a 2D picture of a 3D flock. Nearer arrows are
darker and thicker, and each robot has a drop line to a floor shadow, so overlapping arrows can be seen to be at
different depths.
