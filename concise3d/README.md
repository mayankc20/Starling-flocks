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

## 5. Safety backup (on commands)

Each robot predicts its own path for 1 s holding the command, and its 8 nearest neighbours flying straight with
their estimated velocity (the obstacle with its velocity). If the path would come closer than d_safe + 0.5 BL
(or closer than it already is), it takes the safest of the 27 extreme commands (yaw rate, pitch rate, acceleration
each at −max, 0 or +max). `d_safe` is a parameter (`--d_safe`, default 5 BL).

## Results

Video (N = 30, seed 2, d_safe 5, arrows 2 BL, 120 s): closest pair 6.29 BL, closest to the obstacle surface 4.94 BL,
mean speed 5.0 BL/s (slowest robot 2.0), mean |yaw rate| 44°/s and |pitch rate| 31°/s, safety backup used in 3.2 %
of robot-steps. The flock splits into 2–3 groups at the obstacle (t ≈ 40 s) and is one group again at t ≈ 82 s.

16 runs with the obstacle (N = 30 × 8 seeds, N = 10 and 50 × 4 seeds, 120 s each):
- closest pair 6.1 to 7.0 BL in every run (d_safe 5); closest approach to the obstacle surface 3.75 BL (limit 2.5)
- mean speed stays at cruise 5 BL/s; individual robots range 2 to 10 BL/s; polar order 0.87 to 0.97
- the safety backup replaced the model's command in 2 to 4 % of robot-steps
- N = 30: split in 3 of 8 runs, rejoined in all 3 (12, 24 and 52 s after the obstacle appeared)
- N = 50: never split; N = 10: split in all 4 runs into 2 or 3 small groups that had not rejoined after 90 s

**Arrows overlapping on screen** is depth, not contact: the video is a 2D picture of a 3D flock. Nearer arrows are
darker and thicker, and each robot has a drop line to a floor shadow, so overlapping arrows can be seen to be at
different depths.
