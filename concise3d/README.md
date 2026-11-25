# Concise 3D MIND-FLOCK for drones: variable speed, robot commands, camera-only, moving obstacle

Two files, nothing else needed besides numpy, scipy, matplotlib and ffmpeg.

| File | What it is |
|---|---|
| `model.py` | The model: MIND-FLOCK on direction × speed, camera-only sensing, a vehicle driven by yaw-rate / pitch-rate / acceleration commands, a look-ahead safety backup, one moving obstacle. |
| `simulate.py` | Runs one flock through the obstacle; writes the video, a plot (groups, closest pair, speeds, commands) and a CSV. |

```
python simulate.py --seed 4                     # N=30 (3 kinds of 10), d_safe=5 BL, arrows 2 BL: the video in the project folder
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

## 2. One persistent wave function per robot (driven, damped Schrödinger equation)

Each robot keeps one wave function ψ_i(u, s) on its perceptual space (direction u × speed s) for the whole flight:

    i ∂ψ/∂t = H_free ψ − iΓ (ψ − J_t) − i W_t ψ

- **J_t = E_t[scene]**, physical → perceptual: what the camera sees now, as a superposition of cue states
  ⟨u, s | d, σ⟩ = exp[(λ/2)(u·d − 1)] · exp[−(s − σ)² / (2 σ_s²)]:

      J_t = c0 ⟨U_i, s0| + √w Σ_k ⟨ĥ_k, ŝ_k| + √(1−w) Σ_k e^{iφ_k} ⟨b_k, ŝ_k| + Σ_j a_avoid g_j ⟨−b_j, s_esc| + a_obs g_obs ⟨around, s_obs|

  φ_k = π for another kind (the paper's minus sign). n = 6 partners drawn by similarity (δ = 0.3), w = 0.5, λ = 50.
- **H_free = D (Δ_direction + Δ_speed)**: coherent spreading, so certainty fades when nothing new is seen. ψ is held
  in the world frame (attitude from the IMU), so the robot's own turning needs no extra term.
- **Γ**: damping; ψ relaxes toward J_t with memory time 1/Γ = 0.4 s. For fixed cues the steady state is ψ = J
  (the paper's superposition); Γ → ∞ gives the earlier fresh-ψ-each-step model.
- **W_t ≥ 0**: absorbing potential on unsafe states (section 5).

Discrete step (dt = 0.1 s): ψ ← e^{−iH_free dt} ψ;  ψ ← e^{−Γdt} ψ + (1 − e^{−Γdt}) J_t;  ψ ← e^{−W dt} ψ.
So the paper's memory μ = e^{−2Γdt} comes out of the dynamics.

**Read-out, perceptual → physical:** every 0.1 s the next heading and speed are one measurement of the
heading–speed operator, P(u, s) = ⟨ψ|E(u, s)|ψ⟩ / ⟨ψ|ψ⟩ (Born rule), with no collapse: ψ keeps evolving.
Measurement rather than the expectation ⟨ψ|û|ψ⟩, because the expectation averages two modes (left and right of an
obstacle averages to straight into it).

## 3. Controller: target → commands

    r = clip(wrap(ψ* − ψ) / τ_c),   q = clip((θ* − θ) / τ_c),   a = clip((s* − s) / τ_c),   τ_c = 0.3 s

## 4. Camera-only sensing (plus own IMU)

- bearing b_k: where the neighbour is in the image (von Mises–Fisher noise, κ = 3000, about 1°)
- range r_k: from apparent size (all drones the same size), 2 % noise
- neighbour velocity: v̂_k = v_i + Δ(r_k b_k)/Δt over 5 frames (0.5 s), smoothed by a simple tracker;
  gives heading ĥ_k and speed ŝ_k
- time to contact τ_j = (r_j − d_safe)/(−ṙ_j), from looming

## 5. Safety inside the wave function (absorbing potential)

Safety is not a separate layer: each step ψ is multiplied by e^{−W dt} = M (a soft projector P_safe onto safe states),

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

## Results (persistent ψ, N = 30 as 3 kinds of 10, d_safe 5 BL)

6 runs of 120 s with the obstacle for each memory time 1/Γ:

| memory 1/Γ | closest pair (BL) | closest to obstacle (BL) | fallback | split | rejoined within the run | same-kind share, last 10 s |
|---|---|---|---|---|---|---|
| 0.1 s | 5.29–6.61 | 3.88–4.69 | 0 | 1 of 6 | 1 of 1 | 0.35–0.80 |
| 0.4 s (default) | 6.10–6.52 | 3.90–4.55 | 0 | 2 of 6 | 2 of 2 | 0.36–0.74 |
| 1 s | 5.70–6.70 | 3.28–4.75 | 0 | 5 of 6 | 3 of 5 | 0.40–0.68 |

d_safe 5, obstacle margin 2.5, fully mixed 0.31. Longer memory makes the flock split more and rejoin more slowly.

Video (seed 4, memory 0.4 s): closest pair 6.51 BL, closest to the obstacle surface 3.94 BL, no decision outside ψ;
the flock splits into 2 groups at t ≈ 49 s, just after passing the obstacle, and is one group again at t ≈ 79 s;
same-kind share rises from 0.32 to 0.58.

**Arrows overlapping on screen** is depth, not contact: the video is a 2D picture of a 3D flock. Nearer arrows are
darker and thicker, and each robot has a drop line to a floor shadow, so overlapping arrows can be seen to be at
different depths.
