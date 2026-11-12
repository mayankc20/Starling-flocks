# MIND-FLOCK in 3D under drone limits, against classical flocking rules

Code: `model/mindflock3d.py` (simulator, controllers, measures), `experiments/mindflock_compare.py` (sweeps, ~30 min on 4 cores),
`experiments/mindflock_plots.py` (figures, videos). Full CSVs and videos: `/mnt/project-files/mindflock-3d/`.

## Setup

- **3D perception (step 1).** Every direction (own heading, partners' headings, partners' bearings) enters as an angular bump
  `exp[(λ/2)(u·d − 1)]` on a 2000-point sphere grid, so `|ψ|²` is a von Mises–Fisher mixture with interference (κ = λ = 50).
  The bearing of a partner of another kind has a minus sign. Born-rule draw, memory μ = 0.8.
- **Drone limits (step 2), the same for every controller.** Sensing range R = 25 BL, field of view 360°, optional bearing noise
  (vMF), range and velocity noise, latency (delayed states, no prediction), per-link packet loss, drones that fail mid-run (land).
  Vehicle: speed ≤ 1 BL/step, acceleration ≤ 0.2 BL/step², turn rate ≤ 180°/s, cruise 0.5 BL/step.
  **Nothing removes overlaps**: centres closer than 1 BL (one drone diameter) count as a collision.
  Units: BL = drone diameter; for a 0.3 m drone, cruise is 1.5 m/s and R = 7.5 m; one step = 0.1 s.
- **Collision avoidance in MIND-FLOCK.** Every neighbour inside the safety radius r_safe adds a bump at the escape direction −b,
  amplitude 3·(r_safe − d)/(r_safe − 1). With r_safe = 5 BL (fish keep away from neighbours out to several body lengths) there
  were no collisions at nominal settings; at 2–3 BL there were.
- **Baselines (step 3)**, each tuned on small grids at nominal settings for cohesion without collisions:
  Reynolds boids (separation in r_safe, alignment + cohesion to n nearest), boids with signed cohesion (pushed away from other kinds:
  the classical analogue of the minus sign), 3D Vicsek (n nearest, vMF noise), Olfati-Saber Algorithm 1 (σ-norm lattice d = 5 BL,
  interaction 1.5 d; 1.2 d fragmented here) plus a cruise term, and Olfati-Saber with the attraction to other kinds reversed;
  MIND-FLOCK without the minus sign.
- N = 30 drones, 3 kinds, start mixed in a ball, roughly aligned. 300 s per run, measures over the last 150 s. 10 seeds (main, δ),
  6 seeds otherwise. Defaults: n = 6 partners, w = 0.5, δ = 0.3.
- Measures: segregation `seg = 1 − M/M_random` (M = mixing entropy over 6 nearest), cohesion `C_R_min` = lowest share in the largest
  group connected within range R over the measuring window, collisions per drone per minute, closest approach, polarization.
  "Whole" = C_R_min ≥ 0.99; "segregated and whole" = seg ≥ 0.5 and whole.

## Results

**Main comparison (fig_main.png).** Only MIND-FLOCK is segregated, whole and collision-free (10/10 runs).

| controller | seg | whole | collisions / drone / min | closest (BL) |
|---|---|---|---|---|
| MIND-FLOCK | 0.73 | 10/10 | 0 | 1.36 |
| MIND-FLOCK, no minus | 0.22 | 10/10 | 0.18 | 0.26 |
| boids | 0.01 | 10/10 | 0 | 2.72 |
| boids, signed cohesion | 0.59 | 0/10 (26 % in largest group) | 0 | 0.96 |
| Vicsek | 0.05 | 0/10 | 0.53 | 0.19 |
| Olfati-Saber | −0.01 | 3/10 | 0 | 1.25 |
| Olfati-Saber, signed | 0.26 | 0/10 | 0 | 4.10 |

- The classical way to segregate (push away from other kinds) sorts the kinds but they fly apart: segregation without cohesion.
- Without the minus sign MIND-FLOCK stays whole but mixed, and collides more (it packs tighter).

**δ (fig_delta.png).** Segregated and whole only at δ = 0.3 (10/10) and 0.4 (5/10). Below 0.3 the kinds separate into different
groups; above 0.5 they mix. The window is narrower than in the 2D paper (N = 240).

**Number of partners (fish attend to 11–19).** n = 6: seg 0.72; n = 12: 0.62 (still 6/6 segregated and whole, 5/6 collision-free);
n = 18: 0.37. In a group of 30 (10 per kind), more partners average the other kinds' bumps away. The fish range works up to ~12 here;
worth re-testing with N = 60+ where 18 partners is a smaller share. **Safety radius:** 2 BL → 0.65 collisions/drone/min, 3 BL → 0.07,
5 BL → 0, 7 BL → 0 but groups start to split (3/6 whole). So 5 BL is the sweet spot.

**Group size.** N = 60: seg 0.87, whole 6/6, but 4/6 runs had a near-collision (closest 0.89 BL, rate ≈ 0.003/drone/min).
N = 12 (4 per kind): whole and collision-free but not segregated (seg 0.07) — too few of each kind to form subgroups.

**Robustness (fig_robust.png).** MIND-FLOCK stays segregated and whole under low noise, 100 ms latency, 20–50 % link loss,
20 % failures and a 90°/s turn cap. It breaks under:
- high sensing noise (18° bearing, 20 % range): kinds separate into separate groups (seg 0.98, whole 0/6);
- 300–500 ms latency: still whole (5/6, 3/6) but 0.07–0.16 collisions/drone/min;
- 80 % link loss: stays whole but mixes (seg 0.01);
- 120° field of view and 10 BL range: groups split. Drones need near-omnidirectional sensing (or comms) of ~25 BL.
Boids stay whole almost everywhere but never segregate, and collide badly with 300 ms latency (2.9/drone/min) or 80 % link loss (17.6).
Olfati-Saber is collision-free but fragments under nearly every perturbation.

## Open points for the paper

- Avoidance as an escape-direction bump is a design choice; a negative bump at the bearing would interfere and can attract.
- Latency hurts collisions most; a constant-velocity prediction of delayed states is the obvious fix.
- Fairness: baselines were tuned on cohesion and collisions only, at nominal settings, with the gains listed in `mindflock3d.BASE`.
