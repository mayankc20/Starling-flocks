# Flight constraints (gravity, banking, speed with climb) on small 3D groups

model/flight3d.py: heading chosen as in the 3D model, then flown under constraints: banked turns (tan φ = vω/g, |φ| ≤ 60°,
roll ≤ 360°/s), pitch ≤ 90°/s, speed dv/dt = −g u_z + (v0 − v)/τ (τ = 1 s), gravity weight γ = 10 on the states.
"quad": same but turns without banking. Cruise 2 body lengths per step (5 m/s for a 0.25 m bird), w = 0.8, δ = 0.3 / 0.65,
N = 10–50, 5 seeds, 300 s, start mixed and flying level. Sight range 100 body lengths: at the base 25 every condition breaks up
at this speed (first run kept in the R25 note below). Medians over runs.

| condition | switching per s | nearest neighbour (BL) | group radius (BL) | mean turn rate | |bank| | speed CV |
|---|---|---|---|---|---|---|
| free 3D | 0.32 | 3.6 | 10 | 172°/s | – | 0 |
| + gravity weight | 0.13 | 6.8 | 20 | 140°/s | – | 0 |
| + banking + speed (flight) | 0.11 | 9.6 | 47 | 55°/s | 24° | 7 % |
| quadrotor (speed, no banking) | 0.13 | 8.0 | 41 | 141°/s | – | 7 % |
| flight + speed control (flight_sc) | 0.20 | 4.1 | 16 | 55°/s | 23° | 5 % |

- Every constraint loosens the group. Quad ≈ flight, so the extra loosening after gravity comes from speed changing with climb
  (birds at different speeds drift apart), not from banking. Banking mainly smooths turns (172 → 55°/s).
- N = 10 flight/quad groups often split (intact 51 % / 3 % of the time); N ≥ 20 hold together.
- Speed control fixes most of it (flight_sc: each bird matches the speed of its 6 nearest at 2/s and speeds up toward
  neighbours ahead at 3/s). Every group stays whole, N = 10 included; radius 11–20 BL (vs 42–64 without), nearest neighbour
  3.5–4.7 BL, switching 0.13–0.24 per s rising with N, polarization 0.98. That is close to free 3D in spacing while keeping
  level flight and smooth banked turns.

R25 note: at sight 25 BL the same runs dispersed in all conditions (free runs intact 3–68 %).
Files: flight_small.png (means; flight_sc not in it), flight_sc_*.csv, CSVs, flight_sc_N30_dl0.3.mp4, videos flight_N30_*.mp4, free_N30_dl0.3.mp4. Code: experiments/flight_small.py.
