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

- Every constraint loosens the group. Quad ≈ flight, so the extra loosening after gravity comes from speed changing with climb
  (birds at different speeds drift apart), not from banking. Banking mainly smooths turns (172 → 55°/s).
- N = 10 flight/quad groups often split (intact 51 % / 3 % of the time); N ≥ 20 hold together.
- Real small groups are tighter than this: the model needs speed matching (or stronger attraction at speed) once speed varies.

R25 note: at sight 25 BL the same runs dispersed in all conditions (free runs intact 3–68 %).
Files: flight_small.png (means), CSVs, videos flight_N30_*.mp4, free_N30_dl0.3.mp4. Code: experiments/flight_small.py.
