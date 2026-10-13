# Small groups (10–50): neighbour switching and spacing, 2D vs 3D vs 3D + gravity

Base speed (0.5 body lengths per step), w = 0.8, δ = 0.3 / 0.5 / 0.65, 5 seeds, 300 s, measured 100–300 s. Medians over runs.
Switching = share of each bird's 3 nearest neighbours that changed in the last second.

| | switching per s | nearest neighbour (BL) | CV of group radius | Φ | runs that stayed whole |
|---|---|---|---|---|---|
| 2D | 0.13–0.19 | 1.1 | 0.09–0.55 (N = 10 worst) | 0.96–0.98 | N = 10: 86 %, others 100 % |
| 3D | 0.23–0.28 | 1.2–1.3 | 0.08–0.14 | 0.94–0.96 | 100 % |
| 3D + gravity (γ = 10) | 0.07–0.15 | 2.0–2.4 | 0.05–0.10 | 0.97 | 100 % |

- 3D gives the parakeet picture best: steady spacing with frequent neighbour swaps. Switching rises with group size.
- Gravity (level-flight preference) keeps pitch at ~6.5° (vs ~30°), halves switching and doubles spacing.
- δ barely changes switching in small groups.

Files: small_groups.png (means; a few split 2D runs inflate the 2D means), CSVs, videos small_d3_N30_*.mp4.
Code: experiments/small_groups.py, model/flock3d.py (gamma).
