# 3D extension of the full model, compared with 2D

Model: `model/flock3d.py` (branch claude/project-thread-fw7ydu, PR #1). Same rules as the 2D full model; headings are unit vectors drawn
from a 2000-point sphere grid (von Mises-Fisher states, λ = 50), the view is a 270° cone, start is a mixed ball. N = 240, G = 3, μ = 0.95,
5000 steps (500 s at 0.1 s/step), 3 seeds per δ, measures averaged over 250-500 s. A 4000-point grid gives the same numbers (grid4000 row).

| w = 0.8 | δ = 0.3 | 0.4 | 0.5 | 0.6 | 0.65 |
|---|---|---|---|---|---|
| mixing M, 2D / 3D | 0.00 / 0.02 | 0.13 / 0.16 | 0.63 / 0.59 | 0.78 / 0.89 | 0.81 / 0.90 |
| polarization Φ, 2D / 3D | 0.95 / 0.93 | 0.96 / 0.93 | 0.97 / 0.94 | 0.97 / 0.95 | 0.97 / 0.95 |
| share in largest contact cluster, 2D / 3D | 0.34 / 0.86 | 0.87 / 0.99 | 0.94 / 0.95 | 0.87 / 0.89 | 0.85 / 0.94 |
| longest/thinnest axis, 3D | 2.8 | 2.1 | 1.7 | 2.0 | 2.0 |
| thinnest axis vs heading, |cos|, 3D | 0.73 | 0.92 | 0.95 | 0.98 | 0.99 |

Findings
- The δ threshold for segregation is the same in 3D as in 2D: segregated below about 0.4, mixed above about 0.55.
- 3D holds together better: at δ = 0.3 the three kinds stay as touching single-kind lobes of one flock (86 % in one contact cluster),
  while in 2D they split into three separate subgroups that travel together (34 %). Same at w = 0.6 (99 % vs 33 %). At w = 0.9 the 3D flock stays more segregated (M 0.12 vs 0.32).
- Kinds sit side by side across the direction of travel, not front to back, in both dimensions (low |cos|), except mixed 2D flocks.
- 3D flocks are flattened along the heading: a disc or wall facing forward (thinnest axis along motion, ratio 2-2.8). Real starling flocks
  (Ballerini et al. 2008) are thin vertically, with velocity lying in the flock's plane; the model has no gravity or banking, so it lacks that.
- Polarization is slightly lower in 3D (0.93-0.95 vs 0.95-0.97): the same λ spreads headings over two angles instead of one.
- w = 1 (alignment only) fragments into about 6 clusters in both.
- δ cycle 0.4-0.5, period 200 s: mixing swings as in 2D; 3D stays in one cluster (98 %), 2D at 94 %.

Files: flock3d_vs_2d.png (all metrics), flock3d_snapshots.png and flock2d_snapshots.png (final states), flock3d_traces.png and
flock3d_cycle_traces.png (time series), d3_*.mp4 (3D videos, camera follows the flock), CSVs (per run, summary, time series).
Reproduce: `python experiments/flock3d_compare.py` (~50 min on 4 cores).
