# w sweep: flock shape against the heading weight w

`w` sets how a partner enters the perceptual state: heading with amplitude sqrt(w) (alignment), bearing with sqrt(1 - w) (attraction). Full model of the paper otherwise (N = 240, G = 3, 5000 steps, seeds 5000-5004). Measures are averaged over the last 1000 steps; "one group" uses the final step.

| w | polarization | flock shape | one group (C > 0.9, sight range) | one contact cluster (≥ 90 %) |
|---|---|---|---|---|
| 0.0 | 0.03 | three dense, round single-kind clumps, no common heading | 4/5 | 0/5 |
| 0.2 | 0.04 | same; clumps drift apart | 2/5 | 0/5 |
| 0.4 | 0.22 | same; clumps drift apart | 0/5 | 0/5 |
| 0.6 | 0.81 | three single-kind subgroups, most elongated (length/width ≈ 2), long along motion | 3/5 | 0/5 |
| 0.8 (paper) | 0.96 | three single-kind subgroups, elongated along motion, travelling together | 5/5 | 0/5 |
| 0.9 | 0.97 | kinds merge into one larger, looser, partly mixed flock | 5/5 | 3/5 |
| 0.95 | 0.98 | one loose flock, kinds in bands | 5/5 | 3/5 |
| 1.0 | 0.98 | no attraction: sparse, fully mixed, 4-8 fragments of ≥ 5 agents | 5/5 | 0/5 |

Files: `w_sweep_metrics.png`, `w_sweep_one_group.png`, `w_sweep_snapshots.png`, CSVs (per run, summary, one-group table, time series).
Reproduce: `python experiments/w_sweep.py` (~7 min on 4 cores).
