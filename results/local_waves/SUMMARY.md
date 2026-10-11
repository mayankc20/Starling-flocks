# Local waves of w and δ (0.1 s per step)

Per-agent w and δ in an excitable medium: rest (w 0.8, δ 0.3) -> excited (w 0.96, δ 0.55) for 1 s -> refractory 3 s.
Excitation passes to neighbours within 3 body lengths (prob. 0.6 per step); waves start at a random edge agent every ~3 s or ~10 s.
N = 240, 300 s sorting then 150 s of waves, 2 seeds; slow agents (5 bl/s, mu 0.95) and fast agents (10 bl/s, mu 0.8).
Script: `experiments/local_waves.py`. Animations (mp4) are in the project files only.

| agents | waves every | peak share excited | density sd | mixing M (mean / max) | contact clusters | polarization |
|---|---|---|---|---|---|---|
| fast | 3 s | 0.61 | 0.054 | 0.000 / 0.004 | 3.0 | 0.95 |
| fast | 10 s | 0.59 | 0.055 | 0.000 / 0.006 | 3.0 | 0.95 |
| slow | 3 s | 0.76 | 0.112 | 0.006 / 0.031 | 2.5 | 0.96 |
| slow | 10 s | 0.55 | 0.114 | 0.005 / 0.020 | 2.6 | 0.96 |

A wave crosses a subgroup in ~0.5 s and jumps to the next subgroup when they are within 3 body lengths, so a whole
flock is swept in ~1.5 s, a starling-like time. But a 1 s change of w and δ hardly changes where the agents are:
density, mixing and polarization show no clear response to the waves.
