# Faster w/δ cycles, one step read as 0.1 s

Breathing schedule (w 0.72-0.96, δ 0.05-0.55, in phase) after 300 s of sorting, then 400 s of cycling; N = 240, 3 seeds.
Script: `experiments/fast_cycles.py`. The 10 s animation and the full time series (7 MB) are in the project files only.

| period | density swing (amplitude) | mixing swing | contact-cluster swing | density lag |
|---|---|---|---|---|
| 2 s | 0.007 | 0 | 0 | |
| 5 s | 0.03 | 0 | 0 | |
| 10 s | 0.06 | 0 | 0 | |
| 20 s | 0.10 | 0 | 0.01 | ~14 s |
| 50 s | 0.15 | 0.001 | 0.03 | ~30 s |
| 200 s | 0.27 | 0.006 | 0.27 | |

The flock filters out fast cycles. At 0.1 s per step an agent moves 5 body lengths/s and its perceptual memory (mu = 0.95)
spans ~2 s, so subgroups ~10 body lengths across need tens of seconds to merge or part. Faster starling-like changes need
faster agent dynamics (larger v0, shorter memory) or local rather than flock-wide parameter changes.
