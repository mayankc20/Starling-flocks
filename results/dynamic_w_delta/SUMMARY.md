# w and δ varying over time

Full model, N = 240, G = 3, λ = 50, 9000 steps (first 1000 at w = 0.8, δ = 0.3), 3 seeds per schedule (6000-6002).
Script: `experiments/dynamic_w_delta.py`. Animations (mp4, one seed) are kept in the project files, not in git.

| schedule | w(t) | δ(t) | density of main cluster | mixing M after sorting | subgroups |
|---|---|---|---|---|---|
| constant | 0.8 | 0.3 | settles ~0.8 | 0 | 3 separate, after ~3000 steps |
| breathing | 0.72-0.96, period 2000 | 0.05-0.55, in phase | pulses 0.35 <-> 1.0 each period | blips to ~0.1 | touch (1 contact cluster) and part (3) each period |
| deep | 0.75-1.0, period 4000 | 0.0-0.66, in phase | pulses 0.3 <-> 1.0 | up to ~0.2 at the loose phase | merge and part; at δ = 0 one subgroup left for good in 2 of 3 runs |
| drift (random walks) | 0.6-1.0 | 0.0-0.6 | wanders 0.4-1.0, irregular | ~0 | irregular merging and parting |

Polarization stays ~0.95 throughout; C (sight range) stays 1 except the deep runs at δ = 0.
