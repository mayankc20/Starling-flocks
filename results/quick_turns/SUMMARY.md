# Quick turns: no memory and an escape cue (0.1 s per step)

`model/starling_ext.py` adds an extra term to the perceptual state (direction + amplitude per agent); with the term off it
reproduces `flock_model.step` exactly. Script: `experiments/quick_turns.py`. μ = 0, 10 body lengths/s, N = 240,
300 s settling then 150 s with predator attacks, 3 seeds. Animations (mp4) are in the project files only.

| setting | attacks every | peak share reacting | flock turn rate, mean / max (°/s) | polarization | C (sight range) | mixing M |
|---|---|---|---|---|---|---|
| no predator | | 0 | 8 / 37 | 0.89 | 1.00 | 0.03 |
| self-spreading (only edge birds react), radial, cue² 6 | 5 s | 0.05 | 8 / 37 | 0.89 | 1.00 | 0.04 |
| imposed wave, radial dodge, cue² 6 | 5 s | 0.82 | 20 / 522 | 0.64 | 0.73 | 0.03 |
| imposed wave, radial dodge, cue² 2 | 10 s | 0.82 | 12 / 65 | 0.74 | 0.84 | 0.02 |
| imposed wave, shared dodge, cue² 2 | 10 s | 0.86 | 12 / 248 | 0.81 | 0.90 | 0.05 |
| imposed wave, shared dodge, cue² 6 | 10 s | 0.91 | 20 / 420 | 0.72 | 0.85 | 0.09 |

- Alignment alone does not carry a turn: when only the edge birds dodge, the rest of the flock does not notice.
- An imposed wave with a shared dodge gives whole-flock turns of 250-420°/s within ~1 s, but each attack disorders the
  flock and it needs ~20 s to re-align; with frequent or strong attacks it splits (C < 0.9).
- A radial dodge (each bird flees the predator point) scatters the flock instead of turning it.
