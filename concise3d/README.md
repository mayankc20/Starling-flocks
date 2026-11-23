# Concise 3D MIND-FLOCK: open space, hard safe distance, moving obstacle

Two files, nothing else needed besides numpy, scipy, matplotlib and ffmpeg.

| File | What it is |
|---|---|
| `model.py` | The model (about 200 lines): MIND-FLOCK in 3D open space, a hard safe distance, one moving spherical obstacle. |
| `simulate.py` | Runs one flock through the obstacle and writes the video, a split/rejoin plot and a CSV. |

```
python simulate.py                              # N=30, d_safe=3 BL, seed 0, about 2 min
python simulate.py --d_safe 5 --N 20 --seed 2 --out out_d5
```

## Model in one paragraph

Units: body lengths (BL) and steps of 0.1 s; cruise speed 0.5 BL/step (5 BL/s). No walls, no periodic box.
Each agent sees neighbours within R = 40 BL, draws n = 6 partners in proportion to similarity
(δ = 0.3 attention to other kinds, 3 kinds), and builds a percept on the sphere from angular bumps:
own heading (1), partners' headings (√w), partners' bearings (√(1−w), minus sign for another kind), w = 0.5.
Two avoidance bumps are added: escape from neighbours closer than r_avoid = 2·d_safe, and
"steer around" the obstacle (outward normal + own heading's tangential part) within 8 BL of its surface.
The new heading is drawn by the Born rule, turning is capped at 180°/s. No percept memory (μ = 0).

**Hard collision avoidance.** After the move, a safety filter pushes apart any pair closer than
`d_safe` (and any agent closer than `d_safe/2` to the obstacle surface). The soft escape bumps act first, so the
filter only catches what they miss. `d_safe` is a parameter (`--d_safe`, default 3 BL).
Agents are drawn as arrows of length `2/3 · d_safe` (2 BL) centred on the agent, so each arrow lies inside a ball of
radius 1/3 · d_safe and two arrows can never touch in 3D (they can overlap in the 2D projection of the video,
which is depth, not contact).

**Moving obstacle.** At t = 30 s a sphere of radius 9 BL appears 60 BL ahead of the flock and flies straight at it
(head-on) at 0.3 BL/step.

**Split/rejoin indicator.** Agents closer than 10 BL (1.7 · r_avoid) are linked; a group is a connected set of
at least 2 agents. The plot shows the number of groups and the share of the flock in the largest group.

## Result of the default run (N = 30, seed 0)

- Closest pair over the whole 120 s: 3.003 BL (d_safe 3, arrows 2 BL): arrows never touch.
- Closest approach to the obstacle surface: 1.51 BL (limit 1.5).
- The flock splits when it meets the obstacle (up to 4 groups for a moment, then 2 groups with 80/20 %)
  and is one group again about 34 s after the obstacle appeared (t ≈ 64 s).

Over 18 further runs (N = 10, 30, 50; 6 seeds each), the safe distance held in every run, also at d_safe = 2 and 5.
The flock split in 8 of 18 runs and rejoined in all 8, 12 to 65 s after the obstacle appeared; in the other runs it
flowed around the obstacle as one connected group. Small groups (N = 10) split least (1 of 6).
