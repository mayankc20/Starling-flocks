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

**Collision avoidance without jumps.** Positions change only by integrating velocity (`X += V`). Safety is enforced
on the velocity, by a discrete control barrier function (CBF): the desired velocities are changed as little as
possible so that every pair closes at most `gamma · (d − d_safe)` per step (`gamma = 0.2`), and every agent approaches
the obstacle surface at most `gamma · (gap − d_safe/2)`. The distance can then only shrink geometrically toward
`d_safe` and never cross it. The filtered velocity then goes through the vehicle limits (turn ≤ 180°/s, speed
change ≤ 0.1 BL/step², i.e. about 3 m/s² for a 0.3 m drone, speed in [0, 2 v0]); filter and limits are applied
three times in a row so the result satisfies both. The soft escape bumps act first, smoothly, from r_avoid inward.
`d_safe` is a parameter (`--d_safe`, default 3 BL).
Agents are drawn as arrows of length `2/3 · d_safe` (2 BL) centred on the agent, so each arrow lies inside a ball of
radius 1/3 · d_safe and two arrows can never touch in 3D (they can overlap in the 2D projection of the video,
which is depth, not contact).

**Moving obstacle.** At t = 30 s a sphere of radius 9 BL appears 60 BL ahead of the flock and flies straight at it
(head-on) at 0.3 BL/step.

**Split/rejoin indicator.** Agents closer than 10 BL (1.7 · r_avoid) are linked; a group is a connected set of
at least 2 agents. The plot shows the number of groups and the share of the flock in the largest group.

## Result of the default run (N = 30, seed 1)

- Closest pair over the whole 120 s: 3.47 BL (d_safe 3, arrows 2 BL): arrows never touch.
- Closest approach to the obstacle surface: 2.14 BL (limit 1.5).
- The flock splits when it meets the obstacle (up to 3 groups for a moment, then 2 groups of about 60/40 %)
  and is one group again at t = 57 s, 27 s after the obstacle appeared.
- Speeds change smoothly; the plot shows mean and slowest agent. Agents slow down, and some briefly hover
  (speed 0), to keep the distance. That is fine for drones; for birds a minimum speed would be needed.

Over 24 runs (N = 10, 30, 50 with 6 seeds each; d_safe = 2 and 5 with 3 seeds each), the closest pair always
stayed above d_safe (lowest 3.29 BL at d_safe 3) and the obstacle gap above d_safe/2. The largest velocity change in
one step was 0.18 BL/step, set by the turn and acceleration limits. With the soft escape bumps switched off, the CBF
alone still held the distance (closest pair 3.03 to 3.05 BL in 4 runs).
The flock split in 9 of 24 runs and rejoined in all 9, 13 to 90 s after the obstacle appeared; in the other runs it
flowed around the obstacle as one connected group. Groups of 10 never split.
