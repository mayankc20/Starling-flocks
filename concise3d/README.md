# Concise 3D MIND-FLOCK: open space, hard safe distance, moving obstacle

Two files, nothing else needed besides numpy, scipy, matplotlib and ffmpeg.

| File | What it is |
|---|---|
| `model.py` | The model (about 200 lines): MIND-FLOCK in 3D open space, a hard safe distance, one moving spherical obstacle. |
| `simulate.py` | Runs one flock through the obstacle and writes the video, a split/rejoin plot and a CSV. |

```
python simulate.py --seed 3                     # N=30, d_safe=5 BL, arrows 2 BL: the video in the project folder
python simulate.py --d_safe 3 --seed 1 --out out_d3
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
`d_safe` is a parameter (`--d_safe`, default 5 BL).
Agents are drawn as arrows of fixed length 2 BL (`--arrow`) centred on the agent; while the arrow is shorter than
`d_safe`, two arrows can never touch in 3D (they can overlap in the 2D projection of the video,
which is depth, not contact).

**Moving obstacle.** At t = 30 s a sphere of radius 9 BL appears 60 BL ahead of the flock and flies straight at it
(head-on) at 0.3 BL/step.

**Split/rejoin indicator.** Agents closer than 10 BL (1.7 · r_avoid) are linked; a group is a connected set of
at least 2 agents. The plot shows the number of groups and the share of the flock in the largest group.

## Video in the project folder (N = 30, d_safe = 5, arrows 2 BL, seed 3)

- Closest pair over 120 s: 6.06 BL (d_safe 5). Closest approach to the obstacle surface 3.75 BL (limit 2.5).
- The flock splits into 2 groups at the obstacle (t ≈ 42 s) and is one group again at t ≈ 56 s.
- At d_safe = 5 the flock is looser and the 9 BL obstacle splits it less often: 1 of 12 seeds (seed 3 is that one);
  in the other 11 it flowed around as one connected group. Closest pair over those 12 runs: 5.53 BL.

## Earlier default (d_safe = 3, seed 1)

- Closest pair 3.47 BL; splits into 2–3 groups and rejoins 27 s after the obstacle appears.
- Agents slow down, and some briefly hover (speed 0), to keep the distance. Fine for drones; birds would need a
  minimum speed.
Over 24 runs (N = 10, 30, 50 with 6 seeds each; d_safe = 2 and 5 with 3 seeds each), the closest pair always
stayed above d_safe (lowest 3.29 BL at d_safe 3) and the obstacle gap above d_safe/2. The largest velocity change in
one step was 0.18 BL/step, set by the turn and acceleration limits. With the soft escape bumps switched off, the CBF
alone still held the distance (closest pair 3.03 to 3.05 BL in 4 runs).
The flock split in 9 of 24 runs and rejoined in all 9, 13 to 90 s after the obstacle appeared; in the other runs it
flowed around the obstacle as one connected group. Groups of 10 never split.
