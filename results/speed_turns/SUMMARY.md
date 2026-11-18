# MIND-FLOCK drones with variable speed and sharp turns

Code: `model/mindflock_speed.py` (speed regimes, speed rules, turn commands; reuses `model/mindflock3d.py`),
`experiments/speed_turns.py` (sweep, ~25 min on 4 cores), `experiments/speed_turns_plots.py` (figures, video).
Same drones as before: N = 30, 3 kinds, n = 6, w = 0.5, δ = 0.3, r_safe = 5 BL, a_max = 0.2 BL/step², 180°/s turn cap,
one step = 0.1 s. 300 s per run, measures over the last 150 s, 6 seeds per setting (seeds 200–205).
"Whole" = never less than 99 % in the largest group within R; "segregated and whole" = seg ≥ 0.5 and whole.

## What was added

**Speed regimes** (what drones are asked to do): fixed cruise 0.5 BL/step (old model); shared speed command that jumps
0.5 → 0.8 → 0.25 BL/step every 30 s, given to all drones or to 20 % of them; each drone wandering in speed on its own
(Ornstein–Uhlenbeck, sd 0.15 BL/step = 30 % of cruise, 10 s time constant).

**Speed rules** (how a drone picks its speed):
- *const*: just fly the preferred speed (no coordination).
- *match*: relax toward the own preference, toward the partners' mean speed, speed up if partners are ahead
  (k_gap = 0.003 per BL), and brake for neighbours ahead inside r_safe **only when the gap is closing**.
- *born*: the model's own logic on a 1D speed grid: bumps at own speed, preference, partners' speeds, gap speed,
  braking speed; memory and Born-rule draw.

**Sharp turns**: every 20 s the goal direction is rotated 45–180° in the horizontal plane (alternating left/right),
given to 20 %, 50 % or all drones (mixed over kinds) as one more bump in their percept: weak standing amplitude 0.1 plus
a 2 s pulse of amplitude 2 after each command.

## Results

**Speed (fig_speed.png)**

| regime \| rule | seg | segregated & whole | collision-free runs | notes |
|---|---|---|---|---|
| fixed \| const (old model) | 0.75 | 6/6 | 4/6 | |
| own speeds (OU) \| const | 0.00 | **0/6** (16 % in largest group) | 0/6 | speed wander alone breaks the flock |
| own speeds (OU) \| match | 0.68 | **6/6** | **6/6** | speed spread 0.15 → 0.05 |
| own speeds (OU) \| born | 0.76 | 6/6 | 2/6 | |
| steps, all \| const | 0.82 | 5/6 | 3/6 | |
| steps, all \| match | 0.77 | **6/6** | **6/6** | reaches new level (±10 %) in 6.8 s, runs ~7 % below command |
| steps, all \| born | 0.78 | 6/6 | 2/6 | slow to track (22 s) |
| steps, 20 % \| const | 0.50 | 0/6 (splits, 80 %) | 2/6 | |
| steps, 20 % \| match | 0.70 | 6/6 | 6/6 | stays together but only partly follows (error 0.11 BL/step) |

- Speed coordination is needed as soon as drones differ in speed. Without it, individual speed wander breaks the flock
  (as in the earlier small-group flight study); with matching + closing-gap braking, everything stays segregated,
  whole and collision-free.
- Braking matters: without it (first test) collisions were 0.3 /drone/min and the flock split. Braking on any close
  neighbour ahead (not only when closing), together with k_gap = 0.01, made the flock run 10–15 % slower than commanded.
- Gap gain: k_gap = 0.01 pulls drones too tight (4/6 whole, collisions); 0 and 0.003 both work, 0.003 keeps more segregation.
- The Born rule on speed keeps the flock together and segregated, but its draw adds speed jitter that gives rare
  collisions (≈ 0.004–0.008 /drone/min). Deterministic matching is the safer choice for drones; the Born version is the
  one that is "the model all the way down" for the paper.

**Sharp turns (fig_turns.png)**

| informed | 45° | 90° | 135° | 180° |
|---|---|---|---|---|
| 100 % | 1.0 s, no splits, no collisions | 1.9 s, clean | 2.1 s, 0.17 collisions/turn | 2.0 s, 0.26 collisions/turn |
| 50 % | 1.6 s, clean | 3.3 s, 0.1 coll/turn | 6.3 s, splits (89 %) | 4.7 s, splits (45 %), 0.8 coll/turn |
| 20 % | 4.5 s | 8.6 s, only half the turns done within 15 s | 7.0 s, half | 7.9 s, half |

(time for the flock's mean heading to come within 20° of the new goal; segregation stayed 0.67–0.86 throughout.)
- Commanded to all drones as a short pulse, the flock turns up to 90° in ~2 s, staying whole and segregated.
- U-turns (135–180°) are fast but drones cross each other's paths: collisions appear. A half-informed flock is the
  worst case: the two halves pull apart and run into each other.
- With 20 % informed the flock stays whole but follows slowly (like the ~20 s re-alignment found earlier in 2D).
  Making turning partners "louder" (salience by turning rate) was tried and did not speed this up; it raised alignment
  and destroyed segregation (seg ≈ 0.1), so it was dropped.
- A strong standing goal to all drones (amplitude 1, no pulse) turns as fast (2.7 s) but costs segregation (0.55):
  the goal swamps the bearing terms. Short pulses are better.

**Speed changes and turns together (fig_traces.png, video_speed_turns.mp4)** — speed steps + 90° turns, matching rule:
segregated & whole 6/6, 4/6 collision-free (0.002 /drone/min, same level as the old fixed-speed model), turns in 2 s,
speed tracked in 7 s; segregation is lower (0.62) because the flock keeps being reshuffled. Medium sensing noise: fine
(seg 0.69, 6/6). 300 ms latency: collisions 0.018 /drone/min (no run collision-free), whole 6/6 but segregated in only 4/6 — latency remains the weak spot.
Own-speed wander + turns: whole and collision-free, but segregated in only 3/6 (seg 0.45).

## Next steps
- Banked turns for U-turns (turn radius instead of turning on the spot) or a speed drop during large turns, to remove
  the crossing collisions.
- Better information transfer when few drones are informed (e.g., goal relayed over comms, or speed-up of informed drones).
- Constant-velocity prediction for delayed states (latency).
- Compare against boids/Olfati-Saber with the same speed and turn commands.
