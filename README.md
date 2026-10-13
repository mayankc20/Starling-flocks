# Starling flocks

Extensions of the mixed-flock model of *Segregation with cohesion* (Chaurasiya, Agrawal, Mohanty, Behera) toward starling-like flock behaviour.

| Path | What it is |
|---|---|
| `model/flock_model.py` | Unchanged copy of `flock_model.py` from [Segregation-with-cohesion-in-mixed-flocks](https://github.com/mayankc20/Segregation-with-cohesion-in-mixed-flocks), the base model. |
| `experiments/w_sweep.py` | Sweep of `w` (weight of partners' headings against their bearings) and its effect on flock shape. |
| `model/flock3d.py` | The full model in three dimensions: headings on a sphere grid (von Mises-Fisher states), cone of view, start in a ball; `shape_nd()` measures flock shape in 2D or 3D. |
| `model/flight3d.py` | Flight constraints on the 3D model: banked turns, pitch-rate cap, speed that changes with climb (gravity). |
| `experiments/small_groups.py` | Groups of 10-50 in 2D, 3D and 3D with gravity: neighbour switching and spacing. |
| `experiments/flight_small.py` | Groups of 10-50 with flight constraints (free, gravity, banking + speed, quadrotor). |
| `experiments/flock3d_compare.py` | 3D against 2D at w = 0.8, δ = 0.3-0.65, a few values of w, and a δ cycle; snapshots, videos, metrics. |

## Run

```
pip install -r requirements.txt
QUICK=1 python experiments/w_sweep.py   # tiny check, seconds
python experiments/w_sweep.py           # 8 values of w x 5 seeds, a few minutes on 4 cores
```

Results go to `output/w_sweep/` (set `W_OUT` to change).

```
QUICK=1 python experiments/flock3d_compare.py   # tiny check, about 20 s
python experiments/flock3d_compare.py           # 2D and 3D runs, figures, videos; about an hour on 4 cores
```

Results go to `output/flock3d/` (set `F3_OUT` to change).
