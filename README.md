# Starling flocks

Extensions of the mixed-flock model of *Segregation with cohesion* (Chaurasiya, Agrawal, Mohanty, Behera) toward starling-like flock behaviour.

| Path | What it is |
|---|---|
| `model/flock_model.py` | Unchanged copy of `flock_model.py` from [Segregation-with-cohesion-in-mixed-flocks](https://github.com/mayankc20/Segregation-with-cohesion-in-mixed-flocks), the base model. |
| `experiments/w_sweep.py` | Sweep of `w` (weight of partners' headings against their bearings) and its effect on flock shape. |

## Run

```
pip install -r requirements.txt
QUICK=1 python experiments/w_sweep.py   # tiny check, seconds
python experiments/w_sweep.py           # 8 values of w x 5 seeds, a few minutes on 4 cores
```

Results go to `output/w_sweep/` (set `W_OUT` to change).
