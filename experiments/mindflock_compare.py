"""MIND-FLOCK in 3D against classical flocking rules, under drone limits (model/mindflock3d.py).

Experiments (N = 30 drones, 3 kinds, 300 s at 0.1 s per step, measures averaged over the last 150 s):
  main        every controller at the nominal drone settings
  delta       attention to other kinds delta0, MIND-FLOCK with and without the minus sign
  partners    number of attended partners n (fish attend to about 11-19) and the safety radius r_safe
  robust      one drone limit at a time, every controller: sensing noise, latency, link dropouts, drone failures,
              turn-rate cap, narrow field of view, short sensing range
  size        N = 12, 30, 60

    python experiments/mindflock_compare.py              # about an hour on 4 cores
    QUICK=1 python experiments/mindflock_compare.py      # a few minutes
    PARTS=main,robust python experiments/mindflock_compare.py

Results go to MF_OUT (default output/mindflock3d): per-run and summary CSVs, figures. Runs are cached in MF_OUT/cache.
"""
import os, sys, json, hashlib, pickle, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import mindflock3d as mf

QUICK = os.environ.get("QUICK", "0") == "1"
OUT = os.environ.get("MF_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "mindflock3d"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))
PARTS = os.environ.get("PARTS", "main,delta,partners,robust,size").split(",")
T, SEEDS, SEEDS_R = 3000, list(range(100, 110)), list(range(100, 106))
if QUICK:
    T, SEEDS, SEEDS_R = 300, [100, 101], [100]
os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)

NOISE = {"low": dict(kappa_b=200.0, sigma_r=0.05, sigma_v=0.05),     # ~4 deg bearing error, 5 % range and velocity
         "medium": dict(kappa_b=50.0, sigma_r=0.1, sigma_v=0.1),     # ~8 deg, 10 %
         "high": dict(kappa_b=10.0, sigma_r=0.2, sigma_v=0.2)}       # ~18 deg, 20 %
ROBUST = [("nominal", {})] + [(f"noise {k}", v) for k, v in NOISE.items()] \
    + [(f"latency {L * 100} ms", dict(latency=L)) for L in (1, 3, 5)] \
    + [(f"link loss {int(q * 100)} %", dict(p_drop=q)) for q in (0.2, 0.5, 0.8)] \
    + [(f"{int(f * 100)} % drones fail", dict(fail_frac=f)) for f in (0.2, 0.4)] \
    + [(f"turn cap {a} deg/s", dict(omega_max_deg_s=float(a))) for a in (45, 90)] \
    + [("view 120 deg", dict(fov_deg=120.0)), ("range 10 BL", dict(R=10.0))]
SUMMARY_KEYS = ["M", "seg", "C_R", "C_R_min", "C_contact", "kind_cohesion", "Phi", "nn", "radius", "speed",
                "collisions_per_drone_min", "collision_steps_frac", "d_min"]


def key(ctrl, seed, changes):
    p = dict(mf.BASE, **changes)
    s = json.dumps(dict(ctrl=ctrl, seed=seed, T=T, p={k: (str(v) if isinstance(v, float) and not np.isfinite(v) else v)
                                                     for k, v in p.items()}), sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def one(ctrl, seed, changes):
    path = os.path.join(OUT, "cache", key(ctrl, seed, changes) + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    r = mf.run(ctrl, seed, dict(mf.BASE, **changes), T)
    out = dict(summary=r["summary"], rec=r["rec"][::5].reset_index(drop=True))
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out


def batch(jobs, label):
    """jobs: list of (tag dict, ctrl, seed, changes). Returns per-run DataFrame."""
    t0 = time.time()
    order = sorted(range(len(jobs)), key=lambda i: -(jobs[i][1].startswith("mindflock")) * jobs[i][3].get("N", 30) ** 2)
    res = Parallel(n_jobs=N_JOBS)(delayed(one)(jobs[i][1], jobs[i][2], jobs[i][3]) for i in order)
    back = dict(zip(order, res))
    rows = [dict(**jobs[i][0], ctrl=jobs[i][1], seed=jobs[i][2], **back[i]["summary"]) for i in range(len(jobs))]
    print(f"[{label}] {len(jobs)} runs in {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.DataFrame(rows), {i: back[i]["rec"] for i in range(len(jobs))}


def summarize(df, by):
    g = df.groupby(by, sort=False)
    s = g[SUMMARY_KEYS].mean()
    s["runs"] = g.size()
    s["cohesive_share"] = g.apply(lambda d: float((d.C_R_min >= 0.99).mean()), include_groups=False)
    s["collision_free_share"] = g.apply(lambda d: float((d.collisions_per_drone_min == 0).mean()), include_groups=False)
    s["seg_and_cohesive"] = g.apply(lambda d: float(((d.seg >= 0.5) & (d.C_R_min >= 0.99)).mean()), include_groups=False)
    return s.reset_index()


def main():
    out = {}
    if "main" in PARTS:
        jobs = [(dict(part="main"), c, s, {}) for c in mf.CONTROLLERS for s in SEEDS]
        df, recs = batch(jobs, "main")
        out["main"] = df
        ts = []
        for i, (tag, c, s, ch) in enumerate(jobs):
            r = recs[i].copy(); r["ctrl"] = c; r["seed"] = s; ts.append(r)
        pd.concat(ts).to_csv(os.path.join(OUT, "main_timeseries.csv"), index=False)
    if "delta" in PARTS:
        jobs = [(dict(part="delta", delta0=d), c, s, dict(delta0=d)) for c in ("mindflock", "mindflock_plus")
                for d in (0.1, 0.2, 0.3, 0.4, 0.5, 0.67) for s in SEEDS]
        out["delta"], _ = batch(jobs, "delta")
    if "partners" in PARTS:
        jobs = [(dict(part="partners", n=n, r_safe=5.0), c, s, dict(n=n)) for c in ("mindflock", "mindflock_plus", "boids")
                for n in (6, 12, 18) for s in SEEDS_R]
        jobs += [(dict(part="partners", n=6, r_safe=r), c, s, dict(r_safe=r)) for c in ("mindflock", "boids")
                 for r in (2.0, 3.0, 7.0) for s in SEEDS_R]
        out["partners"], _ = batch(jobs, "partners")
    if "robust" in PARTS:
        jobs = [(dict(part="robust", condition=name), c, s, ch) for name, ch in ROBUST for c in mf.CONTROLLERS
                for s in SEEDS_R]
        out["robust"], _ = batch(jobs, "robust")
    if "size" in PARTS:
        jobs = [(dict(part="size", N=N), c, s, dict(N=N)) for N in (12, 30, 60)
                for c in ("mindflock", "mindflock_plus", "boids", "boids_signed", "olfati") for s in SEEDS_R]
        out["size"], _ = batch(jobs, "size")
    by = dict(main=["ctrl"], delta=["ctrl", "delta0"], partners=["ctrl", "n", "r_safe"], robust=["condition", "ctrl"],
              size=["ctrl", "N"])
    for k, df in out.items():
        df.to_csv(os.path.join(OUT, f"{k}_per_run.csv"), index=False)
        summarize(df, by[k]).round(4).to_csv(os.path.join(OUT, f"{k}_summary.csv"), index=False)
    print("done:", OUT)


if __name__ == "__main__":
    main()
