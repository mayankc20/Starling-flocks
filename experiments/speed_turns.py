"""MIND-FLOCK drones with variable speed and sharp turns (model/mindflock_speed.py).

N = 30 drones, 3 kinds, 300 s at 0.1 s per step, measures over the last 150 s, 6 seeds per setting.
  speed   speed regime (fixed, shared steps to all, shared steps to 20 %, individual OU speeds) x speed rule
          (const, match, born); plus the gap gain k_gap of the match rule
  turns   turn commands every 20 s by 45-180 deg, given to 20 %, 50 % or all drones, as a short pulse on a weak goal;
          and a strong standing goal (no pulse) for comparison
  combo   speed steps + turns together, with and without 300 ms latency and medium sensing noise

    python experiments/speed_turns.py         # ~25 min on 4 cores
    QUICK=1 python experiments/speed_turns.py
Results: ST_OUT (default output/speed_turns): per-run and summary CSVs, figures, a video. Runs cached in ST_OUT/cache.
"""
import os, sys, json, hashlib, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import mindflock_speed as ms

QUICK = os.environ.get("QUICK", "0") == "1"
OUT = os.environ.get("ST_OUT", os.path.join(os.path.dirname(__file__), "..", "output", "speed_turns"))
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))
T, SEEDS = (3000, list(range(200, 206))) if not QUICK else (600, [200])
os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)

PULSE = dict(a_goal=0.1, a_pulse=2.0, tau_pulse=2.0, turn_every=20.0)
REGIMES = {"fixed": dict(regime="fixed"), "steps, all": dict(regime="steps"),
           "steps, 20 %": dict(regime="steps", frac_speed=0.2), "own speeds (OU)": dict(regime="ou")}
RULES = ["const", "match", "born"]


def settings():
    S = []
    for rg, c in REGIMES.items():
        for rule in RULES:
            S.append(("speed", f"{rg} | {rule}", dict(c, speed_rule=rule)))
    for kg in (0.0, 0.01):
        S.append(("speed", f"steps, all | match, k_gap {kg}", dict(regime="steps", speed_rule="match", k_gap=kg)))
    for fr in (0.2, 0.5, 1.0):
        for ang in (45, 90, 135, 180):
            S.append(("turns", f"{int(fr * 100)} % informed | {ang} deg", dict(PULSE, frac_turn=fr, turn_deg=ang)))
    S.append(("turns", "100 % informed | 90 deg | standing goal 1.0", dict(turn_every=20.0, frac_turn=1.0, turn_deg=90, a_goal=1.0)))
    man = dict(PULSE, frac_turn=1.0, turn_deg=90, regime="steps", speed_rule="match")
    S.append(("combo", "steps + 90 deg turns", man))
    S.append(("combo", "steps + 90 deg turns, latency 300 ms", dict(man, latency=3)))
    S.append(("combo", "steps + 90 deg turns, noise medium", dict(man, kappa_b=50.0, sigma_r=0.1, sigma_v=0.1)))
    S.append(("combo", "own speeds + 90 deg turns", dict(man, regime="ou")))
    S.append(("combo", "steps + 90 deg turns, const speed rule", dict(man, speed_rule="const")))
    return S


def one(part, label, ch, seed):
    key = hashlib.md5(json.dumps([ms.params(**ch), seed, T], sort_keys=True, default=str).encode()).hexdigest()
    f = os.path.join(OUT, "cache", key + ".pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    r = ms.run(seed, ms.params(**ch), T)
    out = dict(part=part, setting=label, seed=seed, **r["summary"])
    pickle.dump(out, open(f, "wb"))
    return out


if __name__ == "__main__":
    jobs = [(pt, lb, ch, s) for pt, lb, ch in settings() for s in SEEDS]
    rows = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    df["whole"] = df.C_R_min >= 0.99
    df["seg_whole"] = (df.seg >= 0.5) & df.whole
    df["no_coll"] = df.collisions_per_drone_min == 0
    df.to_csv(os.path.join(OUT, "per_run.csv"), index=False)
    cols = ["seg", "C_R_min", "whole", "seg_whole", "collisions_per_drone_min", "no_coll", "d_min", "speed", "speed_sd",
            "Phi", "speed_track_err", "speed_lag_s", "turn_lag_s", "turn_followed", "turn_C_R_min", "turn_coll"]
    order = {lb: i for i, (_, lb, _) in enumerate(settings())}
    sm = df.groupby(["part", "setting"])[[c for c in cols if c in df]].mean().reset_index()
    sm["order"] = sm.setting.map(order); sm = sm.sort_values("order").drop(columns="order")
    sm.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(sm.round(3).to_string(index=False))
