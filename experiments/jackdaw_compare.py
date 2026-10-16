"""Label-free 2D statistics of the public jackdaw flocks (Ling et al., Dryad doi:10.5061/dryad.kb8js06),
in the horizontal plane (x, y; z is vertical), next to the single-kind 2D model.

Usage: python experiments/jackdaw_compare.py DataS1.txt OUT_FOLDER [MODEL_STATS_CSV]
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from flock_stats import frame_stats, persistence, MIN_BIRDS

BODY = 0.34          # jackdaw body length in metres (approximate)
STRIDE = 3           # use every 3rd frame (60 Hz data -> 20 Hz)


def read_flocks(path):
    rows, flock = [], None
    for line in open(path):
        if line.startswith("ID"):
            continue
        if "flock" in line:
            flock = int(line.strip("* \n").split()[-1]); continue
        rows.append([flock] + line.split())
    cols = ["flock", "id", "x", "y", "z", "t", "u", "v", "w", "ax", "ay", "az", "fwb"]
    df = pd.DataFrame(rows, columns=cols).astype(float)
    df[["flock", "id"]] = df[["flock", "id"]].astype(int)
    df["t"] = df["t"].round(4)
    return df


def flock_summary(g):
    times = np.sort(g["t"].unique())[::STRIDE]
    dt = float(np.median(np.diff(times)))
    rows, frames = [], []
    for t in times:
        f = g[g["t"] == t]
        if len(f) < MIN_BIRDS:
            continue
        X = f[["x", "y"]].to_numpy() / BODY
        V = f[["u", "v"]].to_numpy() / BODY
        frames.append((f["id"].to_numpy(), X))
        r = frame_stats(X, V)
        r["vz_share"] = float(np.median(np.abs(f["w"]) / np.linalg.norm(f[["u", "v", "w"]].to_numpy(), axis=1)))
        rows.append(r)
    s = pd.DataFrame(rows).median(numeric_only=True).to_dict()
    s.update(persistence(frames, dt))
    s["frames"] = len(rows)
    s["duration_s"] = float(times[-1] - times[0])
    s["wingbeat_hz"] = float(g["fwb"].median())
    s["speed_ms"] = float(np.median(np.hypot(g["u"], g["v"])))
    return s


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    df = read_flocks(src)
    res = []
    for k, g in df.groupby("flock"):
        s = flock_summary(g); s["label"] = f"jackdaw flock {k}"; res.append(s)
        print(s["label"], "done", flush=True)
    data = pd.DataFrame(res)
    data.to_csv(os.path.join(out, "jackdaw_stats.csv"), index=False)
    print(data.round(3).to_string())
