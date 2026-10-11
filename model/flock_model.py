"""The model of the paper, as a module: the same code as Sections 1-4 of paper.ipynb.

Used by the scripts in checks/. Settings can be changed with environment variables:
QUICK_TEST=1 (tiny runs), N_JOBS (cores), OUT (output folder, default "output").
"""
import os, json, time, hashlib, pickle, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import i0e, i1e
from scipy.stats import mannwhitneyu

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d2"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})
KIND_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100"]
PI = float(np.pi)

# A case says: which share of the partners are the nearest ones (the rest are sampled from the whole view),
# whether similarity is used in the sampling, the phase of the bearing of a partner of another kind,
# and the decision rule.
CASES = {"no_identity": dict(near=0.0, similarity=False, phase=0.0, rule="born"),
         "relational":  dict(near=0.0, similarity=True,  phase=0.0, rule="born"),
         "full":        dict(near=0.0, similarity=True,  phase=PI,  rule="born"),
         "nearest":     dict(near=1.0, similarity=False, phase=PI,  rule="born"),
         "nearest_no_identity": dict(near=1.0, similarity=False, phase=0.0, rule="born"),
         "half_nearest": dict(near=0.5, similarity=True,  phase=PI,  rule="born"),
         "decohered":   dict(near=0.0, similarity=True,  phase=PI,  rule="decohered"),
         "vector":      dict(near=0.0, similarity=True,  phase=PI,  rule="vector"),
         "vector_zero": dict(near=0.0, similarity=True, phase=0.0, rule="vector", other_bearing=0.0),   # other kinds' bearings ignored
         "peak":        dict(near=0.0, similarity=True,  phase=PI,  rule="peak"),       # always the most likely heading, no draw
         "peak_noise":  dict(near=0.0, similarity=True,  phase=PI,  rule="peak_noise"), # most likely heading + noise of fixed width
         "peak_width":  dict(near=0.0, similarity=True,  phase=PI,  rule="peak_width"), # most likely heading + noise as wide as the draw
         "peak_noid":   dict(near=0.0, similarity=False, phase=0.0, rule="peak"),        # the same two rules without identity
         "peak_noise_noid": dict(near=0.0, similarity=False, phase=0.0, rule="peak_noise")}
LABEL = {"no_identity": "no identity", "relational": "relational partners, no phase", "full": "full model",
         "nearest": "nearest partners", "nearest_no_identity": "nearest partners, no identity",
         "half_nearest": "half nearest, half relational",
         "decohered": "probabilities added (no interference)", "vector": "signed vector sum (classical)",
         "vector_zero": "vector sum, other kinds' bearings ignored",
         "peak": "most likely heading taken (no draw)",
         "peak_noise": "most likely heading + noise of fixed width",
         "peak_width": "most likely heading + noise as wide as the draw",
         "peak_noid": "most likely heading, no identity", "peak_noise_noid": "most likely heading + fixed noise, no identity"}
TABLE1 = ["no_identity", "relational", "full", "nearest", "nearest_no_identity", "half_nearest"]
SHOW = ["no_identity", "relational", "full"]            # drawn in the figures and the video
COLOR = {"no_identity": "#8a8a86", "relational": "#eda100", "full": "#4a3aa7"}
DASH = {"no_identity": (0, (4, 2)), "relational": "-", "full": "-"}
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
if __name__ == "__main__": print("numpy", np.__version__, "| pandas", pd.__version__, "| matplotlib", matplotlib.__version__, "| CPU cores:", os.cpu_count())

QUICK_TEST = os.environ.get("QUICK_TEST", "0") == "1"          # True: tiny runs to check that every cell works (about two minutes)
N_JOBS     = int(os.environ.get("N_JOBS", "4"))            # -1 = use all CPU cores; or a number, e.g. 32

BASE = dict(N=240, G=3, v0=0.5, r0=1.0, R=25.0, fov_deg=270.0, n=6, w=0.8, lam=50.0, mu=0.95, delta0=0.3,
            n_dir=120, density=0.25, passes=20)
T_RUN  = 5000               # steps per run
EVERY  = 10                 # record the measures every EVERY steps
C_ONE  = 0.9                # a run ends as one group if C > C_ONE
CONTACT = 3.0               # body lengths: agents closer than this are "in contact" (second, stricter grouping)

MAIN_SEEDS   = list(range(3000, 3030))             # Tables 2 and 3 of the paper (30 runs per case)
SWEEP_SEEDS  = list(range(2000, 2020))             # delta0 sweep (30 runs per value)
SENS_SEEDS   = list(range(2000, 2010))             # noise and sensitivity sweeps (20 runs per value)
SWEEP_DELTAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, "random"]   # "random": similarity ignored, delta0 = (G-1)/G

# Section 7: flock width / sight range.  For each flock size: the ratios tested and the number of runs.
LIMIT_RATIOS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
LIMIT_PLAN   = {240: (LIMIT_RATIOS, 20), 480: (LIMIT_RATIOS, 10), 960: ([2.0, 2.5, 3.0, 3.5, 4.5], 6)}
LIMIT_SEED0  = 4000

NOISE_LAMS = [5.0, 10.0, 20.0, 50.0, 100.0]        # Section 9
SENS = {"phase": [0.0, PI / 4, PI / 2, 3 * PI / 4, PI],      # Section 10; each value replaces the one in BASE / the full model
        "n":     [2, 3, 4, 6, 9, 12],
        "w":     [0.5, 0.65, 0.8, 0.9, 0.95],
        "mu":    [0.0, 0.8, 0.95, 0.99],
        "G":     [2, 3, 4, 5]}
LONG_T, LONG_SEEDS = 15000, list(range(3000, 3010))
BIG_N, BIG_RATIOS, BIG_T, BIG_SEEDS = 960, [3.0, 4.5], 15000, [4000, 4001]   # Section 11: large flock followed for longer
SHOW_SEED    = 3000
VIDEO_EVERY, VIDEO_FPS = 10, 25

if QUICK_TEST:
    BASE["N"] = 60; T_RUN = 150
    MAIN_SEEDS = MAIN_SEEDS[:3]; SWEEP_SEEDS = SWEEP_SEEDS[:2]; SENS_SEEDS = SENS_SEEDS[:2]
    LIMIT_PLAN = {60: ([2.0, 4.0], 2), 120: ([2.0, 4.0], 2)}
    NOISE_LAMS = [10.0, 50.0]; SENS = {"phase": [PI / 2, PI], "n": [3, 6], "w": [0.8], "mu": [0.0, 0.95], "G": [2, 3]}
    LONG_T = 300; LONG_SEEDS = LONG_SEEDS[:2]; VIDEO_EVERY = 5
    BIG_N, BIG_RATIOS, BIG_T, BIG_SEEDS = 120, [2.0], 300, [4000]

OUT = os.environ.get("OUT", "output")              # all tables, figures and saved runs go here
os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
G = BASE["G"]
print(f"N = {BASE['N']}, G = {G}, T = {T_RUN}, main runs per case = {len(MAIN_SEEDS)}, quick test = {QUICK_TEST}")

def similarity_s(G, delta0):
    # similarity of another kind, from the share of attention delta0 given to other kinds
    return delta0 / ((G - 1) * (1 - delta0))

def start(p, seed):
    # disordered, fully mixed start: positions in a disc, random headings, kinds shuffled
    rng = np.random.default_rng(seed)
    N = p["N"]
    radius = np.sqrt(N / (np.pi * p["density"]))
    r, a = radius * np.sqrt(rng.random(N)), 2 * np.pi * rng.random(N)
    X = np.c_[r * np.cos(a), r * np.sin(a)]
    theta = 2 * np.pi * rng.random(N)
    kind = rng.permutation(np.arange(N) % p["G"])
    return X, theta, kind, rng

def start_width(N, density=None):
    # diameter of the starting disc
    return 2 * np.sqrt(N / (np.pi * (BASE["density"] if density is None else density)))

def step(X, theta, mem, kind, S, case, rng, grid, p):
    # One time step for all agents. X: positions, theta: headings, mem: perceptual states.
    dirs, cos_d, sin_d = grid
    N, n = len(theta), p["n"]

    # 1. See. Only positions relative to the agent are used.
    dx = X[None, :, 0] - X[:, None, 0]
    dy = X[None, :, 1] - X[:, None, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    cos_half = np.cos(np.deg2rad(p["fov_deg"]) / 2)
    with np.errstate(invalid="ignore"):
        visible = (dist <= p["R"]) & (dx * np.cos(theta)[:, None] + dy * np.sin(theta)[:, None] >= cos_half * dist)

    # 2. Choose n partners: k_near nearest ones, the others sampled from the rest of the view.
    k_near = int(round(case["near"] * n))
    rows = np.arange(N)[:, None]
    chosen, found = [], []
    free = visible
    if k_near:
        score = np.where(visible, -dist, -np.inf)
        idx = np.argpartition(-score, k_near - 1, axis=1)[:, :k_near]
        chosen.append(idx); found.append(np.isfinite(np.take_along_axis(score, idx, 1)))
        free = visible.copy(); free[rows, idx] = False
    if n - k_near:
        with np.errstate(divide="ignore"):
            score = np.where(free, -rng.standard_exponential((N, N)) / S, -np.inf)
        idx = np.argpartition(-score, n - k_near - 1, axis=1)[:, :n - k_near]
        chosen.append(idx); found.append(np.isfinite(np.take_along_axis(score, idx, 1)))
    partners = np.concatenate(chosen, 1)
    found = np.concatenate(found, 1)                                   # False where fewer partners are available

    # 3. Perceived directions and their amplitudes: own heading, partners' headings, partners' bearings.
    bearing = np.arctan2(dy[rows, partners], dx[rows, partners])
    direction = np.c_[theta, theta[partners], bearing]
    amplitude = np.c_[np.ones(N), np.sqrt(p["w"]) * found, np.sqrt(1 - p["w"]) * found]
    other = kind[partners] != kind[:, None]
    phase = case["phase"]
    if "other_bearing" in case:                                        # control: weight of the bearings of other kinds
        amplitude[:, 1 + n:] *= np.where(other, case["other_bearing"], 1.0)
    if abs(phase - PI) < 1e-12:                                        # phase pi: a minus sign
        sign = np.where(other, -1.0, 1.0)
        on = case.get("phase_on", "bearings")                          # controls can put the phase on the headings instead
        if on in ("bearings", "both"):
            amplitude[:, 1 + n:] *= sign
        if on in ("headings", "both"):
            amplitude[:, 1:1 + n] *= sign
    elif phase != 0.0:                                                 # any other phase: complex amplitudes
        amplitude = amplitude.astype(complex)
        amplitude[:, 1 + n:] *= np.where(other, np.exp(1j * phase), 1.0)
    amplitude /= np.linalg.norm(amplitude, axis=1, keepdims=True)

    # 4. Decision.
    if case["rule"] == "vector":                                       # classical: signed vector sum + noise
        v = np.c_[(amplitude * np.cos(direction)).sum(1), (amplitude * np.sin(direction)).sum(1)]
        mem = v if mem is None else p["mu"] * mem + (1 - p["mu"]) * v
        theta = rng.vonmises(np.arctan2(mem[:, 1], mem[:, 0]), p["lam"]) % (2 * np.pi)
    else:
        half = 0.5 * p["lam"]
        states = np.stack([np.cos(direction), np.sin(direction)], axis=2) @ (half * np.stack([cos_d, sin_d]))
        states -= half
        np.exp(states, out=states)                                     # <theta|d> = exp[(lam/2)(cos(theta - d) - 1)]
        if case["rule"] in ("born", "peak", "peak_noise", "peak_width"):                                     # amplitudes added, then squared
            percept = np.matmul(amplitude[:, None, :], states)[:, 0, :]
            percept /= np.linalg.norm(percept, axis=1, keepdims=True)
            mem = percept if mem is None else np.sqrt(p["mu"]) * mem + np.sqrt(1 - p["mu"]) * percept
            mem /= np.linalg.norm(mem, axis=1, keepdims=True)
            prob = np.abs(mem) ** 2
        else:                                                          # "decohered": probabilities added
            now = np.matmul((np.abs(amplitude) ** 2)[:, None, :], states ** 2)[:, 0, :]
            now /= now.sum(1, keepdims=True)
            mem = now if mem is None else p["mu"] * mem + (1 - p["mu"]) * now
            prob = mem
        pick = 0
        if case["rule"] == "peak":                                     # always take the most likely heading
            pick = prob.argmax(1)
        elif case["rule"] == "peak_noise":                             # most likely heading + fixed von Mises noise
            pick = None
            theta = rng.vonmises(dirs[prob.argmax(1)], p["lam"]) % (2 * np.pi)
        elif case["rule"] == "peak_width":                             # most likely heading + noise as wide as |Psi|^2
            pick = None
            q = prob / prob.sum(1, keepdims=True)
            Rbar = np.clip(np.hypot(q @ np.cos(dirs), q @ np.sin(dirs)), 1e-6, 1 - 1e-9)
            kap = np.where(Rbar < 0.53, 2 * Rbar + Rbar**3 + 5 * Rbar**5 / 6,
                  np.where(Rbar < 0.85, -0.4 + 1.39 * Rbar + 0.43 / (1 - Rbar), 1 / (Rbar**3 - 4 * Rbar**2 + 3 * Rbar)))
            theta = rng.vonmises(dirs[prob.argmax(1)], kap) % (2 * np.pi)
        else:
            cdf = np.cumsum(prob, axis=1)
            pick = np.minimum((cdf < rng.random((N, 1)) * cdf[:, -1:]).sum(1), len(dirs) - 1)
        if pick is not None:
            theta = (dirs[pick] + (dirs[1] - dirs[0]) * (rng.random(N) - 0.5)) % (2 * np.pi)

    # 5. Move. Bodies closer than one body length are moved apart until no overlap is left.
    X = X + p["v0"] * np.c_[np.cos(theta), np.sin(theta)]
    for _ in range(p["passes"]):
        pairs = cKDTree(X).query_pairs(p["r0"], output_type="ndarray")
        if not len(pairs):
            break
        i, j = pairs.T
        d = X[j] - X[i]
        L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        push = 0.5 * (p["r0"] - L) * d / L
        np.add.at(X, i, -push)
        np.add.at(X, j, push)
    return X, theta, mem

def mixing_entropy(X, kind, G, m=10):
    m = min(m, len(X) - 1)
    _, near = cKDTree(X).query(X, m + 1)
    frac = np.stack([(kind[near[:, 1:]] == g).mean(1) for g in range(G)], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(frac > 0, frac * np.log(frac), 0).sum(1)
    return H.mean() / np.log(G)

def groups(X, link):
    pairs = cKDTree(X).query_pairs(link, output_type="ndarray")
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(X),) * 2)
    return connected_components(A, directed=False)[1]

def cohesion(X, link):
    return np.bincount(groups(X, link)).max() / len(X)

def perception_strength(mem, grid):
    prob = np.abs(mem) ** 2
    norm = prob.sum(1)
    return float(np.hypot((prob @ grid[1] / norm).mean(), (prob @ grid[2] / norm).mean()))

def run(case_name, seed, p, T, every=None, frame_every=0, keep_final=False):
    every = EVERY if every is None else every
    case = CASES[case_name]
    Gk = p["G"]
    dirs = np.linspace(0, 2 * np.pi, p["n_dir"], endpoint=False)
    grid = (dirs, np.cos(dirs), np.sin(dirs))
    X, theta, kind, rng = start(p, seed)
    s = similarity_s(Gk, p["delta0"]) if case["similarity"] else 1.0
    S = np.where(kind[:, None] == kind[None, :], 1.0, s)
    mem, frames = None, []
    rec = {"t": [], "M": [], "C": [], "Phi": [], "S": [], "gap_min": [], "gap_max": []}
    for t in range(T + 1):
        if t % every == 0 or t == T:
            rec["t"].append(t)
            rec["M"].append(mixing_entropy(X, kind, Gk))
            rec["C"].append(cohesion(X, p["R"]))
            rec["Phi"].append(np.hypot(np.cos(theta).mean(), np.sin(theta).mean()))
            rec["S"].append(perception_strength(mem, grid) if (mem is not None and case["rule"] == "born") else np.nan)
            cen = np.array([X[kind == g].mean(0) for g in range(Gk)])
            dd = np.linalg.norm(cen[:, None] - cen[None, :], axis=2)[np.triu_indices(Gk, 1)]
            rec["gap_min"].append(dd.min()); rec["gap_max"].append(dd.max())
        if frame_every and t % frame_every == 0:
            frames.append((X.copy(), theta.copy()))
        if t < T:
            X, theta, mem = step(X, theta, mem, kind, S, case, rng, grid, p)
    out = dict(rec={k: np.array(v) for k, v in rec.items()})
    out["M_random"] = float(np.mean([mixing_entropy(X, rng.permutation(kind), Gk) for _ in range(5)]))
    out["contact_groups"] = int(len(np.unique(groups(X, CONTACT * p["r0"]))))
    out["nn"] = float(np.median(cKDTree(X).query(X, 2)[0][:, 1]))
    if keep_final or frame_every:
        out["final"] = (X, theta, kind)
    if frame_every:
        out["frames"] = frames
    return out

def final(r, key, frac=0.2):
    y = r["rec"][key]
    return float(y[-max(1, int(round(len(y) * frac))):].mean())

def job_params(changes):
    p = dict(BASE); p.update(dict(changes)); return p

def job_key(case_name, seed, changes, T, keep_final):
    payload = dict(p=job_params(changes), case=CASES[case_name], seed=seed, T=T, every=EVERY, final=keep_final, contact=CONTACT)
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def _work(case_name, case, seed, params, T, every, keep_final, path):
    CASES[case_name] = case                                  # cases defined later in the notebook reach the workers this way
    out = run(case_name, seed, params, T, every=every, keep_final=keep_final)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return out

ALL_RUNS = []                                                # one line per run, saved as all_runs.csv in Section 17

def run_jobs(jobs, label, keep_final=False):
    # jobs: list of (case, seed, changes, T). Returns {job: result}.
    path = lambda j: os.path.join(OUT, "cache", job_key(*j, keep_final) + ".pkl")
    results, todo = {}, []
    for j in jobs:
        if os.path.exists(path(j)):
            with open(path(j), "rb") as fh:
                results[j] = pickle.load(fh)
        else:
            todo.append(j)
    print(f"[{label}] {len(jobs)} runs: {len(jobs) - len(todo)} loaded, {len(todo)} to compute", flush=True)
    todo.sort(key=lambda j: -job_params(j[2])["N"] ** 2 * j[3])           # slowest runs first
    t0 = time.time()
    args = lambda j: (j[0], CASES[j[0]], j[1], job_params(j[2]), j[3], EVERY, keep_final, path(j))
    try:
        from joblib import Parallel, delayed
        cores = os.cpu_count() if N_JOBS == -1 else N_JOBS
        for b in range(0, len(todo), 4 * cores):
            chunk = todo[b:b + 4 * cores]
            outs = Parallel(n_jobs=N_JOBS)(delayed(_work)(*args(j)) for j in chunk)
            results.update(dict(zip(chunk, outs)))
            print(f"   {min(b + 4 * cores, len(todo))}/{len(todo)} done  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    except ImportError:
        print("   joblib not found: running one after another")
        for k, j in enumerate(todo):
            results[j] = _work(*args(j))
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                print(f"   {k + 1}/{len(todo)} done  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    for j in jobs:
        r, p = results[j], job_params(j[2])
        ALL_RUNS.append({"section": label, "case": j[0], "seed": j[1], "changes": "; ".join(f"{k}={v}" for k, v in j[2]), "steps": j[3],
                         "N": p["N"], "R": p["R"], **{k: final(r, k) for k in ["M", "C", "Phi", "S", "gap_min", "gap_max"]},
                         "groups_by_contact": r["contact_groups"], "nearest_neighbour_distance": r["nn"]})
    return results

def wilson(k, n, z=1.96):
    # 95 % confidence interval of a proportion k/n
    q = k / n; d = 1 + z**2 / n; mid = (q + z**2 / (2 * n)) / d
    h = z * np.sqrt(q * (1 - q) / n + z**2 / (4 * n**2)) / d
    return max(0.0, mid - h), min(1.0, mid + h)

def summary(rs):
    v = {k: np.array([final(r, k) for r in rs]) for k in ["M", "C", "Phi", "S", "gap_min", "gap_max"]}
    v["n"] = len(rs); v["one"] = int((v["C"] > C_ONE).sum())
    v["contact"] = np.array([r["contact_groups"] for r in rs]); v["nn"] = np.array([r["nn"] for r in rs])
    return v

def pm(x):
    return f"{np.mean(x):.2f} ± {np.std(x):.2f}"

def one_group(v):
    lo, hi = wilson(v["one"], v["n"])
    return f"{v['one']}/{v['n']} [{100 * lo:.0f}-{100 * hi:.0f}%]"

def row(label, v):
    return {"case": label, "M": pm(v["M"]), "C": pm(v["C"]), "Phi": pm(v["Phi"]), "one group [95% CI]": one_group(v),
            "groups by contact (median)": int(np.median(v["contact"])), "nearest-neighbour distance": f"{np.median(v['nn']):.2f}"}

def save_fig(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.show()

def style(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
