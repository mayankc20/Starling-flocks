"""Extensions of the base model (flock_model.py, kept unchanged) toward starling-like behaviour.

step_ext() is flock_model.step() for the Born rule (full model: relational partners, phase pi), with:
  * an optional extra term in the perceptual state: direction extra_dir[i] with amplitude extra_amp[i]
    (0 = absent), e.g. an escape cue pointing away from a predator;
  * w, delta0 (through S) and lambda allowed per agent: p["w"] of shape (N, 1), p["lam"] of shape (N,);
  * mu = 0 gives no memory: the heading is drawn from the present percept only.
"""
import numpy as np
from scipy.spatial import cKDTree
import flock_model as fm


def step_ext(X, theta, mem, kind, S, rng, grid, p, extra_dir=None, extra_amp=None):
    dirs, cos_d, sin_d = grid
    N, n = len(theta), p["n"]

    # 1. See.
    dx = X[None, :, 0] - X[:, None, 0]
    dy = X[None, :, 1] - X[:, None, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    cos_half = np.cos(np.deg2rad(p["fov_deg"]) / 2)
    with np.errstate(invalid="ignore"):
        visible = (dist <= p["R"]) & (dx * np.cos(theta)[:, None] + dy * np.sin(theta)[:, None] >= cos_half * dist)

    # 2. n partners sampled from the view in proportion to similarity.
    rows = np.arange(N)[:, None]
    with np.errstate(divide="ignore"):
        score = np.where(visible, -rng.standard_exponential((N, N)) / S, -np.inf)
    partners = np.argpartition(-score, n - 1, axis=1)[:, :n]
    found = np.isfinite(np.take_along_axis(score, partners, 1))

    # 3. Directions and amplitudes: own heading, partners' headings, partners' bearings (minus sign for other kinds),
    #    and the extra cue.
    bearing = np.arctan2(dy[rows, partners], dx[rows, partners])
    w = p["w"]
    direction = np.c_[theta, theta[partners], bearing]
    amplitude = np.c_[np.ones(N), np.sqrt(w) * found, np.sqrt(1 - w) * found]
    other = kind[partners] != kind[:, None]
    amplitude[:, 1 + n:] *= np.where(other, -1.0, 1.0)
    if extra_dir is not None:
        direction = np.c_[direction, extra_dir]
        amplitude = np.c_[amplitude, extra_amp]
    amplitude /= np.linalg.norm(amplitude, axis=1, keepdims=True)

    # 4. Born rule: amplitudes added, squared; heading drawn.
    half = 0.5 * np.broadcast_to(np.asarray(p["lam"], float), (N,))[:, None, None]
    states = np.stack([np.cos(direction), np.sin(direction)], axis=2) @ np.stack([cos_d, sin_d])
    states = np.exp(half * (states - 1))
    percept = np.matmul(amplitude[:, None, :], states)[:, 0, :]
    percept /= np.linalg.norm(percept, axis=1, keepdims=True)
    mem = percept if mem is None else np.sqrt(p["mu"]) * mem + np.sqrt(1 - p["mu"]) * percept
    mem /= np.linalg.norm(mem, axis=1, keepdims=True)
    prob = np.abs(mem) ** 2
    cdf = np.cumsum(prob, axis=1)
    pick = np.minimum((cdf < rng.random((N, 1)) * cdf[:, -1:]).sum(1), len(dirs) - 1)
    theta = (dirs[pick] + (dirs[1] - dirs[0]) * (rng.random(N) - 0.5)) % (2 * np.pi)

    # 5. Move, then remove overlaps.
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
