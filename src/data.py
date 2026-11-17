"""Dataset generation: expert demonstrations, the fixed held-out benchmark and
the disjoint failure-mining pool."""

import numpy as np

from . import config as C
from .expert import expert_rollout


def _solvable(env, task):
    """True if the analytic expert can solve this task. Every benchmark episode
    is expert-verified so that any policy failure is a POLICY failure, not an
    unreachable target."""
    env.reset(task)
    _, _, ok, _, _ = expert_rollout(env)
    return ok


def sample_tasks(env, cond, n, seed, expert_verified=True, max_tries=40):
    """Sample n tasks from a condition, optionally keeping only expert-solvable
    ones. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    tasks, tries = [], 0
    while len(tasks) < n and tries < n * max_tries:
        t = env.sample_task(cond, rng)
        tries += 1
        if (not expert_verified) or _solvable(env, t):
            tasks.append(t)
    if len(tasks) < n:
        raise RuntimeError(f"only found {len(tasks)}/{n} solvable tasks for condition {cond}")
    return tasks


def collect_demos(env, tasks):
    """Run the expert on each task; return per-episode (obs, act) arrays."""
    eps = []
    for t in tasks:
        env.reset(t)
        obs, act, ok, dist, steps = expert_rollout(env)
        if len(obs) > 0:
            eps.append((obs, act))
    return eps


def split_episodes(eps, val_fraction, seed):
    """Episode-level train/val split (no transition leaks across the split)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eps))
    n_val = int(round(len(eps) * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return _stack([eps[i] for i in train_idx]), _stack([eps[i] for i in val_idx])


def _stack(eps):
    if not eps:
        return np.zeros((0, C.OBS_DIM), np.float32), np.zeros((0, C.ACT_DIM), np.float32)
    return (np.concatenate([e[0] for e in eps]), np.concatenate([e[1] for e in eps]))


def build_task_pool(env, conditions, n_per_cond, seed):
    """Build a deterministic pool of expert-verified tasks, keyed by condition.
    Each condition gets its own derived seed so pools built with different base
    seeds share no episodes."""
    pool = {}
    for i, (name, cond) in enumerate(conditions.items()):
        pool[name] = sample_tasks(env, cond, n_per_cond, seed + 1000 * (i + 1))
    return pool


def save_pool(path, pool):
    flat = {}
    for name, tasks in pool.items():
        flat[f"{name}__q0"] = np.stack([t["q0"] for t in tasks])
        flat[f"{name}__target"] = np.stack([t["target"] for t in tasks])
    np.savez(path, **flat)


def load_pool(path):
    z = np.load(path)
    names = sorted({k.split("__")[0] for k in z.files})
    pool = {}
    for name in names:
        q0s, tgts = z[f"{name}__q0"], z[f"{name}__target"]
        pool[name] = [{"q0": q0s[i], "target": tgts[i]} for i in range(len(q0s))]
    return pool
