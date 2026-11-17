"""Failure-focused policy post-training and its matched random-data ablation.

Both arms are identical except for the SOURCE of the new transitions:
  targeted -> expert corrective demonstrations restarted from states the
              baseline policy actually visited on episodes it failed
  random   -> freshly sampled expert demonstrations from the ORIGINAL
              training distribution

Everything else is held constant: starting checkpoint, architecture,
observation normaliser, optimiser, learning rate, epochs, batch size, seed,
new-data budget and replay budget.
"""

import copy

import numpy as np

from . import config as C
from .expert import expert_rollout
from .train import fit


def mine_failure_states(env, policy, dev_pool, evaluate_pool_fn, seed):
    """Roll the baseline out on the DEV pool (disjoint from the test benchmark)
    and keep the states visited on failed episodes."""
    ep, state_log, summary = evaluate_pool_fn(env, policy, dev_pool,
                                              model_name="baseline_dev", seed=seed,
                                              record_states=True)
    return ep, state_log, summary


def build_corrective_data(env, dev_pool, state_log, budget, seed):
    """Restart the expert from states on failed policy rollouts (a DAgger-style
    relabelling) and collect expert state-action pairs until `budget` is hit."""
    rng = np.random.default_rng(seed)
    keys = list(state_log.keys())
    rng.shuffle(keys)
    obs_all, act_all, n_traj = [], [], 0
    cond_counts = {}
    total = 0
    for cond, i in keys:
        states = state_log[(cond, i)]
        if not states:
            continue
        target = dev_pool[cond][i]["target"]
        # restart from the beginning and from two points along the failed rollout
        picks = sorted({0, len(states) // 3, (2 * len(states)) // 3})
        for s in picks:
            env.reset_to(states[s], target)
            o, a, ok, dist, steps = expert_rollout(env)
            if len(o) == 0:
                continue
            obs_all.append(o)
            act_all.append(a)
            n_traj += 1
            total += len(o)
            cond_counts[cond] = cond_counts.get(cond, 0) + len(o)
        if total >= budget:
            break
    if not obs_all:
        raise RuntimeError("no corrective data could be generated")
    obs = np.concatenate(obs_all)[:budget]
    act = np.concatenate(act_all)[:budget]
    return obs, act, n_traj, cond_counts


def sample_random_data(train_xy, budget, seed):
    """Matched-volume control: transitions drawn at random from the original
    training distribution (a fresh expert demo pool)."""
    rng = np.random.default_rng(seed)
    obs, act = train_xy
    idx = rng.choice(len(obs), size=min(budget, len(obs)), replace=len(obs) < budget)
    return obs[idx], act[idx]


def posttrain(baseline, new_xy, replay_xy, seed, tag=""):
    """Fine-tune a COPY of the baseline on (new data + replay). Identical
    hyperparameters for every arm."""
    model = copy.deepcopy(baseline)
    obs = np.concatenate([new_xy[0], replay_xy[0]])
    act = np.concatenate([new_xy[1], replay_xy[1]])
    history, n_steps = fit(model, obs, act, C.FT_EPOCHS, C.FT_LR, C.FT_BATCH, seed)
    return model, history, n_steps, len(obs)
