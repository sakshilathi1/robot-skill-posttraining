"""Policy rollout and evaluation on a fixed task pool.

Success is defined once, here, and never changed:
    an episode SUCCEEDS if ||ee - target|| < SUCCESS_THRESHOLD (0.05 m)
    at any step within MAX_STEPS (60) steps.
`final_error` is the end-effector distance at episode termination.
"""

import numpy as np
import pandas as pd
import torch

from . import config as C


@torch.no_grad()
def rollout(env, policy, task, record_states=False):
    """Run `policy` on `task`. Returns a record dict (and visited states)."""
    obs = env.reset(task)
    dist = env.dist()
    states = []
    success = dist < C.SUCCESS_THRESHOLD
    min_dist = dist
    steps = 0
    for _ in range(C.MAX_STEPS):
        if success:
            break
        if record_states:
            states.append(env.q.copy())
        a = policy.act(obs)
        obs, dist = env.step(a)
        steps += 1
        min_dist = min(min_dist, dist)
        if dist < C.SUCCESS_THRESHOLD:
            success = True
    rec = {
        "success": bool(success),
        "final_error": float(dist),
        "min_error": float(min_dist),
        "steps": steps,
        "target_dist_from_base": float(np.linalg.norm(task["target"])),
        "target_z": float(task["target"][2]),
        "target_azimuth_deg": float(np.rad2deg(np.arctan2(task["target"][1], task["target"][0]))),
    }
    return (rec, states) if record_states else (rec, None)


def evaluate_pool(env, policy, pool, model_name="", seed=None, record_states=False):
    """Evaluate a policy over every condition in a task pool.
    Returns (episode-level DataFrame, per-condition summary DataFrame)."""
    rows, state_log = [], {}
    for cond, tasks in pool.items():
        for i, t in enumerate(tasks):
            rec, states = rollout(env, policy, t, record_states=record_states)
            rec.update(condition=cond, episode=i, model=model_name, seed=seed)
            rows.append(rec)
            if record_states and not rec["success"]:
                state_log[(cond, i)] = states
    ep = pd.DataFrame(rows)
    summary = (ep.groupby("condition")
                 .agg(episodes=("success", "size"),
                      successes=("success", "sum"),
                      success_rate=("success", "mean"),
                      mean_final_error=("final_error", "mean"),
                      mean_steps=("steps", "mean"))
                 .reset_index())
    summary["failures"] = summary["episodes"] - summary["successes"]
    summary["model"] = model_name
    summary["seed"] = seed
    return ep, (state_log if record_states else None), summary


def headline_metrics(ep):
    """Overall / in-distribution / held-out aggregates from an episode table."""
    held = ep[ep["condition"].isin(C.HELD_OUT_CONDITIONS)]
    indist = ep[ep["condition"] == C.IN_DIST_CONDITION]
    return {
        "overall_success": float(ep["success"].mean()),
        "overall_mean_error": float(ep["final_error"].mean()),
        "in_dist_success": float(indist["success"].mean()),
        "in_dist_mean_error": float(indist["final_error"].mean()),
        "held_out_success": float(held["success"].mean()),
        "held_out_mean_error": float(held["final_error"].mean()),
        "total_failures": int((~ep["success"]).sum()),
    }
