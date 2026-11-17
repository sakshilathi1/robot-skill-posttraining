"""Analytic expert controller: PyBullet damped-least-squares IK + clipped
joint-space servoing. This is the demonstrator for behavioural cloning and the
source of all corrective (post-training) data.
"""

import numpy as np
import pybullet as p

from . import config as C


def expert_action(env):
    """Normalised joint delta that moves the arm toward the current target."""
    q_ik = p.calculateInverseKinematics(
        env.robot, C.EE_LINK, env.target.tolist(),
        maxNumIterations=100, residualThreshold=1e-4,
        physicsClientId=env.cid,
    )
    q_ik = np.array(q_ik[: env.n_joints], dtype=np.float64)
    delta = np.clip(q_ik - env.q, -C.MAX_DELTA, C.MAX_DELTA)
    return delta / C.MAX_DELTA


def expert_rollout(env, max_steps=C.MAX_STEPS):
    """Run the expert from the env's CURRENT state. Returns (obs, act, success,
    final_dist, steps). Terminates early on success."""
    obs_list, act_list = [], []
    dist = env.dist()
    success = dist < C.SUCCESS_THRESHOLD
    steps = 0
    for _ in range(max_steps):
        if success:
            break
        o = env.obs()
        a = expert_action(env)
        obs_list.append(o)
        act_list.append(a.astype(np.float32))
        _, dist = env.step(a)
        steps += 1
        if dist < C.SUCCESS_THRESHOLD:
            success = True
    obs = np.array(obs_list, dtype=np.float32) if obs_list else np.zeros((0, C.OBS_DIM), np.float32)
    act = np.array(act_list, dtype=np.float32) if act_list else np.zeros((0, C.ACT_DIM), np.float32)
    return obs, act, success, float(dist), steps
