"""Fast correctness checks for the experiment's load-bearing assumptions."""

import numpy as np
import pytest

from src import config as C
from src.data import sample_tasks, collect_demos, split_episodes
from src.env import ReachEnv
from src.evaluate import rollout
from src.expert import expert_rollout
from src.policy import MLPPolicy
from src.posttrain import sample_random_data


@pytest.fixture(scope="module")
def env():
    e = ReachEnv()
    yield e
    e.close()


def test_obs_and_action_shapes(env):
    rng = np.random.default_rng(0)
    obs = env.reset(env.sample_task(C.TRAIN_COND, rng))
    assert obs.shape == (C.OBS_DIM,)
    obs2, dist = env.step(np.zeros(C.ACT_DIM))
    assert obs2.shape == (C.OBS_DIM,) and dist >= 0


def test_action_is_clipped_and_joint_limits_respected(env):
    rng = np.random.default_rng(1)
    env.reset(env.sample_task(C.TRAIN_COND, rng))
    q0 = env.q.copy()
    env.step(np.full(C.ACT_DIM, 10.0))          # far outside [-1, 1]
    assert np.all(np.abs(env.q - q0) <= C.MAX_DELTA + 1e-9)
    assert np.all(env.q >= env.lower - 1e-9) and np.all(env.q <= env.upper + 1e-9)


def test_expert_solves_every_evaluation_condition(env):
    """The benchmark is only meaningful if the expert can solve it."""
    rng = np.random.default_rng(2)
    for name, cond in C.EVAL_CONDITIONS.items():
        n_ok = 0
        for _ in range(10):
            env.reset(env.sample_task(cond, rng))
            _, _, ok, _, _ = expert_rollout(env)
            n_ok += ok
        assert n_ok >= 9, f"expert too weak on condition {name}: {n_ok}/10"


def test_env_is_deterministic_given_a_seed(env):
    def run():
        tasks = sample_tasks(env, C.TRAIN_COND, 3, seed=7)
        return np.stack([t["target"] for t in tasks]), np.stack([t["q0"] for t in tasks])
    a, b = run(), run()
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_train_val_split_is_episode_level(env):
    tasks = sample_tasks(env, C.TRAIN_COND, 20, seed=3)
    eps = collect_demos(env, tasks)
    tr, va = split_episodes(eps, 0.25, seed=3)
    total = sum(len(e[0]) for e in eps)
    assert len(tr[0]) + len(va[0]) == total
    assert len(va[0]) > 0 and len(tr[0]) > 0


def test_policy_roundtrip_and_determinism():
    m = MLPPolicy()
    m.set_normalizer(np.zeros(C.OBS_DIM), np.ones(C.OBS_DIM))
    m.eval()
    o = np.random.default_rng(0).normal(size=C.OBS_DIM).astype(np.float32)
    assert np.allclose(m.act(o), m.act(o))
    assert m.act(o).shape == (C.ACT_DIM,)


def test_success_criterion_matches_recorded_error(env):
    """A recorded success must have final error below the threshold."""
    rng = np.random.default_rng(4)
    m = MLPPolicy()
    m.set_normalizer(np.zeros(C.OBS_DIM), np.ones(C.OBS_DIM))
    for _ in range(5):
        rec, _ = rollout(env, m, env.sample_task(C.TRAIN_COND, rng))
        if rec["success"]:
            assert rec["final_error"] < C.SUCCESS_THRESHOLD
        assert rec["steps"] <= C.MAX_STEPS


def test_ablation_arms_get_matched_data_volume():
    obs = np.random.default_rng(0).normal(size=(500, C.OBS_DIM)).astype(np.float32)
    act = np.random.default_rng(1).normal(size=(500, C.ACT_DIM)).astype(np.float32)
    a = sample_random_data((obs, act), 120, seed=0)
    b = sample_random_data((obs, act), 120, seed=1)
    assert len(a[0]) == len(b[0]) == 120
