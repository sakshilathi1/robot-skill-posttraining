"""Kinematic reaching environment for a 7-DoF KUKA iiwa arm in PyBullet.

The arm is position-controlled at the kinematic level: the action is a
normalised joint delta and the next joint configuration is
    q_{t+1} = clip(q_t + MAX_DELTA * clip(a_t, -1, 1), joint_limits).
PyBullet supplies the robot model, forward kinematics and the IK solver used
by the expert. Physics integration is deliberately not simulated -- this keeps
rollouts fast and makes the experiment about policy learning, not dynamics.
"""

import numpy as np
import pybullet as p
import pybullet_data

from . import config as C


def _wrap_condition(cond, rng):
    """Sample (radius, azimuth_deg, height) from a condition spec."""
    r = rng.uniform(*cond["r"])
    az = rng.uniform(*cond["az"])
    if cond.get("mirror_az") and rng.random() < 0.5:
        az = -az
    z = rng.uniform(*cond["z"])
    return r, az, z


class ReachEnv:
    """Single-arm reaching environment. One PyBullet DIRECT client per env."""

    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True,
                                physicsClientId=self.cid)
        self.n_joints = 7
        lo, hi = [], []
        for j in range(self.n_joints):
            info = p.getJointInfo(self.robot, j, physicsClientId=self.cid)
            lo.append(info[8])
            hi.append(info[9])
        self.lower = np.array(lo, dtype=np.float64)
        self.upper = np.array(hi, dtype=np.float64)
        self.home = np.zeros(self.n_joints)
        self.home[1] = 0.6
        self.home[3] = -1.2
        self.home[5] = 1.0
        self.q = self.home.copy()
        self.target = np.zeros(3)

    # ------------------------------------------------------------- kinematics
    def set_q(self, q):
        self.q = np.clip(np.asarray(q, dtype=np.float64), self.lower, self.upper)
        for j in range(self.n_joints):
            p.resetJointState(self.robot, j, float(self.q[j]), targetVelocity=0.0,
                              physicsClientId=self.cid)

    def ee_pos(self):
        ls = p.getLinkState(self.robot, C.EE_LINK, computeForwardKinematics=True,
                            physicsClientId=self.cid)
        return np.array(ls[4], dtype=np.float64)

    # -------------------------------------------------------------- task api
    def sample_task(self, cond, rng):
        """Sample a task dict {q0, target} from a condition spec."""
        r, az, z = _wrap_condition(cond, rng)
        th = np.deg2rad(az)
        target = np.array([r * np.cos(th), r * np.sin(th), z])
        q0 = self.home + rng.normal(0.0, cond["q0_sigma"], size=self.n_joints)
        q0 = np.clip(q0, self.lower, self.upper)
        return {"q0": q0, "target": target}

    def reset(self, task):
        self.target = np.asarray(task["target"], dtype=np.float64)
        self.set_q(task["q0"])
        return self.obs()

    def reset_to(self, q, target):
        """Reset directly to an arbitrary (q, target) -- used to restart the
        expert from a state the policy actually visited."""
        self.target = np.asarray(target, dtype=np.float64)
        self.set_q(q)
        return self.obs()

    def obs(self):
        ee = self.ee_pos()
        return np.concatenate([self.q, ee, self.target, self.target - ee]).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.set_q(self.q + C.MAX_DELTA * a)
        ee = self.ee_pos()
        dist = float(np.linalg.norm(self.target - ee))
        return self.obs(), dist

    def dist(self):
        return float(np.linalg.norm(self.target - self.ee_pos()))

    def close(self):
        p.disconnect(physicsClientId=self.cid)
