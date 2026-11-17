"""Central experiment configuration.

Every number that defines the experiment lives here so the controlled
comparison (baseline vs targeted post-training vs random post-training)
is auditable in one place.
"""

# ---------------------------------------------------------------- environment
EE_LINK = 6                 # KUKA iiwa end-effector link index
MAX_STEPS = 60              # fixed episode length (all models, all conditions)
SUCCESS_THRESHOLD = 0.05    # metres; success if ||ee - target|| < this
MAX_DELTA = 0.05            # radians; max per-joint movement per step
OBS_DIM = 16                # 7 joint pos + 3 ee + 3 target + 3 (target - ee)
ACT_DIM = 7

# ------------------------------------------------- task condition definitions
# Targets are sampled in cylindrical coordinates about the robot base.
# `train` is the ONLY distribution the baseline policy ever sees.
TRAIN_COND = dict(r=(0.40, 0.65), az=(-60.0, 60.0), z=(0.25, 0.65), q0_sigma=0.10)

# Held-out benchmark. `train_dist` is in-distribution (control); the other four
# are distribution shifts never seen during baseline training.
EVAL_CONDITIONS = {
    "train_dist":      dict(r=(0.40, 0.65), az=(-60.0, 60.0),  z=(0.25, 0.65), q0_sigma=0.10),
    "far_radius":      dict(r=(0.65, 0.78), az=(-60.0, 60.0),  z=(0.25, 0.65), q0_sigma=0.10),
    "low_height":      dict(r=(0.40, 0.65), az=(-60.0, 60.0),  z=(0.08, 0.25), q0_sigma=0.10),
    "wide_azimuth":    dict(r=(0.40, 0.65), az=(60.0, 115.0),  z=(0.25, 0.65), q0_sigma=0.10, mirror_az=True),
    "perturbed_start": dict(r=(0.40, 0.65), az=(-60.0, 60.0),  z=(0.25, 0.65), q0_sigma=0.60),
}
IN_DIST_CONDITION = "train_dist"
HELD_OUT_CONDITIONS = ["far_radius", "low_height", "wide_azimuth", "perturbed_start"]

# ------------------------------------------------------------------- datasets
N_TRAIN_DEMOS = 300         # expert demonstration episodes (train distribution)
VAL_FRACTION = 0.15         # episode-level train/val split
EVAL_EPISODES_PER_COND = 200
DEV_EPISODES_PER_COND = 200 # disjoint failure-mining pool (NEVER the test set)

# Post-training data budget. Both arms of the ablation get exactly this many
# NEW transitions plus exactly this many REPLAY transitions from the original
# training split -> matched data volume and matched gradient steps.
N_POSTTRAIN_NEW = 800
N_POSTTRAIN_REPLAY = 800

# -------------------------------------------------------------------- policy
HIDDEN = (128, 128)

# ------------------------------------------------------------------ training
BC_EPOCHS = 60
BC_LR = 1e-3
BC_BATCH = 256
FT_EPOCHS = 30              # identical for both post-training arms
FT_LR = 3e-4                # identical for both post-training arms
FT_BATCH = 256

# --------------------------------------------------------------------- seeds
DATA_SEED = 100             # expert demonstration sampling
BENCHMARK_SEED = 1234       # held-out test benchmark (FIXED for every model)
DEV_SEED = 4321             # failure-mining pool (disjoint from benchmark)
TRAIN_SEEDS = [0, 1, 2, 3, 4]     # independent training seeds; results reported as mean +/- std
