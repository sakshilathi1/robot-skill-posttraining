# Resume Results

## Project

Robot Skill Post-Training & Evaluation

## Environment

Simulator: PyBullet (DIRECT mode, kinematic position control)
Robot: 7-DoF KUKA iiwa arm (`kuka_iiwa/model.urdf`)
Task: reach a 3-D Cartesian target; success = end-effector within 0.05 m of the target within 60 steps

## Model

Policy: MLP 16-128-128-7 (PyTorch, CPU), state in / joint-delta out
Training method: behavioural cloning from an analytic IK expert controller
Post-training method: failure-focused fine-tuning on expert corrective demonstrations restarted from states visited on failed policy rollouts

## Dataset

Training demonstrations: 300 expert episodes (2075 train / 357 val transitions, episode-level split)
Corrective demonstrations: 800 transitions (96 expert restarts on failed states)
Evaluation episodes: 1000 per seed (200 x 5 conditions), 5000 baseline episodes over 5 seeds

## Results

Baseline success: 58.2% (+/- 2.7 pp over 5 seeds)
Targeted post-training success: 79.4% (+/- 3.2 pp)
Random post-training success: 66.5% (+/- 5.4 pp)

Baseline held-out success: 48.0% (+/- 3.4 pp)
Targeted held-out success: 74.3% (+/- 3.9 pp)
Random held-out success: 58.1% (+/- 6.7 pp)

Baseline mean final error: 0.1383 m
Targeted mean final error: 0.0642 m
Random mean final error: 0.1063 m

## Failure Analysis

Dominant failure condition: `wide_azimuth` (baseline success 27.1%, 729 failures of 1000 episodes)

Baseline failure count: 2088 of 5000 benchmark episodes across 5 seeds

Most important observed failure pattern:

The baseline behaves well in-distribution (99.0%) but degrades under distribution shift (48.0% on shifted conditions). Failures concentrate in `wide_azimuth`; the full per-condition breakdown is in `results/failure_analysis.csv`.

Failure count by condition:

| condition | episodes | failures | baseline success |
| --- | ---: | ---: | ---: |
| wide_azimuth | 1000 | 729 | 27.1% |
| perturbed_start | 1000 | 618 | 38.2% |
| far_radius | 1000 | 513 | 48.7% |
| low_height | 1000 | 218 | 78.2% |
| train_dist | 1000 | 10 | 99.0% |

## Ablation

Targeted vs random (overall): +12.9 pp
Targeted vs random (held-out): +16.2 pp

Paired per-seed held-out deltas over 5 seeds (same seed, same benchmark, same starting checkpoint):

- targeted - baseline: +26.3 pp (std 5.9), targeted wins on 5/5 seeds
- random - baseline: +10.0 pp (std 5.1), random wins on 5/5 seeds
- targeted - random: +16.2 pp (std 7.2), targeted wins on 5/5 seeds

Where the corrective data actually came from (mean transitions per seed) - the mining procedure allocated its budget to the failing conditions without being told to:

| condition | corrective transitions | share |
| --- | ---: | ---: |
| wide_azimuth | 425 | 51.9% |
| perturbed_start | 238 | 29.0% |
| far_radius | 104 | 12.7% |
| low_height | 52 | 6.3% |

Both arms started from the same baseline checkpoint and used matched data volume (1600 transitions), identical optimiser (Adam), lr=0.0003, 30 epochs, 210 gradient steps and the same frozen observation normaliser. The only difference is the source of the new data.

## Strongest Findings

1. The baseline generalisation gap is real and measured: 99.0% success in-distribution vs 48.0% on shifted held-out conditions, against a 100.0% expert ceiling on the same benchmark.
2. Failures are not uniform: `wide_azimuth` accounts for the lowest baseline success (27.1%), and per-condition slicing localises the weakness rather than reporting a single aggregate score.
3. Targeted post-training improved held-out success by +26.3 pp (48.0% -> 74.3%) using only 800 corrective transitions.
4. The controlled ablation attributes the gain to targeting specifically: targeted beat matched-volume random data by +16.2 pp held-out.
5. The mechanism is visible in the data budget: failure mining was given a flat budget and no hint about which conditions matter, yet spent 52% of it on `wide_azimuth` - exactly the worst condition - and that condition improved the most (+51.8 pp).
6. Not everything was fixed: `perturbed_start` remains the weakest condition after targeted post-training (46.5%, only +8.3 pp over baseline), so start-state perturbation is a residual failure mode that corrective data did not resolve.

## Resume-Ready Bullets

- Improved held-out task success 48.0% -> 74.3% (+26.3 pp) by post-training a PyTorch robot-reaching policy on 800 failure-focused corrective demonstrations identified through benchmark failure analysis.
- Isolated `wide_azimuth` as the dominant source of policy failure (27.1% success vs 99.0% in-distribution) by slicing 5,000 benchmark evaluation episodes across 5 controlled task conditions (4 of them held-out distribution shifts).
- Ran a controlled ablation with matched data volume and matched gradient steps showing a 16.2 pp held-out advantage for targeted over randomly-sampled post-training data, isolating data targeting from data quantity.
- Built a single-command, seed-controlled evaluation pipeline (PyBullet + PyTorch) with an expert-verified 100.0%-solvable held-out benchmark and a failure-mining pool disjoint from the test set, reporting results over 5 training seeds.
- Reduced mean final end-effector error from 0.1383 m to 0.0642 m across 5,000 evaluation episodes while holding the success threshold, episode budget, model architecture and evaluation seeds fixed.
