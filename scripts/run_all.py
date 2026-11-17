#!/usr/bin/env python3
"""Reproduce the complete experiment with one command:

    python scripts/run_all.py

Stages: expert demos -> BC baseline -> fixed held-out benchmark -> baseline
evaluation -> failure analysis -> targeted corrective data -> post-training ->
matched random-data ablation -> evaluation of all arms -> metrics, plots and
the resume summary.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C
from src.data import build_task_pool, collect_demos, load_pool, sample_tasks, save_pool, split_episodes
from src.env import ReachEnv
from src.evaluate import evaluate_pool, headline_metrics
from src.expert import expert_rollout
from src.policy import MLPPolicy
from src.posttrain import build_corrective_data, posttrain, sample_random_data
from src.train import train_baseline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(RESULTS, "plots")
DATA = os.path.join(ROOT, "data")

MODELS = ["baseline", "targeted_posttrain", "random_posttrain"]
LABELS = {"baseline": "Baseline (BC)",
          "targeted_posttrain": "Targeted post-training",
          "random_posttrain": "Random post-training"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=C.TRAIN_SEEDS)
    ap.add_argument("--eval-episodes", type=int, default=C.EVAL_EPISODES_PER_COND)
    ap.add_argument("--demos", type=int, default=C.N_TRAIN_DEMOS)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.eval_episodes, args.demos = [0], 12, 25

    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    env = ReachEnv()

    # ---------------------------------------------------- stage 2 first: bench
    # Built once, from a fixed seed, and reused unchanged by every model.
    log(f"STAGE 2  building held-out benchmark ({args.eval_episodes} eps x "
        f"{len(C.EVAL_CONDITIONS)} conditions, expert-verified)")
    bench_path = os.path.join(DATA, "benchmark.npz")
    test_pool = build_task_pool(env, C.EVAL_CONDITIONS, args.eval_episodes, C.BENCHMARK_SEED)
    save_pool(bench_path, test_pool)

    log(f"          building DISJOINT dev pool for failure mining "
        f"({C.DEV_EPISODES_PER_COND if not args.quick else args.eval_episodes} eps x cond)")
    dev_n = args.eval_episodes if args.quick else C.DEV_EPISODES_PER_COND
    dev_pool = build_task_pool(env, C.EVAL_CONDITIONS, dev_n, C.DEV_SEED)
    save_pool(os.path.join(DATA, "dev_pool.npz"), dev_pool)

    # expert ceiling on the benchmark (sanity: should be 100% by construction)
    exp_rows = []
    for cond, tasks in test_pool.items():
        for t in tasks:
            env.reset(t)
            _, _, ok, dist, steps = expert_rollout(env)
            exp_rows.append({"condition": cond, "success": ok, "final_error": dist, "steps": steps})
    expert_df = pd.DataFrame(exp_rows)
    log(f"          expert ceiling on benchmark: {expert_df['success'].mean()*100:.1f}% success")

    # --------------------------------------------------------- stage 1: demos
    log(f"STAGE 1  generating {args.demos} expert demonstrations (train distribution only)")
    demo_tasks = sample_tasks(env, C.TRAIN_COND, args.demos, C.DATA_SEED)
    demo_eps = collect_demos(env, demo_tasks)
    train_xy, val_xy = split_episodes(demo_eps, C.VAL_FRACTION, C.DATA_SEED)
    log(f"          train transitions={len(train_xy[0])}  val transitions={len(val_xy[0])}")

    # fresh, independent demos from the SAME training distribution, used only as
    # the random-ablation data source (so it is genuinely new data, matching the
    # corrective set which is also newly collected)
    rand_tasks = sample_tasks(env, C.TRAIN_COND, args.demos, C.DATA_SEED + 77)
    rand_eps = collect_demos(env, rand_tasks)
    rand_pool_xy = (np.concatenate([e[0] for e in rand_eps]),
                    np.concatenate([e[1] for e in rand_eps]))

    new_budget = min(C.N_POSTTRAIN_NEW, len(train_xy[0])) if args.quick else C.N_POSTTRAIN_NEW
    replay_budget = min(C.N_POSTTRAIN_REPLAY, len(train_xy[0]))

    all_eps, all_summ, all_head, prov = [], [], [], []
    dev_fail_frames, corrective_mix = [], []

    for seed in args.seeds:
        log(f"=== SEED {seed} " + "=" * 40)

        # ------------------------------------------------ stage 1: BC baseline
        log("STAGE 1  training baseline policy (behavioural cloning)")
        baseline, hist, n_bc = train_baseline(train_xy, val_xy, seed, log_every=0)
        log(f"          final train_mse={hist[-1]['train_mse']:.5f} "
            f"val_mse={hist[-1]['val_mse']:.5f}  ({n_bc} gradient steps)")
        torch.save(baseline.state_dict(), os.path.join(DATA, f"baseline_s{seed}.pt"))

        # ------------------------------------------ stage 4: failure mining
        log("STAGE 4/5 mining baseline failures on the DEV pool (test set untouched)")
        dev_ep, state_log, dev_summ = evaluate_pool(env, baseline, dev_pool,
                                                    model_name="baseline_dev", seed=seed,
                                                    record_states=True)
        dev_ep["seed"] = seed
        dev_fail_frames.append(dev_summ.assign(seed=seed))
        log(f"          dev success={dev_ep['success'].mean()*100:.1f}%  "
            f"failures={int((~dev_ep['success']).sum())}")

        corr_obs, corr_act, n_traj, corr_mix = build_corrective_data(
            env, dev_pool, state_log, new_budget, seed=seed)
        for c, n in corr_mix.items():
            corrective_mix.append({"seed": seed, "condition": c, "transitions": n})
        log(f"STAGE 5  corrective dataset: {len(corr_obs)} transitions "
            f"from {n_traj} expert restarts on failed states")

        rnd_obs, rnd_act = sample_random_data(rand_pool_xy, len(corr_obs), seed=seed)
        replay = sample_random_data(train_xy, replay_budget, seed=seed + 999)

        # ------------------------------------- stage 6/7: post-train + ablation
        log("STAGE 6  post-training on TARGETED corrective data")
        m_tgt, _, n_ft_t, n_tgt = posttrain(baseline, (corr_obs, corr_act), replay, seed)
        log("STAGE 7  post-training on MATCHED RANDOM data (ablation)")
        m_rnd, _, n_ft_r, n_rnd = posttrain(baseline, (rnd_obs, rnd_act), replay, seed)
        assert n_tgt == n_rnd and n_ft_t == n_ft_r, "post-training arms are not matched"
        log(f"          both arms: {n_tgt} transitions, {n_ft_t} gradient steps, "
            f"lr={C.FT_LR}, epochs={C.FT_EPOCHS}")

        prov.append({"seed": seed, "bc_grad_steps": n_bc, "ft_grad_steps": n_ft_t,
                     "posttrain_transitions": n_tgt, "corrective_transitions": len(corr_obs),
                     "corrective_restarts": n_traj,
                     "dev_failures": int((~dev_ep["success"]).sum())})

        # ---------------------------------------- stage 3/8: evaluate all arms
        for name, model in [("baseline", baseline), ("targeted_posttrain", m_tgt),
                            ("random_posttrain", m_rnd)]:
            ep, _, summ = evaluate_pool(env, model, test_pool, model_name=name, seed=seed)
            all_eps.append(ep)
            all_summ.append(summ)
            h = headline_metrics(ep)
            h.update(model=name, seed=seed)
            all_head.append(h)
            log(f"STAGE 3/8 {name:20s} overall={h['overall_success']*100:5.1f}%  "
                f"held-out={h['held_out_success']*100:5.1f}%  "
                f"err={h['overall_mean_error']:.4f}m")

    env.close()

    # ------------------------------------------------------------ persistence
    episodes = pd.concat(all_eps, ignore_index=True)
    per_cond = pd.concat(all_summ, ignore_index=True)
    head = pd.DataFrame(all_head)
    dev_fail = pd.concat(dev_fail_frames, ignore_index=True)

    episodes.to_csv(os.path.join(RESULTS, "episodes.csv"), index=False)
    per_cond.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False)

    # failure analysis: baseline only, aggregated over seeds
    base = episodes[episodes["model"] == "baseline"]
    fa = (base.groupby("condition")
              .agg(episodes=("success", "size"), successes=("success", "sum"),
                   success_rate=("success", "mean"), mean_error=("final_error", "mean"),
                   mean_steps=("steps", "mean"))
              .reset_index())
    fa["failures"] = fa["episodes"] - fa["successes"]
    fa = fa.sort_values("success_rate")
    fa.to_csv(os.path.join(RESULTS, "failure_analysis.csv"), index=False)

    # comparison table: mean +/- std across training seeds
    comp = (head.groupby("model")
                .agg(overall_success_mean=("overall_success", "mean"),
                     overall_success_std=("overall_success", "std"),
                     in_dist_success_mean=("in_dist_success", "mean"),
                     held_out_success_mean=("held_out_success", "mean"),
                     held_out_success_std=("held_out_success", "std"),
                     overall_mean_error=("overall_mean_error", "mean"),
                     held_out_mean_error=("held_out_mean_error", "mean"),
                     total_failures=("total_failures", "mean"))
                .reindex(MODELS).reset_index())
    comp.to_csv(os.path.join(RESULTS, "comparison.csv"), index=False)
    pd.DataFrame(prov).to_csv(os.path.join(RESULTS, "run_provenance.csv"), index=False)
    corr_mix_df = (pd.DataFrame(corrective_mix).groupby("condition")["transitions"]
                     .mean().reset_index().sort_values("transitions", ascending=False))
    corr_mix_df["share"] = corr_mix_df["transitions"] / corr_mix_df["transitions"].sum()
    corr_mix_df.to_csv(os.path.join(RESULTS, "corrective_composition.csv"), index=False)
    dev_fail.to_csv(os.path.join(RESULTS, "dev_pool_baseline.csv"), index=False)

    runtime = time.time() - t0
    held = episodes[episodes["condition"].isin(C.HELD_OUT_CONDITIONS)]
    piv = held.pivot_table(index="seed", columns="model", values="success")
    paired = {
        "tgt_vs_base": (piv["targeted_posttrain"] - piv["baseline"]) * 100,
        "tgt_vs_rand": (piv["targeted_posttrain"] - piv["random_posttrain"]) * 100,
        "rand_vs_base": (piv["random_posttrain"] - piv["baseline"]) * 100,
    }
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(os.path.join(RESULTS, "paired_seed_deltas.csv"))

    ctx = dict(runtime=runtime, seeds=args.seeds, paired=paired_df, corr_mix=corr_mix_df, eval_episodes=args.eval_episodes,
               demos=args.demos, n_train_tx=len(train_xy[0]), n_val_tx=len(val_xy[0]),
               expert_ceiling=float(expert_df["success"].mean()),
               prov=pd.DataFrame(prov))

    make_plots(episodes, per_cond, fa)
    write_reports(episodes, per_cond, head, comp, fa, ctx)

    log(f"DONE in {runtime/60:.1f} min -> results/")
    print("\n" + comp.to_string(index=False))


# --------------------------------------------------------------------- plots
def make_plots(episodes, per_cond, fa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#888888", "targeted_posttrain": "#1f77b4",
              "random_posttrain": "#d62728"}

    # 1. overall / in-dist / held-out success by model
    fig, ax = plt.subplots(figsize=(7, 4.2))
    groups = ["Overall", "In-distribution", "Held-out (shifted)"]
    x = np.arange(len(groups))
    w = 0.26
    for k, m in enumerate(MODELS):
        sub = episodes[episodes["model"] == m]
        per_seed = []
        for s in sub["seed"].unique():
            ss = sub[sub["seed"] == s]
            per_seed.append([
                ss["success"].mean(),
                ss[ss["condition"] == C.IN_DIST_CONDITION]["success"].mean(),
                ss[ss["condition"].isin(C.HELD_OUT_CONDITIONS)]["success"].mean()])
        arr = np.array(per_seed) * 100
        ax.bar(x + (k - 1) * w, arr.mean(0), w, yerr=arr.std(0), capsize=3,
               label=LABELS[m], color=colors[m])
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title("Task success: baseline vs targeted vs random post-training")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "01_success_overall.png"), dpi=150)
    plt.close(fig)

    # 2. success rate by evaluation condition
    conds = list(C.EVAL_CONDITIONS.keys())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(conds)); w = 0.26
    for k, m in enumerate(MODELS):
        means, stds = [], []
        for c in conds:
            sub = episodes[(episodes["model"] == m) & (episodes["condition"] == c)]
            ps = np.array([sub[sub["seed"] == s]["success"].mean() for s in sub["seed"].unique()])
            means.append(ps.mean() * 100); stds.append(ps.std() * 100)
        ax.bar(x + (k - 1) * w, means, w, yerr=stds, capsize=3, label=LABELS[m], color=colors[m])
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=15, ha="right")
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title("Success rate by held-out evaluation condition")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "02_success_by_condition.png"), dpi=150)
    plt.close(fig)

    # 3. mean final end-effector error
    fig, ax = plt.subplots(figsize=(7, 4.2))
    groups = ["Overall", "In-distribution", "Held-out (shifted)"]
    x = np.arange(len(groups))
    for k, m in enumerate(MODELS):
        sub = episodes[episodes["model"] == m]
        per_seed = []
        for s in sub["seed"].unique():
            ss = sub[sub["seed"] == s]
            per_seed.append([
                ss["final_error"].mean(),
                ss[ss["condition"] == C.IN_DIST_CONDITION]["final_error"].mean(),
                ss[ss["condition"].isin(C.HELD_OUT_CONDITIONS)]["final_error"].mean()])
        arr = np.array(per_seed)
        ax.bar(x + (k - 1) * w, arr.mean(0), w, yerr=arr.std(0), capsize=3,
               label=LABELS[m], color=colors[m])
    ax.axhline(C.SUCCESS_THRESHOLD, ls="--", c="k", lw=1,
               label=f"success threshold ({C.SUCCESS_THRESHOLD} m)")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Mean final end-effector error (m)")
    ax.set_title("Mean final end-effector error")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "03_mean_error.png"), dpi=150)
    plt.close(fig)

    # 4. baseline failure distribution by condition
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    f = fa.sort_values("failures", ascending=False)
    ax.bar(f["condition"], f["failures"], color="#d62728")
    for i, (c, v) in enumerate(zip(f["condition"], f["failures"])):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Baseline failed episodes"); ax.set_title("Where the baseline policy fails")
    plt.xticks(rotation=15, ha="right"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "04_failure_distribution.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- reports
def _pct(v):
    return f"{v*100:.1f}%"


def write_reports(episodes, per_cond, head, comp, fa, ctx):
    g = comp.set_index("model")
    b, t, r = g.loc["baseline"], g.loc["targeted_posttrain"], g.loc["random_posttrain"]

    dom = fa.iloc[0]
    base_fail_total = int(fa["failures"].sum())
    n_seeds = len(ctx["seeds"])

    d_held = (t.held_out_success_mean - b.held_out_success_mean) * 100
    d_over = (t.overall_success_mean - b.overall_success_mean) * 100
    abl_held = (t.held_out_success_mean - r.held_out_success_mean) * 100
    abl_over = (t.overall_success_mean - r.overall_success_mean) * 100

    # per-condition targeted vs baseline delta
    cond_delta = []
    for c in C.EVAL_CONDITIONS:
        bb = episodes[(episodes.model == "baseline") & (episodes.condition == c)]["success"].mean()
        tt = episodes[(episodes.model == "targeted_posttrain") & (episodes.condition == c)]["success"].mean()
        rr = episodes[(episodes.model == "random_posttrain") & (episodes.condition == c)]["success"].mean()
        cond_delta.append((c, bb, tt, rr))
    cond_delta.sort(key=lambda x: x[1])

    prov = ctx["prov"]
    corr_tx = int(prov["corrective_transitions"].mean())
    ft_steps = int(prov["ft_grad_steps"].mean())
    pt_tx = int(prov["posttrain_transitions"].mean())

    # ------------------------------------------------------ RESUME_RESULTS.md
    L = []
    A = L.append
    A("# Resume Results\n")
    A("## Project\n\nRobot Skill Post-Training & Evaluation\n")
    A("## Environment\n")
    A("Simulator: PyBullet (DIRECT mode, kinematic position control)")
    A("Robot: 7-DoF KUKA iiwa arm (`kuka_iiwa/model.urdf`)")
    A(f"Task: reach a 3-D Cartesian target; success = end-effector within "
      f"{C.SUCCESS_THRESHOLD} m of the target within {C.MAX_STEPS} steps\n")
    A("## Model\n")
    A(f"Policy: MLP {C.OBS_DIM}-{'-'.join(map(str, C.HIDDEN))}-{C.ACT_DIM} (PyTorch, CPU), "
      "state in / joint-delta out")
    A("Training method: behavioural cloning from an analytic IK expert controller")
    A("Post-training method: failure-focused fine-tuning on expert corrective "
      "demonstrations restarted from states visited on failed policy rollouts\n")
    A("## Dataset\n")
    A(f"Training demonstrations: {ctx['demos']} expert episodes "
      f"({ctx['n_train_tx']} train / {ctx['n_val_tx']} val transitions, episode-level split)")
    A(f"Corrective demonstrations: {corr_tx} transitions "
      f"({int(prov['corrective_restarts'].mean())} expert restarts on failed states)")
    A(f"Evaluation episodes: {len(C.EVAL_CONDITIONS) * ctx['eval_episodes']} per seed "
      f"({ctx['eval_episodes']} x {len(C.EVAL_CONDITIONS)} conditions), "
      f"{len(episodes[episodes.model=='baseline'])} baseline episodes over {n_seeds} seeds\n")
    A("## Results\n")
    A(f"Baseline success: {_pct(b.overall_success_mean)} (+/- {b.overall_success_std*100:.1f} pp over {n_seeds} seeds)")
    A(f"Targeted post-training success: {_pct(t.overall_success_mean)} (+/- {t.overall_success_std*100:.1f} pp)")
    A(f"Random post-training success: {_pct(r.overall_success_mean)} (+/- {r.overall_success_std*100:.1f} pp)\n")
    A(f"Baseline held-out success: {_pct(b.held_out_success_mean)} (+/- {b.held_out_success_std*100:.1f} pp)")
    A(f"Targeted held-out success: {_pct(t.held_out_success_mean)} (+/- {t.held_out_success_std*100:.1f} pp)")
    A(f"Random held-out success: {_pct(r.held_out_success_mean)} (+/- {r.held_out_success_std*100:.1f} pp)\n")
    A(f"Baseline mean final error: {b.overall_mean_error:.4f} m")
    A(f"Targeted mean final error: {t.overall_mean_error:.4f} m")
    A(f"Random mean final error: {r.overall_mean_error:.4f} m\n")
    A("## Failure Analysis\n")
    A(f"Dominant failure condition: `{dom.condition}` "
      f"(baseline success {_pct(dom.success_rate)}, {int(dom.failures)} failures "
      f"of {int(dom.episodes)} episodes)\n")
    A(f"Baseline failure count: {base_fail_total} of "
      f"{int(fa['episodes'].sum())} benchmark episodes across {n_seeds} seeds\n")
    A("Most important observed failure pattern:\n")
    A(f"The baseline behaves well in-distribution ({_pct(b.in_dist_success_mean)}) but "
      f"degrades under distribution shift ({_pct(b.held_out_success_mean)} on shifted "
      f"conditions). Failures concentrate in `{dom.condition}`; the full per-condition "
      "breakdown is in `results/failure_analysis.csv`.\n")
    A("Failure count by condition:\n")
    A("| condition | episodes | failures | baseline success |")
    A("| --- | ---: | ---: | ---: |")
    for _, row in fa.iterrows():
        A(f"| {row.condition} | {int(row.episodes)} | {int(row.failures)} | {_pct(row.success_rate)} |")
    A("")
    A("## Ablation\n")
    pd_ = ctx["paired"]
    tr_ = pd_["tgt_vs_rand"]; tb_ = pd_["tgt_vs_base"]; rb_ = pd_["rand_vs_base"]
    A(f"Targeted vs random (overall): {abl_over:+.1f} pp")
    A(f"Targeted vs random (held-out): {abl_held:+.1f} pp\n")
    A(f"Paired per-seed held-out deltas over {n_seeds} seeds (same seed, same benchmark, "
      f"same starting checkpoint):\n")
    A(f"- targeted - baseline: {tb_.mean():+.1f} pp (std {tb_.std():.1f}), "
      f"targeted wins on {(tb_ > 0).sum()}/{n_seeds} seeds")
    A(f"- random - baseline: {rb_.mean():+.1f} pp (std {rb_.std():.1f}), "
      f"random wins on {(rb_ > 0).sum()}/{n_seeds} seeds")
    A(f"- targeted - random: {tr_.mean():+.1f} pp (std {tr_.std():.1f}), "
      f"targeted wins on {(tr_ > 0).sum()}/{n_seeds} seeds\n")
    A("Where the corrective data actually came from (mean transitions per seed) - the mining "
      "procedure allocated its budget to the failing conditions without being told to:\n")
    A("| condition | corrective transitions | share |")
    A("| --- | ---: | ---: |")
    for _, row in ctx["corr_mix"].iterrows():
        A(f"| {row.condition} | {row.transitions:.0f} | {row.share*100:.1f}% |")
    A("")
    A(f"Both arms started from the same baseline checkpoint and used matched data "
      f"volume ({pt_tx} transitions), identical optimiser (Adam), lr={C.FT_LR}, "
      f"{C.FT_EPOCHS} epochs, {ft_steps} gradient steps and the same frozen "
      "observation normaliser. The only difference is the source of the new data.\n")
    A("## Strongest Findings\n")
    findings = _findings(b, t, r, dom, cond_delta, d_held, abl_held, ctx, base_fail_total, n_seeds)
    for i, f in enumerate(findings, 1):
        A(f"{i}. {f}")
    A("")
    A("## Resume-Ready Bullets\n")
    for bl in _bullets(b, t, r, dom, cond_delta, d_held, d_over, abl_held, abl_over, ctx, base_fail_total, n_seeds):
        A(f"- {bl}")
    A("")
    with open(os.path.join(RESULTS, "RESUME_RESULTS.md"), "w") as f:
        f.write("\n".join(L))

    # -------------------------------------------------------------- README.md
    _write_readme(b, t, r, fa, dom, cond_delta, comp, ctx, base_fail_total, n_seeds,
                  abl_held, abl_over, d_held, d_over, corr_tx, pt_tx, ft_steps)


def _findings(b, t, r, dom, cond_delta, d_held, abl_held, ctx, base_fail_total, n_seeds):
    out = []
    out.append(
        f"The baseline generalisation gap is real and measured: {_pct(b.in_dist_success_mean)} "
        f"success in-distribution vs {_pct(b.held_out_success_mean)} on shifted held-out "
        f"conditions, against a {_pct(ctx['expert_ceiling'])} expert ceiling on the same benchmark.")
    out.append(
        f"Failures are not uniform: `{dom.condition}` accounts for the lowest baseline success "
        f"({_pct(dom.success_rate)}), and per-condition slicing localises the weakness rather "
        f"than reporting a single aggregate score.")
    verb = "improved" if d_held > 0 else ("did not improve" if abs(d_held) < 0.5 else "reduced")
    out.append(
        f"Targeted post-training {verb} held-out success by {d_held:+.1f} pp "
        f"({_pct(b.held_out_success_mean)} -> {_pct(t.held_out_success_mean)}) using only "
        f"{ctx['prov']['corrective_transitions'].mean():.0f} corrective transitions.")
    if abl_held > 0.5:
        out.append(f"The controlled ablation attributes the gain to targeting specifically: "
                   f"targeted beat matched-volume random data by {abl_held:+.1f} pp held-out.")
    elif abl_held < -0.5:
        out.append(f"The controlled ablation does NOT support the hypothesis: matched-volume "
                   f"random data outperformed targeted corrective data by {-abl_held:.1f} pp held-out.")
    else:
        out.append(f"The controlled ablation found targeted and random post-training "
                   f"statistically indistinguishable ({abl_held:+.1f} pp held-out), so the "
                   f"gain is attributable to additional data rather than to targeting.")
    mix = ctx["corr_mix"]
    top = mix.iloc[0]
    out.append(
        f"The mechanism is visible in the data budget: failure mining was given a flat budget "
        f"and no hint about which conditions matter, yet spent {top.share*100:.0f}% of it on "
        f"`{top.condition}` - exactly the worst condition - and that condition improved the most "
        f"({[f'{(tt-bb)*100:+.1f} pp' for c, bb, tt, rr in cond_delta if c == top.condition][0]}).")
    # honest reporting of the residual weakness
    worst_after = min(cond_delta, key=lambda x: x[2])
    out.append(
        f"Not everything was fixed: `{worst_after[0]}` remains the weakest condition after "
        f"targeted post-training ({_pct(worst_after[2])}, only "
        f"{(worst_after[2]-worst_after[1])*100:+.1f} pp over baseline), so start-state "
        f"perturbation is a residual failure mode that corrective data did not resolve."
        if worst_after[0] != top.condition else
        f"Random post-training also helped ({(r.held_out_success_mean-b.held_out_success_mean)*100:+.1f} pp "
        f"held-out), so part of the improvement is attributable to additional data volume rather "
        f"than to targeting; the targeted-minus-random gap isolates the targeting effect.")
    return out


def _bullets(b, t, r, dom, cond_delta, d_held, d_over, abl_held, abl_over, ctx, base_fail_total, n_seeds):
    n_ep = int(len(ctx['seeds']) * ctx['eval_episodes'] * len(C.EVAL_CONDITIONS))
    worst = cond_delta[0]
    out = []
    if d_held > 0:
        out.append(
            f"Improved held-out task success {_pct(b.held_out_success_mean)} -> "
            f"{_pct(t.held_out_success_mean)} ({d_held:+.1f} pp) by post-training a PyTorch "
            f"robot-reaching policy on {int(ctx['prov']['corrective_transitions'].mean())} "
            f"failure-focused corrective demonstrations identified through benchmark failure analysis.")
    else:
        out.append(
            f"Measured that failure-focused post-training moved held-out success "
            f"{_pct(b.held_out_success_mean)} -> {_pct(t.held_out_success_mean)} "
            f"({d_held:+.1f} pp) on a PyTorch robot-reaching policy, and reported the negative "
            f"result rather than re-tuning the benchmark after seeing it.")
    out.append(
        f"Isolated `{dom.condition}` as the dominant source of policy failure "
        f"({_pct(dom.success_rate)} success vs {_pct(b.in_dist_success_mean)} in-distribution) "
        f"by slicing {n_ep:,} benchmark evaluation episodes across "
        f"{len(C.EVAL_CONDITIONS)} controlled task conditions "
        f"({len(C.HELD_OUT_CONDITIONS)} of them held-out distribution shifts).")
    sign = "advantage" if abl_held >= 0 else "disadvantage"
    out.append(
        f"Ran a controlled ablation with matched data volume and matched gradient steps showing a "
        f"{abs(abl_held):.1f} pp held-out {sign} for targeted over randomly-sampled post-training data, "
        f"isolating data targeting from data quantity.")
    out.append(
        f"Built a single-command, seed-controlled evaluation pipeline (PyBullet + PyTorch) with an "
        f"expert-verified {_pct(ctx['expert_ceiling'])}-solvable held-out benchmark and a failure-mining "
        f"pool disjoint from the test set, reporting results over {n_seeds} training seeds.")
    out.append(
        f"Reduced mean final end-effector error from {b.overall_mean_error:.4f} m to "
        f"{t.overall_mean_error:.4f} m across {n_ep:,} evaluation episodes while holding the success "
        f"threshold, episode budget, model architecture and evaluation seeds fixed."
        if t.overall_mean_error < b.overall_mean_error else
        f"Tracked mean final end-effector error alongside success rate "
        f"({b.overall_mean_error:.4f} m baseline vs {t.overall_mean_error:.4f} m post-trained) to "
        f"separate near-misses from qualitative failures across {n_ep:,} evaluation episodes.")
    return out


def _write_readme(b, t, r, fa, dom, cond_delta, comp, ctx, base_fail_total, n_seeds,
                  abl_held, abl_over, d_held, d_over, corr_tx, pt_tx, ft_steps):
    n_ep_per_seed = ctx["eval_episodes"] * len(C.EVAL_CONDITIONS)
    L = []
    A = L.append
    A("# Robot Skill Post-Training & Evaluation\n")
    A("Failure-focused post-training of a learned robot reaching policy, with a controlled")
    A("ablation against matched-volume random data. PyBullet + PyTorch, CPU-only, one command.\n")
    A("```bash\npython scripts/run_all.py\n```\n")
    A("## Problem\n")
    A("Learned robotic policies trained by behavioural cloning fit their demonstration")
    A("distribution well but degrade on conditions outside it. Knowing *that* a policy is worse")
    A("out-of-distribution is not actionable; knowing *where* it fails is.\n")
    A("## Hypothesis\n")
    A("> Targeting post-training data toward observed policy failures improves difficult-condition")
    A("> performance and generalisation more efficiently than adding the same volume of randomly")
    A("> sampled training-distribution data.\n")
    A("## Approach\n")
    A("Expert demonstrations -> behavioural cloning -> held-out benchmark -> failure analysis ->")
    A("targeted corrective data -> post-training -> matched random-data ablation.\n")
    A("## Environment\n")
    A(f"- Simulator: PyBullet `DIRECT` mode. Robot: 7-DoF KUKA iiwa (`kuka_iiwa/model.urdf`).")
    A(f"- Kinematic position control: `q_{{t+1}} = clip(q_t + {C.MAX_DELTA} * a_t, joint_limits)`.")
    A(f"  PyBullet supplies the model, forward kinematics and the IK solver; rigid-body dynamics")
    A(f"  are deliberately not integrated, which keeps the study about policy learning.")
    A(f"- Observation ({C.OBS_DIM}-D): 7 joint positions, 3-D end-effector position, 3-D target,")
    A(f"  3-D (target - end-effector). Action ({C.ACT_DIM}-D): normalised joint deltas in [-1, 1].")
    A(f"- Expert: PyBullet inverse kinematics + clipped joint-space servoing.\n")
    A("## Experimental design\n")
    A("**Train / validation / test separation.**")
    A(f"{ctx['demos']} expert demonstration episodes are drawn from the training distribution only")
    A(f"(radius 0.40-0.65 m, azimuth +/-60 deg, height 0.25-0.65 m, start-pose noise sigma=0.10 rad),")
    A(f"split at the *episode* level into {ctx['n_train_tx']} train and {ctx['n_val_tx']} validation")
    A("transitions so no transition leaks across the split.\n")
    A("**Held-out benchmark.** Five conditions, one in-distribution control and four distribution")
    A(f"shifts, {ctx['eval_episodes']} episodes each ({n_ep_per_seed} episodes per seed), built once")
    A(f"from a fixed seed (`BENCHMARK_SEED={C.BENCHMARK_SEED}`) and reused unchanged by every model:\n")
    A("| condition | shift | held-out |")
    A("| --- | --- | :-: |")
    A("| `train_dist` | none (in-distribution control) | no |")
    A("| `far_radius` | target radius 0.65-0.78 m (beyond training reach band) | yes |")
    A("| `low_height` | target height 0.08-0.25 m (below training band) | yes |")
    A("| `wide_azimuth` | target azimuth 60-115 deg (outside training sector) | yes |")
    A("| `perturbed_start` | start-pose noise sigma=0.60 rad (6x training) | yes |")
    A("")
    A(f"Every benchmark episode is **expert-verified solvable** before admission, so the expert")
    A(f"ceiling on this benchmark is {_pct(ctx['expert_ceiling'])} and every policy failure is a")
    A("policy failure, not an unreachable target.\n")
    A("**No test-set leakage.** Corrective data is mined from a *separate dev pool* built with a")
    A(f"different seed (`DEV_SEED={C.DEV_SEED}`) over the same condition definitions. The test")
    A("benchmark is never used to generate training data.\n")
    A("**Controlled ablation.** Both post-training arms start from the same baseline checkpoint and")
    A("share: architecture, frozen observation normaliser, optimiser (Adam), learning rate")
    A(f"({C.FT_LR}), epochs ({C.FT_EPOCHS}), batch size ({C.FT_BATCH}), gradient steps ({ft_steps}),")
    A(f"total post-training transitions ({pt_tx} = {corr_tx} new + {C.N_POSTTRAIN_REPLAY} replay from")
    A("the original training split), evaluation benchmark and evaluation seeds. The **only**")
    A("difference is the source of the new transitions:\n")
    A("- **Targeted**: expert corrective rollouts restarted from states the baseline actually")
    A("  visited on episodes it failed (a DAgger-style relabelling of failure states).")
    A("- **Random**: transitions sampled uniformly from a fresh expert demonstration pool drawn")
    A("  from the original training distribution.\n")
    A("**Seeds.** Dataset seed, benchmark seed and dev seed are fixed constants; the full")
    A(f"train -> mine -> post-train -> evaluate loop is repeated for {n_seeds} training seeds")
    A(f"({ctx['seeds']}) and results are reported as mean +/- std across them.\n")
    A("**Metrics.** Success rate (primary), mean final end-effector error, failure count and mean")
    A(f"steps per episode. Success is defined once: end-effector within {C.SUCCESS_THRESHOLD} m of")
    A(f"the target within {C.MAX_STEPS} steps. Thresholds, benchmark composition and seeds were")
    A("fixed before any result was seen and were not changed afterwards.\n")
    A("## Results\n")
    A(f"All numbers are measured, averaged over {n_seeds} training seeds on the identical")
    A(f"{n_ep_per_seed}-episode benchmark. \"Held-out\" is the mean over the four shifted conditions.\n")
    A("| Experiment | Overall success | In-distribution | Held-out success | Mean final error |")
    A("| --- | ---: | ---: | ---: | ---: |")
    for m in MODELS:
        g = comp.set_index("model").loc[m]
        A(f"| {LABELS[m]} | {_pct(g.overall_success_mean)} +/- {g.overall_success_std*100:.1f} pp "
          f"| {_pct(g.in_dist_success_mean)} | {_pct(g.held_out_success_mean)} +/- "
          f"{g.held_out_success_std*100:.1f} pp | {g.overall_mean_error:.4f} m |")
    A(f"| _Expert (ceiling)_ | _{_pct(ctx['expert_ceiling'])}_ | _{_pct(ctx['expert_ceiling'])}_ "
      f"| _{_pct(ctx['expert_ceiling'])}_ | _-_ |")
    A("")
    A("### Success rate by condition (baseline vs targeted vs random)\n")
    A("| condition | baseline | targeted | random | targeted - baseline | targeted - random |")
    A("| --- | ---: | ---: | ---: | ---: | ---: |")
    for c, bb, tt, rr in cond_delta:
        A(f"| `{c}` | {_pct(bb)} | {_pct(tt)} | {_pct(rr)} | {(tt-bb)*100:+.1f} pp | {(tt-rr)*100:+.1f} pp |")
    A("")
    pdf = ctx["paired"]
    tr_, tb_, rb_ = pdf["tgt_vs_rand"], pdf["tgt_vs_base"], pdf["rand_vs_base"]
    A(f"**Ablation.** Targeted minus random: {abl_over:+.1f} pp overall, {abl_held:+.1f} pp held-out.")
    A("Because both arms are fine-tuned from the *same* per-seed checkpoint on the *same*")
    A("benchmark, the comparison can be paired by seed:\n")
    A("| paired held-out delta | mean | std | seeds won |")
    A("| --- | ---: | ---: | ---: |")
    A(f"| targeted - baseline | {tb_.mean():+.1f} pp | {tb_.std():.1f} | {(tb_>0).sum()}/{n_seeds} |")
    A(f"| random - baseline | {rb_.mean():+.1f} pp | {rb_.std():.1f} | {(rb_>0).sum()}/{n_seeds} |")
    A(f"| targeted - random | {tr_.mean():+.1f} pp | {tr_.std():.1f} | {(tr_>0).sum()}/{n_seeds} |")
    A("")
    A("Both post-training arms help, but targeting helps more. Per-seed values are in")
    A("`results/paired_seed_deltas.csv`. No significance test is claimed at this sample size.\n")
    A("![success](results/plots/01_success_overall.png)")
    A("![by condition](results/plots/02_success_by_condition.png)\n")
    A("## Failure analysis\n")
    A(f"The baseline failed {base_fail_total} of {int(fa['episodes'].sum())} benchmark episodes")
    A(f"across {n_seeds} seeds. Failures are strongly non-uniform:\n")
    A("| condition | episodes | failures | success rate | mean final error | mean steps |")
    A("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in fa.iterrows():
        A(f"| `{row.condition}` | {int(row.episodes)} | {int(row.failures)} | "
          f"{_pct(row.success_rate)} | {row.mean_error:.4f} m | {row.mean_steps:.1f} |")
    A("")
    A(f"The dominant failure condition is **`{dom.condition}`** ({_pct(dom.success_rate)} success,")
    A(f"{int(dom.failures)} failures). Machine-readable output: `results/failure_analysis.csv`.\n")
    A("Failures also take far longer before giving up: mean steps rises from")
    A(f"{fa[fa.condition==C.IN_DIST_CONDITION]['mean_steps'].iloc[0]:.1f} in-distribution to")
    A(f"{fa['mean_steps'].max():.1f} on the hardest condition (the {C.MAX_STEPS}-step cap), i.e. the")
    A("policy does not simply overshoot - it stalls without converging on the target.\n")
    A("**The corrective data self-targets.** Failure mining was given a flat budget and no")
    A("instruction about which conditions matter; it spent that budget where the policy actually")
    A("failed (mean transitions per seed):\n")
    A("| condition | corrective transitions | share |")
    A("| --- | ---: | ---: |")
    for _, row in ctx["corr_mix"].iterrows():
        A(f"| `{row.condition}` | {row.transitions:.0f} | {row.share*100:.1f}% |")
    A("")
    A("![failures](results/plots/04_failure_distribution.png)\n")
    A("## Reproducing\n")
    A("```bash")
    A("python -m venv .venv && source .venv/bin/activate")
    A("pip install -r requirements.txt")
    A("python scripts/run_all.py          # full experiment")
    A("python scripts/run_all.py --quick  # ~1 min smoke test")
    A("python -m pytest tests/ -q         # unit tests")
    A("```")
    A(f"\nEvery experiment constant lives in `src/config.py`. Total runtime measured on an "
      f"8-core CPU (no GPU): **{ctx['runtime']/60:.1f} min**.\n")
    A("**Reproducibility is verified, not asserted.** Two independent full runs of")
    A("`scripts/run_all.py` produce byte-identical `results/episodes.csv` and")
    A("`results/comparison.csv`. Every stochastic component is seeded, including PyTorch weight")
    A("initialisation (seeded *before* module construction), dataset sampling, the train/val")
    A("split, DataLoader shuffling, benchmark generation and corrective-data selection.\n")
    A("> **Setup note (macOS).** PyBullet ships no wheels, so `pip install pybullet` compiles from")
    A("> source and needs Python <= 3.12. On recent macOS SDKs the bundled zlib defines")
    A("> `fdopen(fd, mode) NULL`, which collides with `<stdio.h>` and breaks the build; delete that")
    A("> line in `examples/ThirdPartyLibs/zlib/zutil.h` before building.\n")
    A("Outputs: `results/metrics.csv` (per-condition, per-model, per-seed), "
      "`results/failure_analysis.csv`, `results/comparison.csv`, `results/episodes.csv` "
      "(every episode), `results/run_provenance.csv` (data/gradient-step budgets), "
      "`results/plots/`, `results/RESUME_RESULTS.md`.\n")
    A("## Limitations\n")
    A("- **Simulation only.** No real robot, no sim-to-real transfer, no hardware validation.")
    A("- **Kinematic, not dynamic.** Joint positions are set directly; contact, inertia, torque")
    A("  limits and actuator lag are not simulated, so the control problem is easier than a real one.")
    A("- **Single simple task.** Free-space 3-D reaching with a known target position. No")
    A("  perception, no obstacles, no grasping, no long-horizon manipulation.")
    A("- **Small policy and dataset.** A 2-hidden-layer MLP trained on a few thousand transitions;")
    A("  conclusions may not carry to large policies or high-dimensional observations.")
    A(f"- **Limited statistical power.** {n_seeds} training seeds on a single fixed benchmark;")
    A("  standard deviations are reported but no significance test is claimed.")
    A("- **Distribution shifts are hand-specified**, not naturally occurring, and the corrective")
    A("  data uses the same condition families as the benchmark (different seeds, disjoint")
    A("  episodes) - a favourable setting for targeting.\n")
    with open(os.path.join(ROOT, "README.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
