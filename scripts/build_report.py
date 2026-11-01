"""Generate a coursework report from measured CSV/JSON artifacts.

Missing or partial experiments are reported explicitly. This script never
invents a learning curve, extrapolates to the requested episode budget, or
substitutes smoke-test results for the main experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def read_json(path, required=True):
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    if required:
        raise FileNotFoundError(path)
    return None


def read_episodes(path):
    with Path(path).open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No measured episodes in {path}")
    if [int(row["episode"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Report requires a contiguous episode log beginning at episode 1")
    return rows


def link(path, report_path, label=None):
    relative = Path(os.path.relpath(Path(path).resolve(), Path(report_path).resolve().parent)).as_posix()
    return f"[{label or Path(path).name}]({relative})"


def figure(path, report_path, caption):
    if not Path(path).exists():
        return ""
    return "!" + link(path, report_path, caption) + "\n\n" + caption + "\n"


def mean(rows, field):
    return sum(float(row[field]) for row in rows) / len(rows)


def fmt_interval(metric):
    ci = metric.get("normal_approx_95_ci")
    return f"{metric['mean']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else f"{metric['mean']:.4f} (interval unavailable)"


def evaluation_table(evaluation):
    result = ["| Policy | Evaluation episodes | Mean collective return [95% CI] | Any-agent success | All-agent success |",
              "|---|---:|---:|---:|---:|"]
    names = {"stochastic": "PPO, sampled actions", "deterministic": "PPO, greedy actions", "random": "Uniform random actions"}
    for policy in ["stochastic", "deterministic", "random"]:
        metric = evaluation["policies"][policy]
        result.append(f"| {names[policy]} | {metric['collective_return']['n_episodes']} | "
                      f"{fmt_interval(metric['collective_return'])} | "
                      f"{100 * metric['any_agent_success']['rate']:.1f}% | "
                      f"{100 * metric['all_agents_success']['rate']:.1f}% |")
    return "\n".join(result)


def generate(run_dir, output, single_agent_dir=None):
    run_dir, output = Path(run_dir), Path(output)
    project_root = Path(__file__).resolve().parents[1]
    validation_path = project_root / "VALIDATION.json"
    validation = read_json(validation_path, required=False)
    github_status_path = project_root / "GITHUB_SUBMISSION.json"
    github_status = read_json(github_status_path, required=False)
    config = read_json(run_dir / "config.json")
    training = read_json(run_dir / "summary.json", required=False)
    evaluation = read_json(run_dir / "evaluation" / "summary.json", required=False)
    wandb_status_path = run_dir / "wandb_sync_status.json"
    wandb_status = read_json(wandb_status_path, required=False)
    rows = read_episodes(run_dir / "episodes.csv")
    agents = sum(key.startswith("agent_") and key.endswith("_return") for key in rows[0])
    target, measured = int(config["total_episodes"]), len(rows)
    complete = bool(training and training.get("status") == "complete"
                    and training.get("completed_episodes") == measured == target
                    and (run_dir / "checkpoints" / "final.pt").exists())
    if training and training.get("completed_episodes") != measured:
        raise ValueError("Training summary and episode log disagree; finish or reconcile the run before reporting")
    if evaluation and evaluation["training_completed_episodes"] != measured:
        raise ValueError("Evaluation refers to a different checkpoint episode count; reevaluate the final checkpoint")
    window = min(1000, measured)
    first, last = rows[:window], rows[-window:]
    first_mean, last_mean = mean(first, "collective_return"), mean(last, "collective_return")
    transitions = int(training["joint_environment_steps"]) if training else int(rows[-1]["global_step"])
    elapsed = float(training["elapsed_seconds"]) if training else float(rows[-1]["elapsed_seconds"])
    algorithm = "independent PPO (IPPO)" if agents > 1 else "PPO"
    cloud_verified = bool(wandb_status and wandb_status.get("cloud_verified") is True)
    if cloud_verified:
        if (int(wandb_status["remote_completed_episode"]) != measured
                or int(wandb_status["remote_joint_environment_steps"]) != transitions):
            raise ValueError("Verified W&B history and local training totals disagree")
        if not wandb_status["all_cloud_returns_match_local_100_episode_means"]:
            raise ValueError("W&B verification does not confirm agreement with local returns")
        history_points = int(wandb_status["history_points"])
        history_interval = int(wandb_status["history_interval_episodes"])
        run_url = wandb_status["run_url"]
        wandb_description = (
            "The offline training history was synchronized to W&B and verified against the local CSV: "
            f"**{history_points:,} logged points, one every {history_interval:,} episodes**, through "
            f"episode {int(wandb_status['remote_completed_episode']):,}. All cloud return points match "
            f"the corresponding local {history_interval}-episode means. The remote run is "
            f"`{wandb_status['run_state']}`. [W&B run]({run_url}); "
            + link(wandb_status_path, output, "Cloud verification record") + ".\n")
        cloud_submission_status = (
            f"W&B cloud synchronization and verification are complete: [verified run]({run_url}). ")
    else:
        wandb_description = (
            "W&B data were saved offline; cloud synchronization is pending, and no cloud dashboard URL is claimed.\n"
            if config["wandb_mode"] == "offline" else
            "No online dashboard availability is inferred solely from the logging configuration.\n")
        cloud_submission_status = "W&B cloud synchronization remains pending. "
    verified_source_commit = github_status.get("verified_source_commit") if github_status else None
    github_uploaded = bool(github_status and github_status.get("uploaded") is True
                           and isinstance(verified_source_commit, str)
                           and verified_source_commit.strip())
    if github_uploaded:
        github_submission_status = (
            f"The completed source is published in the [GitHub fork]({github_status['repository_url']}) "
            f"on branch `{github_status['branch']}`. The uploaded source commit "
            f"`{verified_source_commit}` was verified. "
            + link(github_status_path, output, "GitHub publication verification record") + ".\n")
    else:
        github_submission_status = (
            "GitHub publication remains pending; no verified uploaded source commit is recorded.\n")
    lines = [
        "# PPO on MultiGrid: implementation and experiments\n",
        "## Experiment status\n",
        f"The recorded {algorithm} run completed **{measured:,} of {target:,} requested episodes**. "
        + ("The training log and final summary agree on completion." if complete else
           "This is a partial experiment; the requested training budget has not been verified as complete."),
        f"\nThere are {agents} independently controlled agent(s), {transitions:,} recorded joint environment transitions, "
        f"and {transitions * agents:,} agent decisions. Measured training time was {elapsed / 60:.2f} minutes. "
        "All numerical results below are calculated from saved local logs.\n",
        "## Task and environment\n",
        f"The environment is `{config['env_id']}`. The required multi-agent task uses a 15 by 15 grid, "
        "three agents, 30 interior wall obstacles, a fixed goal at grid coordinate (13, 13), "
        "and a 100-step joint episode limit. Each agent receives only its own 5 by 5 partial view "
        "and its own heading. The seven discrete actions are turn left, turn right, move forward, "
        "pick up, drop, toggle, and done (a no-op in this task).\n",
        "In the three-agent task, an agent that reaches the goal receives "
        r"$r_t^i = 1 - 0.9t/100$ (using the environment's 1-based step counter) and respawns. Other agents receive no reward from that event. "
        "Reaching the goal does not end the joint episode; one agent can score repeatedly before the timeout. "
        "The sticky `env.done[i]` flag records whether the agent has succeeded at least once. "
        "It is not an absorbing termination signal for learning. Collective episodic return is "
        r"$R_{\mathrm{team}} = \sum_{i=1}^{N}\sum_{t=0}^{T-1} r_t^i$. "
        "This sum is used for reporting, while each policy learns from its own rewards.\n",
        f"The obstacle layout uses seed {config['layout_seed']}. Episode and action-order randomness use "
        "per-environment seeded random generators, with varying starting positions. Wall-clock reseeding "
        "was removed so the experiment and evaluations can be reproduced. A cached observation path "
        "reuses only immutable wall/goal views and overlays the visible agents; regression tests compare "
        "it against the original observation construction, including occlusion and goal respawns.\n",
        "## Independent PPO implementation\n",
        "Each agent has its own actor-critic network, Adam optimizer, rollout buffer, rewards, "
        "advantages, and value targets. There is no policy sharing, centralized critic, or use of another "
        "agent's separate observation. Other agents can still be visible inside the agent's legitimate "
        "local field of view. Batched rollout inference stacks copies of the separate networks for speed; "
        "it does not mix or tie their trainable parameters.\n",
        "The encoded grid contains categorical object, color/agent-identity, and state/direction IDs. "
        "Their one-hot dimensions are 11, 6, and 4 per visible cell. The 25 cells and four-way own-heading "
        f"encoding give 529 input features. Two fully connected layers of {config['hidden_size']} tanh units "
        "feed a categorical actor and a scalar critic. The actor and critic share the encoder within "
        "one agent, but no network weights are shared across agents.\n",
        "For agent $i$, the probability ratio and clipped surrogate are\n",
        r"$$\rho_t^i(\theta_i)=\frac{\pi_{\theta_i}(a_t^i\mid o_t^i)}{\pi_{\theta_{i,\mathrm{old}}}(a_t^i\mid o_t^i)},$$" + "\n",
        r"$$L_{\mathrm{clip}}^i=\mathbb{E}_t\left[\min\left(\rho_t^i\widehat A_t^i,\operatorname{clip}(\rho_t^i,1-\epsilon,1+\epsilon)\widehat A_t^i\right)\right].$$" + "\n",
        "The minimized loss combines the negative clipped objective, value loss, and an entropy bonus:\n",
        r"$$L^i=-L_{\mathrm{clip}}^i+c_vL_V^i-c_e\mathbb{E}_t[H(\pi_i(\cdot\mid o_t^i))].$$" + "\n",
        "Generalized advantage estimation uses the TD residual "
        r"$\delta_t^i=r_t^i+\gamma m_tV_i(o_{t+1}^i)-V_i(o_t^i)$ and recursion "
        r"$\widehat A_t^i=\delta_t^i+\gamma\lambda m_t\widehat A_{t+1}^i$. "
        "Actual terminations cut both bootstrapping and the advantage trace. For a time-limit truncation, "
        "the implementation adds the discounted value of the actual final observation to the training "
        "reward before cutting the trace. It never bootstraps from the next episode's reset observation. "
        "Logged returns always use the original environment reward, not this bootstrap adjustment. "
        "Goal respawns in the continuing multi-agent task do not cut the trace.\n",
        "PPO updates shuffle each agent's own rollout into minibatches, normalize advantages within "
        "each minibatch, use clipped value loss when configured, and clip the gradient norm. "
        "Approximate KL can stop additional update epochs after an epoch exceeds the configured threshold. "
        "The learning rate is linearly annealed according to completed episodes.\n",
        "## Training configuration\n",
        "| Setting | Recorded value |\n|---|---:|",
    ]
    settings = [
        ("Training seed", "seed"), ("Layout seed", "layout_seed"),
        ("Target joint episodes", "total_episodes"), ("Parallel environment instances", "num_envs"),
        ("Rollout horizon per environment", "num_steps"), ("Hidden units per layer", "hidden_size"),
        ("Initial learning rate", "learning_rate"), ("Discount gamma", "gamma"),
        ("GAE lambda", "gae_lambda"), ("PPO clipping coefficient", "clip_coef"),
        ("Minibatches per update", "num_minibatches"), ("Maximum update epochs", "update_epochs"),
        ("Entropy coefficient", "entropy_coef"), ("Value coefficient", "value_coef"),
        ("Gradient norm limit", "max_grad_norm"), ("KL early-stop threshold", "target_kl"),
        ("Clipped value loss", "clip_value_loss"), ("Learning-rate annealing", "anneal_lr"),
        ("Device", "device"), ("W&B logging mode", "wandb_mode"),
    ]
    lines.extend(f"| {name} | {config[key]} |" for name, key in settings)
    lines += [
        "\nThe training seed is one run, not a population of independent trained models. "
        "All completed episodes are preserved in `episodes.csv`. W&B reports rolling training summaries; "
        "the local CSV remains the source of truth. "
        + wandb_description,
        "## Measured learning curve\n",
        f"The first {window:,} recorded episodes had mean collective return **{first_mean:.4f}**; "
        f"the final {window:,} had mean **{last_mean:.4f}**, a change of **{last_mean-first_mean:+.4f}**. "
        "These are descriptive windows from the same training run, not an independent-seed significance test.\n",
        "| Agent | First-window mean return | Final-window mean return | Final-window success rate |\n|---|---:|---:|---:|",
    ]
    for i in range(agents):
        lines.append(f"| {i} | {mean(first, f'agent_{i}_return'):.4f} | {mean(last, f'agent_{i}_return'):.4f} | {100*mean(last, f'agent_{i}_success'):.1f}% |")
    lines += ["", figure(run_dir / "figures" / "learning_curve.png", output,
                         "Learning curve from every recorded training episode. The labeled trailing mean makes the trend easier to read; the raw episode returns are retained. No missing episodes are extrapolated.")]
    lines += ["## Held-out evaluation\n"]
    if evaluation:
        first_seed, final_seed = evaluation["held_out_seed_range"]
        lines += [
            f"The saved checkpoint after {evaluation['training_completed_episodes']:,} episodes was evaluated "
            f"on seeds {first_seed:,} through {final_seed:,}. The same seeds were used for "
            "sampled policy actions, greedy policy actions, and independent uniform random actions. "
            "Policies were frozen throughout evaluation. Starting-state seeds were held out, while "
            "the fixed obstacle layout was the same as in training; this evaluates new episodes on the "
            "trained map, not generalization to unseen maps.\n",
            evaluation_table(evaluation), "",
            "Return intervals use mean +/- 1.96 times the sample standard deviation divided by the "
            "square root of the evaluation episode count. They describe Monte Carlo uncertainty across "
            "evaluation episodes for one checkpoint. Success is reaching the goal at least once per "
            "agent; the full JSON also provides Wilson intervals for success probabilities.\n",
            "| Policy | Agent | Mean individual return | Success rate |\n|---|---:|---:|---:|",
        ]
        for policy in ["stochastic", "deterministic", "random"]:
            metrics = evaluation["policies"][policy]
            for i in range(agents):
                lines.append(f"| {policy.capitalize()} | {i} | {metrics[f'agent_{i}_return']['mean']:.4f} | {100*metrics[f'agent_{i}_success']['rate']:.1f}% |")
        lines.append("")
        for policy in ["stochastic", "deterministic"]:
            difference = evaluation["paired_improvement_over_random"][policy]
            lines.append(f"The {policy} policy's mean paired return difference relative to random actions was "
                         f"**{fmt_interval(difference)}**. Pairing is by starting-state seed.\n")
        lines.append(figure(run_dir / "figures" / "evaluation_comparison.png", output,
                            "Held-out returns: each point is one episode; diamonds show policy means and bars show approximate 95% confidence intervals. Horizontal offsets only separate overlapping points."))
    else:
        lines.append("No final-checkpoint evaluation has been recorded for this run. No held-out performance claim is made.\n")
    lines += ["## Trajectory evidence\n",
              "The video renderer places the full grid beside every agent's own partial observation, "
              "the next action, the previous transition's reward, and the cumulative individual rewards. "
              "Frame $t$ shows state $s_t$ and proposed action $a_t$; the displayed return includes only "
              "transitions already completed before $a_t$. The full-grid display is for interpretation "
              "and is not an input to the policies.\n"]
    videos = sorted((run_dir / "evaluation").glob("trajectory_*.mp4"))
    if videos:
        lines.append("Each video is the **first held-out episode**, selected in advance, with no filtering for high return or success.\n")
        for path in videos:
            lines.append("- " + link(path, output, path.stem.replace("_", " ")))
            for suffix in [".gif", ".json"]:
                if path.with_suffix(suffix).exists():
                    lines.append("- " + link(path.with_suffix(suffix), output, path.stem + suffix))
        lines.append("")
    else:
        lines.append("No trajectory video has been recorded for this run yet.\n")
    if single_agent_dir:
        single_dir = Path(single_agent_dir)
        single_summary = read_json(single_dir / "summary.json", required=False)
        single_eval = read_json(single_dir / "evaluation" / "summary.json", required=False)
        if single_summary and single_eval:
            lines += [
                "## Single-agent debugging experiment\n",
                f"Before the full multi-agent run, PPO was checked on `{single_eval['environment']}` "
                f"for {single_summary['completed_episodes']:,} episodes and "
                f"{single_summary['joint_environment_steps']:,} transitions. In this single-agent environment, "
                "goal success terminates the episode. It therefore differs from the three-agent continuing "
                "respawn task; its absolute returns are not a like-for-like multi-agent baseline.\n",
                evaluation_table(single_eval), "",
                figure(single_dir / "figures" / "learning_curve.png", output,
                       "Single-agent debugging curve. This separate experiment checks that the PPO implementation can learn before the full multi-agent run."),
            ]
    lines += [
        "## Verification and limitations\n",
        "The accompanying regression tests check environment shapes and seven-action semantics, "
        "per-agent goal rewards and respawning, independent seeded randomness, cached/reference "
        "observation parity, separate policy and rollout storage, batched/individual inference agreement, "
        "GAE boundary and timeout handling, exact episode counts, checkpoint continuation, evaluation "
        "reward accounting, paired evaluation seeds, and absence of policy updates during evaluation.\n",
    ]
    if validation:
        lines.append(
            f"The recorded validation run passed **{validation['tests_passed']} tests with "
            f"{validation['tests_failed']} failures**. The final artifact audit passed "
            f"**{validation['final_artifact_audit_passed']} checks with "
            f"{validation['final_artifact_audit_failed']} failures**. This includes checking the "
            "completed training budget, checkpoints, evaluation outputs and presentation artifacts. "
            + link(validation_path, output, "Validation record") + "; "
            + link(project_root / validation["final_artifact_audit"], output, "Detailed artifact audit") + ".\n")
    else:
        lines.append("No test-run validation record was found for this report generation.\n")
    visual_qa = run_dir / "figures" / "qa_summary.json"
    if visual_qa.exists():
        lines.append("Final figures and both trajectory videos were visually reviewed, with PDF text-size "
                     "and collision checks saved beside the figures. "
                     + link(visual_qa, output, "Figure and video QA record") + ".\n")
    lines += [
        "The evidence is limited to one training seed and one fixed obstacle layout. Evaluation confidence "
        "intervals must not be interpreted as uncertainty across retrained models. The policy is feedforward "
        "under partial observability, so observations can alias different situations. Greedy action "
        "selection may behave differently from the stochastic training policy. A rise in collective "
        "return alone does not establish learned cooperation: agents optimize separate rewards and "
        "may differ in their individual success rates. Uniform random actions are a basic reference, "
        "not a comparison against other trained multi-agent methods. No state-of-the-art claim is made.\n",
        "## External submission status\n",
        "The code, checkpoints, measured results, figures, videos and this report are available locally. "
        + cloud_submission_status + github_submission_status,
        "## Reproduction and artifacts\n",
        "Use the recorded configuration and the commands in the project README. The essential artifacts are:\n",
        "- " + link(run_dir / "config.json", output, "Exact training configuration"),
        "- " + link(run_dir / "episodes.csv", output, "All measured training episodes"),
        "- " + link(run_dir / "updates.jsonl", output, "PPO update diagnostics"),
    ]
    for file, label in [(run_dir / "checkpoints" / "final.pt", "Final independent actor-critic checkpoint"),
                        (run_dir / "evaluation" / "episodes.csv", "Individual held-out episode results"),
                        (run_dir / "evaluation" / "summary.json", "Evaluation protocol and statistical summaries")]:
        if file.exists():
            lines.append("- " + link(file, output, label))
    lines += [
        "\n## References and attribution\n",
        "1. Schulman et al. (2017). [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347).",
        "2. Schulman et al. (2015). [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438).",
        "3. de Witt et al. (2020). [*Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?*](https://arxiv.org/abs/2011.09533).",
        "4. [CleanRL PPO reference implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py).",
        "5. [Natasha Jaques' MultiGrid starter repository](https://github.com/natashamjaques/multigrid).",
        "\nThe supplied CleanRL-style PPO script is archived in `legacy/ppo_original.py`. Its clipped-PPO "
        "and GAE components informed the independent multi-agent implementation in "
        "`multiagent_metacontroller.py` and `ippo_agent.py`. The active `ppo.py` is a thin single-agent "
        "entry point into this same tested core; `main.py` is the main training entry point. "
        "Third-party copyright notices and source links are preserved in "
        "`THIRD_PARTY_NOTICES.md` and `LICENSE`. The StarCraft paper motivates independent PPO as a "
        "baseline; it is not evidence for the MultiGrid results reported here.\n",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(output.resolve()), "recorded_episodes": measured,
            "requested_episodes": target, "verified_training_complete": complete,
            "evaluation_present": evaluation is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/ippo_seed1"))
    parser.add_argument("--single-agent-dir", type=Path, default=Path("results/single_agent_seed1"))
    parser.add_argument("--output", type=Path, default=Path("REPORT.md"))
    args = parser.parse_args()
    print(json.dumps(generate(args.run_dir, args.output, args.single_agent_dir), indent=2))


if __name__ == "__main__":
    main()
