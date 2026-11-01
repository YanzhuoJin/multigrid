"""Strict, read-only audit of the completed 100,000-episode submission.

The only output written is the requested JSON audit. Missing, partial, corrupt,
or inconsistent results fail; an in-progress run is never marked complete.
Load only checkpoints produced locally by this project (torch pickle format).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
EXPECTED_EPISODES = 100_000
EXPECTED_STEPS = 10_000_000
POLICIES = ("stochastic", "deterministic", "random")
# Source reviewed against the supplied project before the full run. A later
# environment edit requires a fresh review; it must not silently pass this audit.
REVIEWED_ENVIRONMENT_SHA256 = {
    "env_adapter.py": "0135f4572cf0658f1a764b2ef42abb4669309b24cfdf51981341edea8d4597e0",
    "envs/gym_multigrid/multigrid.py": "acae8a4d26990ec9eb4e898d8c95aac4ed29b9f30200daeff6c317cacd4b302b",
    "envs/gym_multigrid/multigrid_envs/cluttered.py": "fb066f64d5ae99b02b99039e7b33689ca742e9ec8282c14de621bb60813d89c0",
    "envs/gym_multigrid/register.py": "1c127408797cace2db119e25089ec9f81024c8c4817d7609955b87fd43b7d590",
}


def json_safe(value):
    """Keep a corrupt numeric input report valid JSON instead of writing NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_value": repr(value)}
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class Audit:
    def __init__(self):
        self.checks = []
        self.artifacts = {}

    def check(self, name, passed, detail):
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def attempt(self, name, operation):
        try:
            return operation()
        except Exception as error:
            self.check(name, False, f"{type(error).__name__}: {error}")
            return None

    def artifact(self, path):
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Required nonempty file is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.artifacts[str(path)] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        return path

    def json(self, path):
        value = json.loads(self.artifact(path).read_text())
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        return value

    def csv(self, path):
        with self.artifact(path).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No episode rows: {path}")
        return rows


def audit_training_rows(audit, rows, expected_episodes=EXPECTED_EPISODES):
    episodes = np.array([int(row["episode"]) for row in rows])
    lengths = np.array([int(row["episode_length"]) for row in rows])
    steps = np.array([int(row["global_step"]) for row in rows])
    returns = np.array([[float(row[f"agent_{i}_return"]) for i in range(3)] for row in rows])
    collective = np.array([float(row["collective_return"]) for row in rows])
    successes = np.array([[int(row[f"agent_{i}_success"]) for i in range(3)] for row in rows])
    audit.check("exact_episode_count", len(rows) == expected_episodes,
                {"observed": len(rows), "required": expected_episodes})
    audit.check("unique_contiguous_episode_indices",
                np.array_equal(episodes, np.arange(1, expected_episodes + 1)),
                "Indices must be exactly 1 through the required final episode, once each.")
    audit.check("fixed_100_step_episodes", np.all(lengths == 100),
                {"minimum": int(lengths.min()), "maximum": int(lengths.max())})
    audit.check("exact_joint_environment_steps",
                lengths.sum() == expected_episodes * 100 and steps[-1] == expected_episodes * 100,
                {"sum_episode_lengths": int(lengths.sum()), "final_global_step": int(steps[-1])})
    audit.check("nondecreasing_logged_global_steps", np.all(np.diff(steps) >= 0),
                "Parallel episodes may share a global-step count; it must never decrease.")
    audit.check("finite_nonnegative_training_rewards",
                np.isfinite(returns).all() and np.isfinite(collective).all() and
                (returns >= 0).all() and (collective >= 0).all(),
                "All recorded individual and collective returns are finite and nonnegative.")
    audit.check("collective_equals_sum_of_agents",
                np.allclose(collective, returns.sum(axis=1), rtol=0, atol=1e-9),
                "Every collective return equals the sum of the three individual returns.")
    audit.check("success_flags_match_positive_rewards",
                np.isin(successes, [0, 1]).all() and np.array_equal(successes, returns > 0),
                "Success is reaching the goal at least once, not joint termination.")
    return collective


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit_checkpoint(audit, path, label):
    import torch
    payload = torch.load(audit.artifact(path), map_location="cpu", weights_only=False)
    agents = payload["agents"]
    audit.check(f"{label}_checkpoint_completion",
                payload["completed_episodes"] == EXPECTED_EPISODES and
                payload["global_step"] == EXPECTED_STEPS and
                payload["algorithm"] == "independent_ppo",
                {key: payload.get(key) for key in ("algorithm", "completed_episodes", "global_step")})
    digests = [state_digest(state) for state in agents]
    audit.check(f"{label}_three_distinct_finite_policies",
                len(agents) == 3 and len(set(digests)) == 3 and
                all(torch.isfinite(value).all().item() for state in agents for value in state.values()),
                {"policy_count": len(agents), "policy_sha256": digests})
    storages = [{value.untyped_storage().data_ptr() for value in state.values()} for state in agents]
    audit.check(f"{label}_independent_parameter_storage",
                all(not storages[i].intersection(storages[j])
                    for i in range(len(storages)) for j in range(i + 1, len(storages))),
                "Policy state dictionaries share no parameter storage.")
    audit.check(f"{label}_three_optimizers", len(payload["optimizers"]) == 3,
                {"optimizer_count": len(payload["optimizers"])})
    spec = payload["model_spec"]
    audit.check(f"{label}_local_observation_and_action_dimensions",
                tuple(spec["obs_shape"]) == (5, 5, 3) and spec["n_actions"] == 7,
                spec)
    return payload, digests


def audit_environment_contract(audit, payload):
    from env_adapter import make_env
    env = make_env(seed=1, layout_seed=52)
    snapshot = pickle.loads(payload["runtime"]["environments"][0])
    required = (env.width, env.height, env.n_agents, env.max_steps,
                env.n_clutter, env.layout_seed, env.competitive, len(env.actions))
    observed = (snapshot.width, snapshot.height, snapshot.n_agents, snapshot.max_steps,
                snapshot.n_clutter, snapshot.layout_seed, snapshot.competitive, len(snapshot.actions))
    audit.check("saved_environment_task_contract",
                required == observed == (15, 15, 3, 100, 30, 52, False, 7),
                {"width_height_agents_steps_clutter_layout_competitive_actions": list(observed)})
    audit.check("saved_layout_matches_fixed_seed_52",
                np.array_equal(env._static_grid.encode(), snapshot._static_grid.encode()),
                "The saved static walls/goal equal the declared reference map.")
    positions = [(x, y) for x in range(15) for y in range(15)
                 if env.grid.get(x, y) is not None and env.grid.get(x, y).type == "goal"]
    audit.check("single_goal_at_13_13", positions == [(13, 13)], {"goal_positions": positions})
    env.step_count = 1
    first_reward = env._reward()
    env.step_count = 100
    last_reward = env._reward()
    audit.check("unmodified_goal_reward_function",
                math.isclose(first_reward, 0.991) and math.isclose(last_reward, 0.1),
                {"reward_step_1": first_reward, "reward_step_100": last_reward,
                 "formula": "1 - 0.9 * step_count / max_steps"})
    env.close()
    snapshot.close()


def audit_evaluation(audit, run_dir, expected_count):
    summary = audit.json(run_dir / "evaluation/summary.json")
    rows = audit.csv(run_dir / "evaluation/episodes.csv")
    audit.check("evaluation_uses_final_checkpoint",
                summary["training_completed_episodes"] == EXPECTED_EPISODES and
                summary["training_global_step"] == EXPECTED_STEPS and
                summary["environment"] == "MultiGrid-Cluttered-Fixed-15x15" and
                summary["layout_seed"] == 52,
                {key: summary.get(key) for key in ("training_completed_episodes", "training_global_step", "environment", "layout_seed")})
    seed_start, seed_end = map(int, summary["held_out_seed_range"])
    expected_seeds = list(range(seed_start, seed_end + 1))
    audit.check("held_out_seed_count", len(expected_seeds) == expected_count,
                {"range": [seed_start, seed_end], "required_count": expected_count})
    audit.check("only_expected_evaluation_policies", set(row["policy"] for row in rows) == set(POLICIES),
                {"policies": sorted(set(row["policy"] for row in rows))})
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        seeds = [int(row["episode_seed"]) for row in selected]
        values = np.array([[float(row[f"agent_{i}_return"]) for i in range(3)] for row in selected])
        collective = np.array([float(row["collective_return"]) for row in selected])
        audit.check(f"evaluation_{policy}_matched_unique_seeds",
                    len(selected) == expected_count and seeds == expected_seeds,
                    {"episodes": len(selected), "unique_seeds": len(set(seeds))})
        audit.check(f"evaluation_{policy}_returns",
                    len(selected) > 0 and np.isfinite(values).all() and np.isfinite(collective).all() and
                    np.allclose(values.sum(axis=1), collective, rtol=0, atol=1e-9) and
                    all(int(row["episode_length"]) == 100 for row in selected),
                    "Finite rewards, collective=sum(individual), and 100-step evaluation episodes.")
        recorded = summary["policies"][policy]["collective_return"]
        audit.check(f"evaluation_{policy}_summary_matches_csv",
                    recorded["n_episodes"] == expected_count and
                    math.isclose(recorded["mean"], float(collective.mean()), abs_tol=1e-10),
                    {"reported_mean": recorded["mean"], "csv_mean": float(collective.mean())})
    return summary, rows


def audit_media(audit, run_dir, summary, rows):
    import imageio.v2 as imageio
    from PIL import Image
    seed = int(summary["held_out_seed_range"][0])
    for policy in ("stochastic", "deterministic"):
        prefix = run_dir / "evaluation" / f"trajectory_{policy}_seed{seed}"
        trajectory = audit.json(prefix.with_suffix(".json"))
        steps = trajectory["trajectory"]
        rewards = np.array([step["rewards"] for step in steps])
        cumulative = np.array([step["cumulative_returns"] for step in steps])
        selected = next(row for row in rows if row["policy"] == policy and int(row["episode_seed"]) == seed)
        expected_return = np.array([float(selected[f"agent_{i}_return"]) for i in range(3)])
        audit.check(f"{policy}_trajectory_data_alignment",
                    len(steps) == 100 and trajectory["episode_seed"] == seed and
                    trajectory["policy"] == policy and rewards.shape == (100, 3) and
                    np.allclose(cumulative, rewards.cumsum(axis=0), rtol=0, atol=1e-9) and
                    np.allclose(cumulative[-1], expected_return, rtol=0, atol=1e-9) and
                    [step["step"] for step in steps] == list(range(100)) and
                    all(len(step["actions"]) == len(step["positions_before"]) == 3 for step in steps),
                    "The first evaluation seed has 100 aligned actions, positions, rewards and individual cumulative returns.")
        with imageio.get_reader(audit.artifact(prefix.with_suffix(".mp4"))) as video:
            count = video.count_frames()
            first, last = video.get_data(0), video.get_data(count - 1)
        audit.check(f"{policy}_video_decodable",
                    count >= 100 and first.shape == last.shape and first.shape[:2] == (816, 1280),
                    {"frames": count, "shape": list(first.shape)})
        with Image.open(audit.artifact(prefix.with_suffix(".gif"))) as gif:
            audit.check(f"{policy}_animated_preview", gif.n_frames > 1,
                        {"frames": gif.n_frames, "size": list(gif.size)})
        with Image.open(audit.artifact(prefix.with_suffix(".png"))) as screenshot:
            screenshot.verify()
        audit.check(f"{policy}_final_frame_png", True, "Final trajectory image is decodable.")


def audit_figures(audit, run_dir, evaluation_count):
    from PIL import Image
    for stem in ("learning_curve", "evaluation_comparison"):
        for suffix in (".png", ".pdf", ".svg"):
            path = audit.artifact(run_dir / "figures" / (stem + suffix))
            if suffix == ".png":
                with Image.open(path) as image:
                    image.verify()
            elif suffix == ".pdf" and not path.read_bytes().startswith(b"%PDF-"):
                raise ValueError(f"Invalid PDF header: {path}")
            elif suffix == ".svg" and "<svg" not in path.read_text():
                raise ValueError(f"Invalid SVG: {path}")
        notes = audit.json(run_dir / "figures" / (stem + ".notes.json"))
        valid = (notes["input_rows"] == EXPECTED_EPISODES and notes["measured_episodes"] == EXPECTED_EPISODES
                 and notes["target_reached"]) if stem == "learning_curve" else (
                    notes["input_rows"] == 3 * evaluation_count and
                    notes["episodes_per_policy"] == [evaluation_count] * 3)
        audit.check(f"{stem}_complete_measured_data", valid and notes["excluded_rows"] == 0, notes)


def audit_submission(run_dir, evaluation_episodes=200):
    run_dir = Path(run_dir).resolve()
    audit = Audit()
    config = audit.attempt("load_config", lambda: audit.json(run_dir / "config.json"))
    if config is not None:
        audit.check("assignment_training_configuration",
                    config.get("total_episodes") == EXPECTED_EPISODES and
                    config.get("env_id") == "MultiGrid-Cluttered-Fixed-15x15" and config.get("layout_seed") == 52,
                    {key: config.get(key) for key in ("total_episodes", "env_id", "seed", "layout_seed", "num_envs", "wandb_mode")})
    rows = audit.attempt("load_training_csv", lambda: audit.csv(run_dir / "episodes.csv"))
    if rows:
        audit.attempt("training_csv_integrity", lambda: audit_training_rows(audit, rows))
    summary = audit.attempt("load_training_summary", lambda: audit.json(run_dir / "summary.json"))
    if summary is not None:
        audit.check("training_summary_complete",
                    summary.get("status") == "complete" and summary.get("completed_episodes") == EXPECTED_EPISODES and
                    summary.get("joint_environment_steps") == EXPECTED_STEPS and summary.get("agent_decisions") == 3 * EXPECTED_STEPS,
                    summary)
        if rows:
            def check_last_window():
                window = int(summary["last_window_episodes"])
                measured = np.mean([float(row["collective_return"]) for row in rows[-window:]])
                audit.check("training_summary_last_window_matches_csv", window > 0 and
                            math.isclose(measured, summary["last_window_collective_return"], abs_tol=1e-9),
                            {"window": window, "measured_mean": float(measured)})
            audit.attempt("training_summary_last_window_integrity", check_last_window)
    final = audit.attempt("load_final_checkpoint", lambda: audit_checkpoint(audit, run_dir / "checkpoints/final.pt", "final"))
    latest = audit.attempt("load_latest_checkpoint", lambda: audit_checkpoint(audit, run_dir / "checkpoints/latest.pt", "latest"))
    if final and latest:
        audit.check("final_and_latest_policy_weights_match", final[1] == latest[1],
                    "Both final checkpoint files contain the same three completed policies.")
    if final and config:
        audit.check("checkpoint_config_matches_run_config", final[0]["config"] == config,
                    "Final checkpoint and saved run configuration agree exactly.")
    if final:
        audit.attempt("environment_contract_integrity", lambda: audit_environment_contract(audit, final[0]))
    evaluation = audit.attempt("evaluation_artifacts_integrity",
                               lambda: audit_evaluation(audit, run_dir, evaluation_episodes))
    if evaluation:
        if config and "seed" in config and "num_envs" in config:
            start, end = evaluation[0]["held_out_seed_range"]
            training_seeds = set(range(int(config["seed"]),
                                       int(config["seed"]) + int(config["num_envs"])))
            audit.check("evaluation_rng_seeds_held_out",
                        not training_seeds.intersection(range(int(start), int(end) + 1)),
                        "Evaluation RNG seeds are disjoint from training environment seeds; the map is intentionally shared.")
        audit.attempt("trajectory_media_integrity", lambda: audit_media(audit, run_dir, *evaluation))
    else:
        audit.check("trajectory_media_integrity", False, "Cannot validate media without complete evaluation data.")
    audit.attempt("figure_artifacts_integrity", lambda: audit_figures(audit, run_dir, evaluation_episodes))
    wandb_files = [path for path in (run_dir / "wandb").glob("*/run-*.wandb") if path.stat().st_size > 0]
    audit.check("local_wandb_run_record", bool(wandb_files), {"files": [str(path) for path in wandb_files]})
    for path in wandb_files:
        audit.attempt("wandb_record_readable", lambda path=path: audit.artifact(path))
    for relative_path, expected_hash in REVIEWED_ENVIRONMENT_SHA256.items():
        path = audit.attempt("reviewed_source_exists", lambda relative_path=relative_path:
                             audit.artifact(PROJECT_ROOT / relative_path))
        if path:
            observed_hash = audit.artifacts[str(path)]["sha256"]
            audit.check(f"reviewed_source_unchanged:{relative_path}", observed_hash == expected_hash,
                        {"reviewed_sha256": expected_hash, "observed_sha256": observed_hash})
    passed = all(check["passed"] for check in audit.checks)
    return {
        "schema_version": 1, "status": "pass" if passed else "fail",
        "scope": "Local completed training, checkpoints, evaluation and presentation artifacts",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir),
        "required_training_episodes": EXPECTED_EPISODES,
        "required_joint_environment_steps": EXPECTED_STEPS,
        "required_evaluation_episodes_per_policy": evaluation_episodes,
        "passed_checks": sum(check["passed"] for check in audit.checks),
        "failed_checks": sum(not check["passed"] for check in audit.checks),
        "checks": audit.checks, "artifacts": audit.artifacts,
        "declared_environment_changes": [
            "Replaced wall-clock placement and global action-order RNG with reproducible independent streams.",
            "Kept fixed layout_seed=52, 15x15 grid, 30 clutter walls, goal at(13,13), three agents, seven actions, 5x5 views and100-step timeout.",
            "Kept individual reward1-0.9*step_count/100 and immediate goal respawn; sticky success flags are not terminations.",
            "Cached static fields of view and overlaid visible agents; trajectory/view parity tests cover the reference implementation.",
            "Added timeout metadata, NumPy array observations and portable RNG serialization; no reward shaping or centralized inputs added.",
        ],
        "limitations": [
            "One training seed and one fixed map: evaluation intervals describe episode sampling, not independent-training-seed or new-map generalization.",
            "This local audit does not prove W&B cloud synchronization or creation of a remote repository fork; authentication/publishing must be verified separately.",
            "Video decoding, timestamps and reward data are checked automatically; visual readability still needs human inspection.",
            "Environment dimensions/layout/reward endpoints are checked here; behavioral equivalence also relies on the separate regression test suite.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/ippo_seed1"))
    parser.add_argument("--evaluation-episodes", type=int, choices=(100, 200), default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_submission(args.run_dir, args.evaluation_episodes)
    output = args.output or args.run_dir / "submission_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(f"{result['status'].upper()}: {result['passed_checks']} checks passed, {result['failed_checks']} failed")
    for check in result["checks"]:
        if not check["passed"]:
            print(f"  FAIL {check['check']}: {check['detail']}")
    print(f"Audit JSON: {output.resolve()}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
