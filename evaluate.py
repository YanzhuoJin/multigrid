"""Reproducible held-out evaluation and fully annotated MultiGrid trajectories.

The default video is the first stochastic-policy evaluation episode, chosen
before seeing its return. All policies use the same held-out episode seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from env_adapter import make_env
from ippo_agent import encode_observations, load_checkpoint


def summarize(values):
    """Episode-level Monte Carlo uncertainty, not training-seed uncertainty."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Summary requires non-empty finite observations")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else None
    sem = std / math.sqrt(len(values)) if std is not None else None
    return {
        "n_episodes": int(len(values)), "mean": mean, "sample_std": std,
        "standard_error": sem,
        "normal_approx_95_ci": [mean - 1.96 * sem, mean + 1.96 * sem]
        if sem is not None else None,
    }


def proportion_summary(values):
    """Wilson 95% confidence interval for binary episode success."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isin(values, [0, 1]).all():
        raise ValueError("Success observations must be non-empty binary values")
    n, p, z = len(values), float(values.mean()), 1.96
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z / denominator * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {"n_episodes": n, "rate": p, "wilson_95_ci": [center - radius, center + radius]}


def action_names(env, actions):
    return ["finished" if action is None else env.actions(int(action)).name for action in actions]


def _font(size, bold=False):
    from PIL import ImageFont
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def compose_frame(env, obs, actions, history, seed, policy, training_episodes):
    """Show s_t, the proposed a_t, and returns earned before a_t is applied."""
    from PIL import Image, ImageDraw
    from envs.gym_multigrid.multigrid import AGENT_COLOURS

    width, height = 1280, 816
    frame = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(frame)
    regular, small, title = _font(19), _font(16), _font(26, True)
    colors = [tuple(map(int, color)) for color in AGENT_COLOURS]
    step = len(history)
    n_agents = env.n_agents
    cumulative = np.cumsum(np.asarray(history), axis=0) if history else np.zeros((1, n_agents))
    returns = cumulative[-1]
    previous = np.asarray(history[-1]) if history else np.zeros(n_agents)

    algorithm = "Independent PPO" if n_agents > 1 else "PPO"
    draw.text((24, 16), f"{algorithm} | MultiGrid trajectory", fill="#17233d", font=title)
    draw.text((24, 52), f"{policy} policy | held-out seed {seed} | step {step}/{env.max_steps} | trained {training_episodes:,} episodes", fill="#40506c", font=regular)
    draw.text((24, 89), "Full environment state", fill="#17233d", font=regular)
    full = Image.fromarray(env.render(mode="rgb_array", highlight=False, tile_size=32))
    full.thumbnail((512, 512), Image.Resampling.NEAREST)
    frame.paste(full, (24, 122))
    draw.text((24, 654), f"Collective cumulative return: {float(returns.sum()):.3f}", fill="#17233d", font=regular)
    draw.text((24, 685), "Green square = goal; gray cells = obstacles", fill="#40506c", font=small)
    draw.text((24, 715), "Selected in advance: first evaluation episode.", fill="#40506c", font=small)
    ending_rule = "Goal success ends the episode." if env.competitive else "Agents respawn after reaching the goal."
    draw.text((24, 741), ending_rule, fill="#40506c", font=small)
    draw.text((24, 781), "View and action correspond to current state; rewards include completed transitions only.", fill="#40506c", font=small)

    card_width = min(220, 684 // n_agents)
    names = action_names(env, actions)
    for agent_id in range(n_agents):
        x = 566 + agent_id * card_width
        color = colors[agent_id % len(colors)]
        draw.text((x, 89), f"Agent {agent_id} | own view", fill=color, font=regular)
        partial = Image.fromarray(env.get_obs_render(np.asarray(obs["image"])[agent_id], tile_size=32))
        partial.thumbnail((card_width - 12, 208), Image.Resampling.NEAREST)
        frame.paste(partial, (x, 122))
        draw.text((x, 344), f"Next: {names[agent_id]}", fill="#17233d", font=regular)
        draw.text((x, 377), f"Last reward: {previous[agent_id]:.3f}", fill="#40506c", font=small)
        draw.text((x, 406), f"Return: {returns[agent_id]:.3f}", fill=color, font=regular)

    draw.text((566, 461), "Cumulative individual rewards", fill="#17233d", font=regular)
    left, top, right, bottom = 610, 504, 1228, 713
    ymax = max(1.0, float(cumulative.max()) * 1.1)
    draw.line([(left, top), (left, bottom), (right, bottom)], fill="#7b8595", width=2)
    for tick in range(5):
        y = bottom - (bottom - top) * tick / 4
        draw.line([(left, y), (right, y)], fill="#e7ebf1", width=1)
        draw.text((566, y - 8), f"{ymax * tick / 4:.1f}", fill="#40506c", font=small)
    for agent_id in range(n_agents):
        values = np.r_[0.0, cumulative[:, agent_id]] if history else [0.0]
        points = [(left + (right - left) * t / env.max_steps,
                   bottom - (bottom - top) * float(value) / ymax)
                  for t, value in enumerate(values)]
        color = colors[agent_id % len(colors)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        px, py = points[-1]
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
    draw.text((left - 4, bottom + 9), "0", fill="#40506c", font=small)
    draw.text((right - 25, bottom + 9), str(env.max_steps), fill="#40506c", font=small)
    draw.text((877, bottom + 30), "Environment step", fill="#40506c", font=small)
    return np.asarray(frame)


def run_episode(agents, config, seed, policy, video_prefix=None, fps=8):
    """Evaluate one episode without changing any model parameters."""
    import imageio.v2 as imageio

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    action_rng = np.random.default_rng(seed + 31337)
    env = make_env(config.get("env_id", "MultiGrid-Cluttered-Fixed-15x15"),
                   seed=seed, layout_seed=config.get("layout_seed", 52))
    obs = env.reset()
    returns = np.zeros(env.n_agents, dtype=np.float64)
    history, trajectory, frames = [], [], []
    video_writer = None
    if video_prefix is not None:
        video_prefix = Path(video_prefix)
        video_prefix.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_prefix.with_suffix(".mp4")),
                                         fps=fps, codec="libx264", quality=8,
                                         macro_block_size=16, pixelformat="yuv420p")
    try:
        done = False
        while not done:
            if policy == "random":
                actions = action_rng.integers(len(env.actions), size=env.n_agents).tolist()
            else:
                encoded = torch.as_tensor(encode_observations(obs), dtype=torch.float32)
                with torch.inference_mode():
                    actions = [int(agent.act(encoded[i:i + 1], deterministic=policy == "deterministic").item())
                               for i, agent in enumerate(agents)]
            if video_writer is not None:
                frame = compose_frame(env, obs, actions, history, seed, policy,
                                      config.get("completed_episodes", 0))
                video_writer.append_data(frame)
                frames.append(frame)
            positions = np.asarray(env.agent_pos).tolist()
            directions = np.asarray(env.agent_dir).tolist()
            next_obs, rewards, done, _ = env.step(actions)
            rewards = np.asarray(rewards, dtype=np.float64)
            returns += rewards
            history.append(rewards.tolist())
            trajectory.append({"step": len(history) - 1, "positions_before": positions,
                               "directions_before": directions, "actions": actions,
                               "action_names": action_names(env, actions),
                               "rewards": rewards.tolist(), "cumulative_returns": returns.tolist()})
            obs = next_obs
        if video_writer is not None:
            final_frame = compose_frame(env, obs, [None] * env.n_agents, history, seed,
                                        policy, config.get("completed_episodes", 0))
            for _ in range(fps):
                video_writer.append_data(final_frame)
                frames.append(final_frame)
            from PIL import Image
            Image.fromarray(final_frame).save(video_prefix.with_suffix(".png"))
            gif_frames = [Image.fromarray(frame).resize((960, 612)) for frame in frames]
            gif_frames[0].save(video_prefix.with_suffix(".gif"), save_all=True,
                               append_images=gif_frames[1:], duration=round(1000 / fps), loop=0)
            video_prefix.with_suffix(".json").write_text(json.dumps({
                "policy": policy, "episode_seed": seed,
                "selection": "First evaluation episode, selected before evaluation (not highest-return).",
                "time_alignment": "Frame t shows state s_t and next action a_t; cumulative returns cover transitions 0 to t-1.",
                "trajectory": trajectory,
            }, indent=2) + "\n")
        row = {"policy": policy, "episode_seed": seed, "episode_length": len(history),
               "collective_return": float(returns.sum()),
               "any_agent_success": int(np.any(env.done)),
               "all_agents_success": int(np.all(env.done))}
        for i in range(env.n_agents):
            row[f"agent_{i}_return"] = float(returns[i])
            row[f"agent_{i}_success"] = int(env.done[i])
        return row
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()


def evaluate(checkpoint, episodes=100, seed=1_000_000, output=None, record=True, fps=8):
    if episodes < 1 or fps < 1 or seed < 0:
        raise ValueError("episodes/fps must be positive, and seed must be nonnegative")
    agents, payload = load_checkpoint(checkpoint, device="cpu")
    for agent in agents:
        agent.eval()
    torch.set_num_threads(1)
    config = dict(payload["config"])
    config["completed_episodes"] = int(payload["completed_episodes"])
    probe = make_env(config.get("env_id", "MultiGrid-Cluttered-Fixed-15x15"),
                     seed=seed, layout_seed=config.get("layout_seed", 52))
    competitive = bool(getattr(probe, "competitive", False))
    fixed_layout = bool(getattr(probe, "fixed_environment", False))
    probe.close()
    output = Path(output or Path(checkpoint).resolve().parent.parent / "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    policies = ["stochastic", "deterministic", "random"]
    rows = []
    for policy in policies:
        for episode in range(episodes):
            video_prefix = output / f"trajectory_{policy}_seed{seed}" if record and episode == 0 and policy != "random" else None
            row = run_episode(agents, config, seed + episode, policy, video_prefix, fps)
            row["evaluation_episode"] = episode + 1
            rows.append(row)
        print(f"{policy}: {episodes} episodes, collective return = "
              f"{np.mean([row['collective_return'] for row in rows if row['policy'] == policy]):.4f}", flush=True)
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "training_completed_episodes": int(payload["completed_episodes"]),
        "training_global_step": int(payload["global_step"]),
        "environment": config.get("env_id"), "layout_seed": config.get("layout_seed", 52),
        "held_out_seed_range": [seed, seed + episodes - 1],
        "protocol": ("Same fixed obstacle layout as training; " if fixed_layout else "Same environment configuration as training; ") + "held-out seeds control starting positions and environment randomness. One training run; confidence intervals measure evaluation-episode uncertainty only.",
        "success_definition": "Agent reached a goal at least once during the joint episode. " + ("The first success terminates the episode." if competitive else "Agents respawn and may earn multiple goal rewards."),
        "competitive": competitive,
        "video_selection": "First episode for stochastic and deterministic policies, specified before results; no success filtering.",
        "policies": {}, "paired_improvement_over_random": {},
    }
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        metrics = {"collective_return": summarize([row["collective_return"] for row in selected])}
        for metric in ["any_agent_success", "all_agents_success"] + [f"agent_{i}_success" for i in range(len(agents))]:
            metrics[metric] = proportion_summary([row[metric] for row in selected])
        for i in range(len(agents)):
            metrics[f"agent_{i}_return"] = summarize([row[f"agent_{i}_return"] for row in selected])
        summary["policies"][policy] = metrics
    baseline = np.array([row["collective_return"] for row in rows if row["policy"] == "random"])
    for policy in policies[:-1]:
        returns = np.array([row["collective_return"] for row in rows if row["policy"] == policy])
        summary["paired_improvement_over_random"][policy] = summarize(returns - baseline)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    if args.episodes < 1 or args.fps < 1 or args.seed < 0:
        parser.error("episodes/fps must be positive, and seed must be nonnegative")
    evaluate(args.checkpoint, args.episodes, args.seed, args.output, not args.no_video, args.fps)


if __name__ == "__main__":
    main()
