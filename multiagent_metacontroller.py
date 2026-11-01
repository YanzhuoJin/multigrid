"""Train independent PPO policies in simultaneous-action MultiGrid episodes."""
from __future__ import annotations

import csv
import json
import os
import pickle
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from env_adapter import make_env
from ippo_agent import (AgentRollout, IndependentPolicyBatch, IPPOAgent,
                        encode_batch, update_policy)


@dataclass
class TrainingConfig:
    env_id: str = "MultiGrid-Cluttered-Fixed-15x15"
    seed: int = 1
    layout_seed: int = 52
    total_episodes: int = 100000
    num_envs: int = 16
    num_steps: int = 128
    hidden_size: int = 64
    learning_rate: float = 0.0003
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    normalize_advantage: bool = True
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    torch_threads: int = 1
    device: str = "cpu"
    log_every: int = 100
    checkpoint_every: int = 10000
    wandb_mode: str = "offline"
    wandb_project: str = "socialrl-ippo"
    wandb_entity: str = ""
    run_dir: str = "results/ippo_seed1"


class MultiAgent:
    """Coordinates separate agents; collective reward is used for reporting only."""

    def __init__(self, config: TrainingConfig, resume: str | None = None):
        self.config = config
        for field in ("total_episodes", "num_envs", "num_steps", "hidden_size", "num_minibatches",
                      "update_epochs", "torch_threads", "log_every", "checkpoint_every"):
            if getattr(config, field) <= 0:
                raise ValueError(f"{field} must be positive")
        torch.set_num_threads(config.torch_threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        self.run_dir = Path(config.run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        if (self.run_dir / "episodes.csv").exists() and not resume:
            raise FileExistsError(f"{self.run_dir / 'episodes.csv'} already exists; use --resume or a new --run-dir")
        self.completed_episodes = 0
        self.global_step = 0
        self.update = 0
        self.elapsed_before = 0.0
        self.started_at = time.monotonic()
        self.recent_returns = deque(maxlen=config.log_every)
        self.recent_successes = deque(maxlen=config.log_every)
        self.num_envs = min(config.num_envs, config.total_episodes)
        self.envs = [make_env(config.env_id, seed=config.seed + i,
                              layout_seed=config.layout_seed) for i in range(self.num_envs)]
        self.observations = [env.reset() for env in self.envs]
        self.n_agents = self.envs[0].n_agents
        self.model_spec = {
            "obs_shape": tuple(np.asarray(self.observations[0]["image"]).shape[-3:]),
            "n_actions": len(self.envs[0].actions), "hidden_size": config.hidden_size,
        }
        self.agents = [IPPOAgent(**self.model_spec).to(self.device) for _ in range(self.n_agents)]
        self.optimizers = [torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
                           for agent in self.agents]
        self.episode_returns = np.zeros((self.num_envs, self.n_agents), dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.active_envs = np.ones(self.num_envs, dtype=bool)
        self.started_episodes = self.num_envs
        self.wandb_id = None
        if resume:
            self.load_models(resume)
        self.policies = IndependentPolicyBatch(self.agents)
        self.rollouts = [AgentRollout.allocate(config.num_steps, self.num_envs,
                                              agent.input_size, self.device) for agent in self.agents]
        self.wandb = None
        self._init_logging(resume)

    @property
    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.started_at

    def _init_logging(self, resume):
        self.csv_path = self.run_dir / "episodes.csv"
        source_csv = self.csv_path
        if resume and not source_csv.exists():
            source_csv = Path(resume).resolve().parent.parent / "episodes.csv"
        # Discard rows newer than a checkpoint after an interrupted training run.
        if resume and source_csv.exists():
            with source_csv.open(newline="") as source:
                reader = csv.DictReader(source)
                fields = reader.fieldnames
                rows = [row for row in reader if int(row["episode"]) <= self.completed_episodes]
            with self.csv_path.open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        elif self.csv_path.exists():
            raise FileExistsError(f"{self.csv_path} already exists; use --resume or a new --run-dir")
        (self.run_dir / "config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")
        exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self.csv_file = self.csv_path.open("a", newline="", buffering=1)
        fields = ["episode", "global_step", "env_index", "episode_length", "collective_return"]
        fields += [f"agent_{i}_return" for i in range(self.n_agents)]
        fields += [f"agent_{i}_success" for i in range(self.n_agents)]
        fields += ["elapsed_seconds"]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fields)
        if not exists:
            self.csv_writer.writeheader()
        updates_path = self.run_dir / "updates.jsonl"
        source_updates = updates_path
        if resume and not source_updates.exists():
            source_updates = Path(resume).resolve().parent.parent / "updates.jsonl"
        if resume and source_updates.exists():
            saved_updates = [line for line in source_updates.read_text().splitlines()
                             if json.loads(line)["update"] <= self.update]
            updates_path.write_text("\n".join(saved_updates) + ("\n" if saved_updates else ""))
        self.updates_file = updates_path.open("a", buffering=1)
        if self.config.wandb_mode != "disabled":
            import wandb
            self.wandb = wandb.init(
                project=self.config.wandb_project, entity=self.config.wandb_entity or None,
                mode=self.config.wandb_mode, dir=str(self.run_dir), config=asdict(self.config),
                name=self.run_dir.name + (f"_resume_{self.completed_episodes}" if resume else ""),
                settings=wandb.Settings(console="off", quiet=True),
            )
            self.wandb_id = self.wandb.id
            self.wandb.define_metric("episode")
            self.wandb.define_metric("train/*", step_metric="episode")
            for i in range(self.n_agents):
                self.wandb.define_metric(f"agent_{i}/*", step_metric="episode")

    def _record_episode(self, env_index):
        self.completed_episodes += 1
        returns = self.episode_returns[env_index].copy()
        successes = np.asarray(self.envs[env_index].done, dtype=bool)
        row = {"episode": self.completed_episodes, "global_step": self.global_step,
               "env_index": env_index, "episode_length": int(self.episode_lengths[env_index]),
               "collective_return": float(returns.sum()), "elapsed_seconds": round(self.elapsed, 3)}
        row.update({f"agent_{i}_return": float(returns[i]) for i in range(self.n_agents)})
        row.update({f"agent_{i}_success": int(successes[i]) for i in range(self.n_agents)})
        self.csv_writer.writerow(row)
        self.recent_returns.append(returns)
        self.recent_successes.append(successes)
        if self.completed_episodes % self.config.log_every == 0:
            mean_returns = np.mean(self.recent_returns, axis=0)
            mean_successes = np.mean(self.recent_successes, axis=0)
            metrics = {"episode": self.completed_episodes, "global_step": self.global_step,
                       "train/collective_return": float(mean_returns.sum()),
                       "train/collective_return_last": float(returns.sum()),
                       "train/episode_length": int(self.episode_lengths[env_index]),
                       "train/steps_per_second": self.global_step / max(self.elapsed, 1e-6)}
            for i in range(self.n_agents):
                metrics[f"agent_{i}/return"] = float(mean_returns[i])
                metrics[f"agent_{i}/success_rate"] = float(mean_successes[i])
            if self.wandb:
                self.wandb.log(metrics)
            print(f"episodes={self.completed_episodes}/{self.config.total_episodes} "
                  f"steps={self.global_step} collective_return_mean={mean_returns.sum():.4f} "
                  f"SPS={metrics['train/steps_per_second']:.0f}", flush=True)
        self.episode_returns[env_index] = 0
        self.episode_lengths[env_index] = 0
        if self.started_episodes < self.config.total_episodes:
            self.observations[env_index] = self.envs[env_index].reset()
            self.started_episodes += 1
        else:
            self.active_envs[env_index] = False

    def train(self):
        next_checkpoint = (self.completed_episodes // self.config.checkpoint_every + 1) * self.config.checkpoint_every
        try:
            while self.completed_episodes < self.config.total_episodes:
                self.update += 1
                if self.config.anneal_lr:
                    fraction = max(0.0, 1 - self.completed_episodes / self.config.total_episodes)
                    for optimizer in self.optimizers:
                        optimizer.param_groups[0]["lr"] = self.config.learning_rate * fraction
                length = 0
                for step in range(self.config.num_steps):
                    if not self.active_envs.any():
                        break
                    valid = self.active_envs.copy()
                    encoded = torch.as_tensor(encode_batch(self.observations), device=self.device)
                    actions, logprobs, values = self.policies(encoded)
                    for i, rollout in enumerate(self.rollouts):
                        rollout.observations[step].copy_(encoded[:, i])
                        rollout.actions[step].copy_(actions[:, i])
                        rollout.logprobs[step].copy_(logprobs[:, i])
                        rollout.values[step].copy_(values[:, i])
                        rollout.valid[step].copy_(torch.as_tensor(valid, device=self.device))
                    action_array = actions.cpu().numpy()
                    rewards = np.zeros((self.num_envs, self.n_agents), dtype=np.float32)
                    boundaries = np.broadcast_to(~valid[:, None], rewards.shape).copy()
                    timeouts = np.zeros_like(boundaries)
                    ended = []
                    for j in np.flatnonzero(valid):
                        observation, reward, done, info = self.envs[j].step(action_array[j].tolist())
                        self.observations[j] = observation
                        self.global_step += 1  # joint environment transitions, not agent actions
                        rewards[j] = reward
                        self.episode_returns[j] += reward
                        self.episode_lengths[j] += 1
                        # env.done is sticky success after respawn, not termination.
                        agent_terminated = np.broadcast_to(
                            info.get("agent_terminated", False), (self.n_agents,))
                        boundaries[j] |= agent_terminated
                        if done:
                            truncated = bool(info.get("truncated", info.get("TimeLimit.truncated", False)))
                            boundaries[j] = True
                            timeouts[j] = truncated & ~agent_terminated
                            ended.append(j)
                    # Bootstrap with the actual final observation BEFORE reset.
                    if timeouts.any():
                        final_values = self.policies(
                            torch.as_tensor(encode_batch(self.observations), device=self.device), sample=False)
                        rewards += self.config.gamma * final_values.cpu().numpy() * timeouts
                    for i, rollout in enumerate(self.rollouts):
                        rollout.rewards[step].copy_(torch.as_tensor(rewards[:, i], device=self.device))
                        rollout.boundaries[step].copy_(torch.as_tensor(boundaries[:, i], device=self.device))
                    for j in ended:
                        self._record_episode(j)
                    length += 1
                bootstrap = self.policies(
                    torch.as_tensor(encode_batch(self.observations), device=self.device), sample=False)
                update_metrics = {"update": self.update, "episode": self.completed_episodes,
                                  "global_step": self.global_step}
                for i, (agent, optimizer, rollout) in enumerate(zip(self.agents, self.optimizers, self.rollouts)):
                    metrics = update_policy(agent, optimizer, rollout, length, bootstrap[:, i], self.config)
                    update_metrics[f"agent_{i}"] = metrics
                self.updates_file.write(json.dumps(update_metrics) + "\n")
                self.policies.refresh()
                if self.completed_episodes >= next_checkpoint:
                    self.save_checkpoint(self.run_dir / "checkpoints" / "latest.pt")
                    next_checkpoint += self.config.checkpoint_every
            final_path = self.run_dir / "checkpoints" / "final.pt"
            self.save_checkpoint(final_path)
            self.save_checkpoint(self.run_dir / "checkpoints" / "latest.pt")
            summary = {"status": "complete", "completed_episodes": self.completed_episodes,
                       "joint_environment_steps": self.global_step,
                       "agent_decisions": self.global_step * self.n_agents,
                       "updates": self.update, "elapsed_seconds": self.elapsed,
                       "last_window_episodes": len(self.recent_returns),
                       "last_window_collective_return": float(np.mean(self.recent_returns, axis=0).sum())
                       if self.recent_returns else None,
                       "checkpoint": str(final_path), "wandb_mode": self.config.wandb_mode}
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            if self.wandb:
                self.wandb.summary.update(summary)
            print(json.dumps(summary, indent=2), flush=True)
            return summary
        except KeyboardInterrupt:
            print("Interrupted. Resume from the most recent completed-update checkpoint.", flush=True)
            raise
        finally:
            self.csv_file.close()
            self.updates_file.close()
            if self.wandb:
                self.wandb.finish()
            for env in self.envs:
                env.close()

    def save_checkpoint(self, path):
        payload = {
            "format_version": 1, "algorithm": "independent_ppo", "config": asdict(self.config),
            "model_spec": self.model_spec, "agents": [a.state_dict() for a in self.agents],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "completed_episodes": self.completed_episodes, "global_step": self.global_step,
            "update": self.update, "elapsed_seconds": self.elapsed,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state()},
            "runtime": {"environments": [pickle.dumps(env) for env in self.envs],
                        "observations": self.observations, "episode_returns": self.episode_returns,
                        "episode_lengths": self.episode_lengths, "active_envs": self.active_envs,
                        "started_episodes": self.started_episodes,
                        "recent_returns": list(self.recent_returns),
                        "recent_successes": list(self.recent_successes)},
        }
        if torch.cuda.is_available():
            payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        self.csv_file.flush()

    def load_models(self, model_path):
        payload = torch.load(model_path, map_location=self.device, weights_only=False)
        saved = payload["config"]
        for key in ("env_id", "layout_seed", "num_envs", "hidden_size"):
            if saved[key] != getattr(self.config, key):
                raise ValueError(f"Resume requires the original {key}={saved[key]!r}")
        if self.config.total_episodes < payload["completed_episodes"]:
            raise ValueError("--total-episodes cannot be below the checkpoint's completed episodes")
        for agent, weights in zip(self.agents, payload["agents"]):
            agent.load_state_dict(weights)
        for optimizer, state in zip(self.optimizers, payload["optimizers"]):
            optimizer.load_state_dict(state)
        self.completed_episodes = payload["completed_episodes"]
        self.global_step = payload["global_step"]
        self.update = payload["update"]
        self.elapsed_before = payload["elapsed_seconds"]
        runtime = payload["runtime"]
        for env in self.envs:
            env.close()
        self.envs = [pickle.loads(state) for state in runtime["environments"]]
        self.num_envs = len(self.envs)
        self.observations = runtime["observations"]
        self.episode_returns = runtime["episode_returns"]
        self.episode_lengths = runtime["episode_lengths"]
        self.active_envs = runtime["active_envs"]
        self.started_episodes = runtime["started_episodes"]
        self.recent_returns.extend(runtime["recent_returns"])
        self.recent_successes.extend(runtime["recent_successes"])
        for j in np.flatnonzero(~self.active_envs):
            if self.started_episodes < self.config.total_episodes:
                self.observations[j] = self.envs[j].reset()
                self.active_envs[j] = True
                self.started_episodes += 1
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        if "cuda" in payload["rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
