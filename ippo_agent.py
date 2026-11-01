"""Independent PPO policy and rollout utilities.

Each policy sees only its own encoded partial observation and heading. There is
no shared network, optimizer, trajectory, value target or reward across agents.
The PPO equations follow the supplied CleanRL-style ppo.py implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


# MiniGrid image channels are categorical IDs, not RGB intensities.
OBJECT_CLASSES, COLOR_CLASSES, STATE_CLASSES = 11, 6, 4
CELL_FEATURES = OBJECT_CLASSES + COLOR_CLASSES + STATE_CLASSES
_objects, _colors, _states = np.indices((OBJECT_CLASSES, COLOR_CLASSES, STATE_CLASSES))
_CELL_LOOKUP = np.concatenate(
    (np.eye(OBJECT_CLASSES, dtype=np.float32)[_objects],
     np.eye(COLOR_CLASSES, dtype=np.float32)[_colors],
     np.eye(STATE_CLASSES, dtype=np.float32)[_states]), axis=-1
)
_DIRECTION_LOOKUP = np.eye(4, dtype=np.float32)


def encode_observations(observation: dict) -> np.ndarray:
    """Return [agents, features], or [envs, agents, features] for stacked input.

The visible agents' colors/headings are legitimate local image information;
other agents' *separate* views and heading observations are never provided.
"""
    images = np.asarray(observation["image"], dtype=np.int64)
    directions = np.asarray(observation["direction"], dtype=np.int64)
    leading_shape = images.shape[:-3]
    cells = _CELL_LOOKUP[images[..., 0], images[..., 1], images[..., 2]]
    return np.concatenate((cells.reshape(*leading_shape, -1),
                           _DIRECTION_LOOKUP[directions]), axis=-1)


def encode_batch(observations: list[dict]) -> np.ndarray:
    return encode_observations({
        "image": np.stack([observation["image"] for observation in observations]),
        "direction": np.stack([observation["direction"] for observation in observations]),
    })


def layer_init(layer: nn.Linear, std: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class IPPOAgent(nn.Module):
    """A single independent actor-critic; its encoder is shared only internally."""

    def __init__(self, obs_shape=(5, 5, 3), n_actions=7, hidden_size=64):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.input_size = int(np.prod(self.obs_shape[:2])) * CELL_FEATURES + 4
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(self.input_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(hidden_size, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, observations: torch.Tensor):
        features = self.encoder(observations)
        return self.actor(features), self.critic(features).squeeze(-1)

    def get_action_and_value(self, observations, action=None):
        logits, value = self(observations)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value

    @torch.no_grad()
    def act(self, observations, deterministic=False):
        logits, _ = self(observations)
        return logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()


class IndependentPolicyBatch:
    """Cache independent policy weights for fast batched rollout inference.

The leading dimension selects the agent. torch.bmm never mixes that dimension;
this is equivalent to calling each separate network and shares no parameters.
The copies are refreshed after every PPO update and are never optimized.
"""

    def __init__(self, agents: list[IPPOAgent]):
        self.agents = agents
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        self.weights = [torch.stack([a.encoder[i].weight for a in self.agents])
                        for i in (0, 2)]
        self.biases = [torch.stack([a.encoder[i].bias for a in self.agents])[:, None]
                       for i in (0, 2)]
        self.output_weights = torch.stack([
            torch.cat((a.actor.weight, a.critic.weight)) for a in self.agents])
        self.output_biases = torch.stack([
            torch.cat((a.actor.bias, a.critic.bias)) for a in self.agents])[:, None]

    @torch.no_grad()
    def __call__(self, observations, sample=True):
        # Incoming [env, agent, feature]; all outputs [env, agent].
        x = observations.transpose(0, 1)
        for weights, biases in zip(self.weights, self.biases):
            x = (torch.bmm(x, weights.transpose(1, 2)) + biases).tanh()
        output = torch.bmm(x, self.output_weights.transpose(1, 2)) + self.output_biases
        if not sample:
            return output[..., -1].transpose(0, 1)
        distribution = Categorical(logits=output[..., :-1])
        actions = distribution.sample()
        return (actions.transpose(0, 1), distribution.log_prob(actions).transpose(0, 1),
                output[..., -1].transpose(0, 1))


@dataclass
class AgentRollout:
    """Separate storage object allocated for one policy's own experience."""
    observations: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    boundaries: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def allocate(cls, steps, environments, features, device):
        shape = (steps, environments)
        return cls(torch.zeros((*shape, features), device=device),
                   torch.zeros(shape, dtype=torch.long, device=device),
                   torch.zeros(shape, device=device), torch.zeros(shape, device=device),
                   torch.zeros(shape, device=device),
                   torch.zeros(shape, dtype=torch.bool, device=device),
                   torch.zeros(shape, dtype=torch.bool, device=device))


def compute_gae(rewards, values, boundaries, next_value, gamma=0.99, gae_lambda=0.95):
    """GAE with per-transition boundaries (True cuts both bootstrap and trace).

Timeout rewards must already include gamma * V(final_observation), so their
trace is cut without losing the bootstrap. Actual terminations get no bonus.
The final next_value applies only to unfinished trajectories at rollout end.
"""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(next_value)
    for t in reversed(range(rewards.shape[0])):
        following_value = next_value if t == rewards.shape[0] - 1 else values[t + 1]
        continuation = (~boundaries[t]).to(values.dtype)
        delta = rewards[t] + gamma * following_value * continuation - values[t]
        last_advantage = delta + gamma * gae_lambda * continuation * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values


def update_policy(agent, optimizer, rollout, length, next_value, config):
    advantages, returns = compute_gae(
        rollout.rewards[:length], rollout.values[:length], rollout.boundaries[:length],
        next_value, config.gamma, config.gae_lambda)
    valid = rollout.valid[:length].reshape(-1)
    observations = rollout.observations[:length].flatten(0, 1)[valid]
    actions = rollout.actions[:length].flatten()[valid]
    old_logprobs = rollout.logprobs[:length].flatten()[valid]
    old_values = rollout.values[:length].flatten()[valid]
    advantages = advantages.flatten()[valid]
    returns = returns.flatten()[valid]
    size = actions.numel()
    if size == 0:
        return {}
    batch_size = max(1, (size + config.num_minibatches - 1) // config.num_minibatches)
    metrics = []
    for _ in range(config.update_epochs):
        indices = torch.randperm(size, device=observations.device)
        epoch_kl = []
        for start in range(0, size, batch_size):
            selected = indices[start:start + batch_size]
            _, logprobs, entropy, values = agent.get_action_and_value(
                observations[selected], actions[selected])
            logratio = logprobs - old_logprobs[selected]
            ratio = logratio.exp()
            advantage = advantages[selected]
            if config.normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
            policy_loss = torch.maximum(
                -advantage * ratio,
                -advantage * ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef)).mean()
            if config.clip_value_loss:
                clipped = old_values[selected] + (values - old_values[selected]).clamp(
                    -config.clip_coef, config.clip_coef)
                value_loss = 0.5 * torch.maximum(
                    (values - returns[selected]).square(),
                    (clipped - returns[selected]).square()).mean()
            else:
                value_loss = 0.5 * (values - returns[selected]).square().mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean().item()
                epoch_kl.append(approx_kl)
                metrics.append((policy_loss.item(), value_loss.item(), entropy.mean().item(),
                                approx_kl, ((ratio - 1).abs() > config.clip_coef).float().mean().item()))
        if config.target_kl and np.mean(epoch_kl) > config.target_kl:
            break
    names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
    result = dict(zip(names, np.mean(metrics, axis=0).tolist()))
    variance = returns.var(unbiased=False).item()
    result["explained_variance"] = (
        1 - (returns - old_values).var(unbiased=False).item() / variance if variance > 0 else 0.0)
    return result


def load_checkpoint(path: str | Path, device="cpu"):
    """Load a locally generated trusted checkpoint for evaluation or analysis."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    agents = [IPPOAgent(**payload["model_spec"]).to(device)
              for _ in payload["agents"]]
    for agent, weights in zip(agents, payload["agents"]):
        agent.load_state_dict(weights)
        agent.eval()
    return agents, payload
