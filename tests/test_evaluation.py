"""Evaluation must preserve model weights, pairing, and reward accounting."""
import csv
import json
from enum import IntEnum

import numpy as np
import pytest
import torch

import evaluate as evaluation
from ippo_agent import IPPOAgent
from scripts.plot_results import trailing_mean


class TinyActions(IntEnum):
    left = 0
    right = 1
    forward = 2


class TinyEpisode:
    n_agents = 3
    actions = TinyActions
    max_steps = 2

    def reset(self):
        self.t = 0
        self.done = [False] * 3
        self.agent_pos = [[1, 1], [2, 1], [3, 1]]
        self.agent_dir = [0, 1, 2]
        return {"image": np.zeros((3, 5, 5, 3)), "direction": self.agent_dir}

    def step(self, actions):
        self.t += 1
        rewards = [1.0, 0.5, 0.0] if self.t == 1 else [0.0, 0.25, 2.0]
        self.done = [True, True, self.t == 2]
        return {"image": np.zeros((3, 5, 5, 3)), "direction": self.agent_dir}, rewards, self.t == 2, {}

    def close(self):
        pass


def test_episode_accounting_uses_sum_and_individual_success(monkeypatch):
    monkeypatch.setattr(evaluation, "make_env", lambda *args, **kwargs: TinyEpisode())
    row = evaluation.run_episode([], {}, seed=27, policy="random")
    assert row["collective_return"] == 3.75
    assert [row[f"agent_{i}_return"] for i in range(3)] == [1.0, 0.75, 2.0]
    assert all(row[f"agent_{i}_success"] == 1 for i in range(3))
    assert row["episode_length"] == 2


def test_held_out_seed_replay_does_not_mutate_policy():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    agents = [IPPOAgent(hidden_size=8) for _ in range(3)]
    before = [{key: value.clone() for key, value in agent.state_dict().items()} for agent in agents]
    first = evaluation.run_episode(agents, {}, seed=1_000_011, policy="stochastic")
    second = evaluation.run_episode(agents, {}, seed=1_000_011, policy="stochastic")
    assert first == second
    assert first["episode_length"] == 100
    assert first["collective_return"] == sum(first[f"agent_{i}_return"] for i in range(3))
    for agent, original in zip(agents, before):
        assert all(torch.equal(original[key], value) for key, value in agent.state_dict().items())
        assert all(parameter.grad is None for parameter in agent.parameters())


def test_evaluation_pairs_all_policy_seeds_and_serializes_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "make_env", lambda *args, **kwargs: TinyEpisode())
    agents = [IPPOAgent(hidden_size=8) for _ in range(3)]
    payload = {"agents": [agent.state_dict() for agent in agents],
               "model_spec": {"obs_shape": (5, 5, 3), "n_actions": 3, "hidden_size": 8},
               "config": {"env_id": "MultiGrid-Cluttered-Fixed-15x15", "layout_seed": 52},
               "completed_episodes": 17, "global_step": 1700}
    # Saved policy action dimensions must match the supplied checkpoint metadata.
    payload["agents"] = [IPPOAgent(n_actions=3, hidden_size=8).state_dict() for _ in range(3)]
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    output = tmp_path / "evaluation"
    result = evaluation.evaluate(checkpoint, episodes=3, seed=1_000_000, output=output, record=False)
    with (output / "episodes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    for policy in ["stochastic", "deterministic", "random"]:
        assert [int(row["episode_seed"]) for row in rows if row["policy"] == policy] == [1_000_000, 1_000_001, 1_000_002]
        assert result["policies"][policy]["collective_return"]["mean"] == 3.75
    assert result["paired_improvement_over_random"]["stochastic"]["mean"] == 0.0
    assert json.loads((output / "summary.json").read_text()) == result
    assert not list(output.glob("*.mp4"))


def test_uncertainty_and_smoothing_edge_cases():
    assert evaluation.summarize([2])["normal_approx_95_ci"] is None
    assert evaluation.summarize([1, 2, 3])["mean"] == 2
    interval = evaluation.proportion_summary([0, 0, 0])["wilson_95_ci"]
    assert interval[0] == pytest.approx(0)
    assert 0 < interval[1] < 1
    np.testing.assert_allclose(trailing_mean([1, 3, 8, 12], 2), [1, 2, 5.5, 10])
    with pytest.raises(ValueError):
        evaluation.summarize([np.nan])
    with pytest.raises(ValueError):
        trailing_mean([1, 2], 0)
