"""Focused mathematical and end-to-end IPPO regression tests."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ippo_agent import (AgentRollout, IndependentPolicyBatch, IPPOAgent,
                        compute_gae, encode_observations, load_checkpoint)
from multiagent_metacontroller import MultiAgent, TrainingConfig


class AdvantageTests(unittest.TestCase):
    def test_gae_cuts_episode_boundary_without_leaking_next_episode(self):
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        values = torch.tensor([[0.5], [0.6], [0.7]])
        boundaries = torch.tensor([[False], [True], [False]])
        advantages, returns = compute_gae(rewards, values, boundaries,
                                          torch.tensor([0.8]), 0.9, 0.95)
        torch.testing.assert_close(advantages[:, 0], torch.tensor([2.237, 1.4, 3.02]))
        torch.testing.assert_close(returns, advantages + values)

    def test_timeout_bootstraps_final_observation_but_not_reset_observation(self):
        gamma = 0.9
        rewards = torch.tensor([[2.0 + gamma * 5.0], [0.0]])
        values = torch.tensor([[0.6], [999.0]])
        advantages, _ = compute_gae(rewards, values, torch.tensor([[True], [True]]),
                                    torch.tensor([1234.0]), gamma, 0.95)
        self.assertAlmostEqual(advantages[0].item(), 5.9, places=5)

    def test_respawn_continuation_retains_later_rewards(self):
        advantages, _ = compute_gae(torch.tensor([[1.0], [2.0]]), torch.zeros(2, 1),
                                    torch.tensor([[False], [True]]), torch.zeros(1), 1, 1)
        torch.testing.assert_close(advantages[:, 0], torch.tensor([3.0, 2.0]))


class IndependenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_observations_do_not_leak_other_agents_heading(self):
        observation = {"image": np.zeros((3, 5, 5, 3), dtype=np.uint8),
                       "direction": np.array([0, 1, 2])}
        original = encode_observations(observation)
        observation["direction"][1] = 3
        changed = encode_observations(observation)
        np.testing.assert_array_equal(original[0], changed[0])
        np.testing.assert_array_equal(original[2], changed[2])
        self.assertFalse(np.array_equal(original[1], changed[1]))

    def test_networks_and_rollouts_have_separate_storage(self):
        agents = [IPPOAgent() for _ in range(3)]
        pointers = [{p.data_ptr() for p in agent.parameters()} for agent in agents]
        self.assertFalse(pointers[0] & pointers[1])
        before = [p.detach().clone() for p in agents[1].parameters()]
        optimizer = torch.optim.Adam(agents[0].parameters(), lr=1e-3)
        logits, values = agents[0](torch.ones(4, agents[0].input_size))
        (logits.square().mean() + values.square().mean()).backward()
        optimizer.step()
        for actual, expected in zip(agents[1].parameters(), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        buffers = [AgentRollout.allocate(4, 2, a.input_size, "cpu") for a in agents]
        buffers[0].rewards.fill_(10)
        self.assertEqual(buffers[1].rewards.sum().item(), 0)

    def test_batched_inference_equals_individual_networks(self):
        agents = [IPPOAgent() for _ in range(3)]
        observations = torch.randn(7, 3, agents[0].input_size)
        batch = IndependentPolicyBatch(agents)
        values = batch(observations, sample=False)
        actions, logprobs, sampled_values = batch(observations)
        for i, agent in enumerate(agents):
            _, individual_logprobs, _, individual_values = agent.get_action_and_value(
                observations[:, i], actions[:, i])
            torch.testing.assert_close(values[:, i], individual_values)
            torch.testing.assert_close(sampled_values[:, i], individual_values)
            torch.testing.assert_close(logprobs[:, i], individual_logprobs)


class TrainingIntegrationTests(unittest.TestCase):
    def config(self, path, episodes=13):
        return TrainingConfig(total_episodes=episodes, num_envs=4, num_steps=32,
                              hidden_size=16, update_epochs=1, num_minibatches=2,
                              checkpoint_every=4, log_every=4, wandb_mode="disabled",
                              run_dir=str(path))

    def test_exact_count_checkpoint_and_resume(self):
        class StopAfterCheckpoint(Exception):
            pass
        with tempfile.TemporaryDirectory(prefix="socialrl-test-") as temporary:
            root = Path(temporary)
            full = MultiAgent(self.config(root / "full")).train()
            self.assertEqual(full["completed_episodes"], 13)
            self.assertEqual(full["joint_environment_steps"], 1300)
            partial = MultiAgent(self.config(root / "resumed"))
            original_save = partial.save_checkpoint
            def interrupt_after_save(path):
                original_save(path)
                raise StopAfterCheckpoint()
            partial.save_checkpoint = interrupt_after_save
            with self.assertRaises(StopAfterCheckpoint):
                partial.train()
            checkpoint = root / "resumed/checkpoints/latest.pt"
            # Simulate rows written after the last checkpoint before a crash.
            with (root / "resumed/episodes.csv").open("a") as target:
                target.write("999,999,0,100,0,0,0,0,0,0,0,0\n")
            with (root / "resumed/updates.jsonl").open("a") as target:
                target.write(json.dumps({"update": 999, "episode": 999}) + "\n")
            resumed = MultiAgent(self.config(root / "resumed"), resume=str(checkpoint)).train()
            self.assertEqual(resumed["joint_environment_steps"], 1300)
            full_agents, full_state = load_checkpoint(root / "full/checkpoints/final.pt")
            resumed_agents, resumed_state = load_checkpoint(root / "resumed/checkpoints/final.pt")
            self.assertEqual(full_state["update"], resumed_state["update"])
            for expected, actual in zip(full_agents, resumed_agents):
                for first, second in zip(expected.parameters(), actual.parameters()):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
            with (root / "resumed/episodes.csv").open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([int(row["episode"]) for row in rows], list(range(1, 14)))
            update_rows = [json.loads(line) for line in
                           (root / "resumed/updates.jsonl").read_text().splitlines()]
            self.assertEqual([row["update"] for row in update_rows],
                             list(range(1, resumed_state["update"] + 1)))


if __name__ == "__main__":
    unittest.main()
