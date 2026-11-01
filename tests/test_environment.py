"""Regression tests for the assignment environment's actual task semantics."""

import unittest
import pickle

import numpy as np

from env_adapter import make_env
from envs.gym_multigrid.multigrid import Agent, MultiGridEnv


def static_layout(env):
    layout = env.grid.encode().copy()
    layout[layout[:, :, 0] == 10] = (1, 0, 0)  # Agents -> empty cells.
    return layout


def relocate_agent(env, agent_id, position, direction):
    env.grid.set(*env.agent_pos[agent_id], None)
    env.agent_dir[agent_id] = direction
    env.place_agent_at_pos(agent_id, np.array(position), rand_dir=False)


class EnvironmentTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_continuation_and_space_rng(self):
        original = make_env(seed=11)
        for _ in range(19):
            original.step(original.action_space.sample())
        restored = pickle.loads(pickle.dumps(original))
        for _ in range(120):
            left_action = original.action_space.sample()
            right_action = restored.action_space.sample()
            np.testing.assert_array_equal(left_action, right_action)
            left, right = original.step(left_action), restored.step(right_action)
            for field in ("image", "direction"):
                np.testing.assert_array_equal(left[0][field], right[0][field])
            self.assertEqual(left[1:], right[1:])
            if left[2]:
                original.reset()
                restored.reset()
        # A resumed environment must also be saveable again.
        pickle.loads(pickle.dumps(restored))
        original.close()
        restored.close()

    def test_cached_and_reference_trajectories_match_including_goal_respawns(self):
        goal_hits = 0
        for layout_seed in (52, 91):
            for seed in (17, 321, 999):
                cached = make_env(seed=seed, layout_seed=layout_seed)
                reference = make_env(seed=seed, layout_seed=layout_seed,
                                     cache_observations=False)
                action_rng = np.random.default_rng(seed)
                for step in range(220):
                    if step % 100 == 0:
                        # Force a real goal interaction in addition to random
                        # actions, so respawn RNG and observations are covered.
                        goal = np.array([13, 13])
                        for direction, delta in enumerate(((1, 0), (0, 1),
                                                           (-1, 0), (0, -1))):
                            position = goal - delta
                            if cached.grid.get(*position) is None:
                                for env in (cached, reference):
                                    relocate_agent(env, 0, position, direction)
                                break
                        actions = [2, 6, 6]
                    else:
                        actions = action_rng.integers(0, 7, size=3)
                    left, right = cached.step(actions), reference.step(actions)
                    goal_hits += int(left[1][0] > 0)
                    for field in ("image", "direction"):
                        np.testing.assert_array_equal(left[0][field], right[0][field])
                    np.testing.assert_array_equal(cached.agent_pos, reference.agent_pos)
                    np.testing.assert_array_equal(left[1], right[1])
                    self.assertEqual(left[2:], right[2:])
                    if left[2]:
                        left_obs, right_obs = cached.reset(), reference.reset()
                        for field in ("image", "direction"):
                            np.testing.assert_array_equal(left_obs[field], right_obs[field])
                cached.close()
                reference.close()
        self.assertGreaterEqual(goal_hits, 12)

    def test_cached_views_preserve_agent_identity_direction_and_occlusion(self):
        env = make_env(seed=811)
        rng = np.random.default_rng(66)
        empty_cells = [(x, y) for x in range(1, 14) for y in range(1, 14)
                       if env._static_grid.get(x, y) is None]
        # Sample tightly grouped agents as well as distant agents. This covers
        # agent overlays on both visible and wall-occluded cells in every pose.
        for trial in range(500):
            for position in env.agent_pos:
                env.grid.set(*position, None)
            anchor = empty_cells[int(rng.integers(len(empty_cells)))]
            nearby = [p for p in empty_cells
                      if abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]) <= 4]
            candidates = nearby if trial % 2 else empty_cells
            positions = rng.choice(len(candidates), size=3, replace=False)
            for agent_id, index in enumerate(positions):
                env.agent_dir[agent_id] = (trial + agent_id) % 4
                env.place_agent_at_pos(agent_id, np.array(candidates[index]), rand_dir=False)
            for agent_id in range(3):
                image, direction = env.gen_agent_obs(agent_id)
                expected, expected_direction = MultiGridEnv.gen_agent_obs(env, agent_id)
                np.testing.assert_array_equal(image, expected)
                self.assertEqual(direction, expected_direction)
                # Consumers may mutate their observation; the cache stays safe.
                image[:] = 255
                np.testing.assert_array_equal(env.gen_agent_obs(agent_id)[0], expected)
        env.close()

    def test_required_task_and_observation_contract(self):
        env = make_env(seed=101)
        obs = env.reset()
        self.assertEqual((env.width, env.height, env.n_agents), (15, 15, 3))
        self.assertEqual(env.max_steps, 100)
        self.assertEqual(len(env.actions), 7)
        self.assertFalse(env.competitive)
        self.assertEqual(obs["image"].shape, (3, 5, 5, 3))
        self.assertEqual(obs["direction"].shape, (3,))
        self.assertEqual(obs["image"].dtype, np.uint8)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(env.grid.get(13, 13).type, "goal")
        walls = sum(cell is not None and cell.type == "wall"
                    for cell in env.grid.grid)
        self.assertEqual(walls, 56 + 30)
        for a in range(3):
            np.testing.assert_array_equal(obs["image"][a], env.gen_agent_obs(a)[0])
        self.assertEqual(len({tuple(pos) for pos in env.agent_pos}), 3)
        env.close()

    def test_shared_fixed_map_and_varying_independent_start_streams(self):
        first, second = make_env(seed=123), make_env(seed=456)
        np.testing.assert_array_equal(static_layout(first), static_layout(second))
        self.assertFalse(np.array_equal(first.agent_pos, second.agent_pos))
        expected_map = static_layout(first)
        starts = set()
        for _ in range(8):
            first.reset()
            starts.add(tuple(tuple(pos) for pos in first.agent_pos))
            np.testing.assert_array_equal(expected_map, static_layout(first))
        self.assertGreater(len(starts), 1)
        first.close()
        second.close()

    def test_replay_is_independent_of_global_rng_and_other_envs(self):
        first, second = make_env(seed=73), make_env(seed=73)
        nuisance = make_env(seed=999)
        actions_rng = np.random.default_rng(555)
        for _ in range(230):
            actions = actions_rng.integers(0, 7, size=3)
            left = first.step(actions)
            np.random.seed(9876)
            np.random.uniform(size=20)
            nuisance.step([0, 1, 2])
            right = second.step(actions)
            np.testing.assert_array_equal(left[0]["image"], right[0]["image"])
            np.testing.assert_array_equal(left[0]["direction"], right[0]["direction"])
            np.testing.assert_array_equal(left[1], right[1])
            self.assertEqual(left[2:], right[2:])
            if left[2]:
                first.reset()
                second.reset()
            if nuisance.step_count >= nuisance.max_steps:
                nuisance.reset()
        first.close()
        second.close()
        nuisance.close()

    def test_goal_rewards_only_successful_agent_and_respawns(self):
        env = make_env(seed=5)
        goal = np.array([13, 13])
        # Find an open neighboring cell and make agent 0 face the real goal.
        for direction, delta in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
            position = goal - delta
            cell = env.grid.get(*position)
            if cell is None or isinstance(cell, Agent):
                if isinstance(cell, Agent) and cell.agent_id != 0:
                    # Rarely another agent starts here; move it to a free cell.
                    free = next((x, y) for x in range(1, 14) for y in range(1, 14)
                                if env.grid.get(x, y) is None)
                    relocate_agent(env, cell.agent_id, free, 0)
                relocate_agent(env, 0, position, direction)
                break
        else:
            self.fail("The fixed layout must have a reachable goal neighbor")
        obs, rewards, done, info = env.step([env.actions.forward,
                                            env.actions.done, env.actions.done])
        self.assertAlmostEqual(rewards[0], 1.0 - 0.9 / 100)
        self.assertEqual(list(rewards[1:]), [0, 0])
        self.assertFalse(done)
        self.assertFalse(info["terminated"])
        self.assertFalse(info["truncated"])
        self.assertEqual(env.done, [True, False, False])
        self.assertFalse(np.array_equal(env.agent_pos[0], goal))
        self.assertEqual(env.grid.get(*goal).type, "goal")
        self.assertTrue(env.observation_space.contains(obs))
        env.close()

    def test_joint_timeout_is_truncation_and_reset_clears_success(self):
        env = make_env(seed=22)
        for step in range(1, 101):
            _, rewards, done, info = env.step([env.actions.done] * 3)
            self.assertEqual(list(rewards), [0, 0, 0])
            self.assertEqual(done, step == 100)
        self.assertTrue(info["TimeLimit.truncated"])
        self.assertTrue(info["truncated"])
        self.assertFalse(info["terminated"])
        env.done[0] = True
        env.reset()
        self.assertEqual(env.step_count, 0)
        self.assertEqual(env.done, [False, False, False])
        env.close()


if __name__ == "__main__":
    unittest.main()
