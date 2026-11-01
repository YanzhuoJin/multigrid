"""Single creation point for the coursework's legacy MultiGrid environment.

The raw API is intentional: ``reset()`` returns a dict and ``step`` returns
``(observations, per_agent_rewards, joint_done, info)``. The fixed task has
100-step time limits; reaching the goal respawns only that agent and does not
terminate a rollout. Use ``info['TimeLimit.truncated']`` to bootstrap the final
observation while still stopping the advantage trace across the next reset.
"""

import gym
import numpy as np

# Importing the package registers the bundled environments.
from envs import gym_multigrid  # noqa: F401


DEFAULT_ENV_ID = "MultiGrid-Cluttered-Fixed-15x15"
FIXED_LAYOUT_ENV_IDS = {
    DEFAULT_ENV_ID,
    "MultiGrid-Cluttered-Fixed-Single-v0",
    "MultiGrid-Cluttered-Fixed-Single-6x6-v0",
}


def _portable_space_rng(space):
    """Preserve seeded Gym streams while avoiding its obsolete pickle hook."""
    space._np_random = np.random.Generator(space.np_random.bit_generator)
    children = getattr(space, "spaces", ())
    if isinstance(children, dict):
        children = children.values()
    for child in children:
        _portable_space_rng(child)


def make_env(env_id=DEFAULT_ENV_ID, seed=52, layout_seed=52,
             cache_observations=True):
    """Return an unwrapped environment with its own deterministic RNG.

    All parallel instances of the fixed assignment task share ``layout_seed``
    while distinct ``seed`` values give independent starting-position streams.
    """
    kwargs = {"seed": int(seed)}
    if env_id in FIXED_LAYOUT_ENV_IDS:
        kwargs["layout_seed"] = int(layout_seed)
        kwargs["cache_observations"] = bool(cache_observations)
    env = gym.make(env_id, **kwargs).unwrapped
    env.action_space.seed(int(seed))
    env.observation_space.seed(int(seed))
    _portable_space_rng(env.action_space)
    _portable_space_rng(env.observation_space)
    return env
