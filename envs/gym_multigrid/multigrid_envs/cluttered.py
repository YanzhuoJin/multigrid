"""Multi-agent goal-seeking task with many static obstacles.
"""
import gym_minigrid.minigrid as minigrid
from gym.utils import seeding

from envs.gym_multigrid import multigrid
from envs.gym_multigrid.register import register

import random
import math
import numpy as np

class ClutteredMultiGrid(multigrid.MultiGridEnv):
  """Goal seeking environment with obstacles."""

  def __init__(self, size=15, n_agents=3, n_clutter=25, randomize_goal=True,
               agent_view_size=5, max_steps=250, walls_are_lava=False,
               **kwargs):
    self.n_clutter = n_clutter
    self.randomize_goal = randomize_goal
    self.walls_are_lava = walls_are_lava
    super().__init__(grid_size=size, max_steps=max_steps, n_agents=n_agents,
                     agent_view_size=agent_view_size, **kwargs)

  def _gen_grid(self, width, height):
    self.grid = multigrid.Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    if self.randomize_goal:
      self.place_obj(minigrid.Goal(), max_tries=100)
    else:
      self.put_obj(minigrid.Goal(), width - 2, height - 2)
    for _ in range(self.n_clutter):
      if self.walls_are_lava:
        self.place_obj(minigrid.Lava(), max_tries=100)
      else:
        self.place_obj(minigrid.Wall(), max_tries=100)

    self.place_agent()

    self.mission = 'get to the green square'

  def step(self, action):
    obs, reward, done, info = multigrid.MultiGridEnv.step(self, action)
    return obs, reward, done, info


class ClutteredMultiGridFixed15x15(ClutteredMultiGrid):
  """Fixed walls and goal, with reproducible but changing agent placements.

  ``layout_seed`` selects the shared map; ``seed`` selects one independent
  episode/transition stream. No wall-clock or global NumPy RNG is used.
  """

  # Only walls and the goal exist in this task, so the visibility mask and
  # static image depend solely on observer position/direction. Agents do not
  # occlude in MiniGrid. Share these immutable images across vector instances.
  _view_caches = {}

  def __init__(self, n_agents=3, size=15, n_clutter=30, layout_seed=52,
               seed=52, cache_observations=True, **kwargs):
    self.layout_seed = int(layout_seed)
    self.randomize_agent_positions = True
    self.cache_observations = bool(cache_observations)
    self._static_grid = None
    super().__init__(n_agents=n_agents, size=size, n_clutter=n_clutter,
                     randomize_goal=False, agent_view_size=5, max_steps=100,
                     fixed_environment=True, seed=seed, **kwargs)

  def _gen_grid(self, width, height):
    if self._static_grid is None:
      # Generate static objects with their own RNG, leaving the episode stream
      # unchanged. Static objects are immutable under all seven task actions.
      episode_rng = self.np_random
      self.np_random, _ = seeding.np_random(self.layout_seed)
      try:
        self.grid = multigrid.Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(minigrid.Goal(), width - 2, height - 2)
        for _ in range(self.n_clutter):
          obj = minigrid.Lava() if self.walls_are_lava else minigrid.Wall()
          self.place_obj(obj, max_tries=100)
      finally:
        self.np_random = episode_rng
      self._static_grid = self.grid
      cache_key = (width, height, self.agent_view_size, self.see_through_walls,
                   self._static_grid.encode().tobytes())
      self._view_cache = self._view_caches.setdefault(cache_key, {})
    self.grid = multigrid.Grid(width, height)
    self.grid.grid = self._static_grid.grid.copy()
    self.place_agent(max_tries=10000)
    self.mission = 'get to the green square'

  def gen_agent_obs(self, agent_id):
    """Cache static fields of view, then overlay visible agents exactly.

    Disable ``cache_observations`` to use the reference Grid implementation;
    the regression tests compare both paths through complete trajectories.
    """
    if not self.cache_observations or self.carrying[agent_id] is not None:
      return super().gen_agent_obs(agent_id)
    x, y = self.agent_pos[agent_id]
    direction = self.agent_dir[agent_id]
    key = (x, y, direction)
    static_image = self._view_cache.get(key)
    if static_image is None:
      live_grid = self.grid
      self.grid = self._static_grid
      try:
        static_image, _ = super().gen_agent_obs(agent_id)
      finally:
        self.grid = live_grid
      static_image.setflags(write=False)
      self._view_cache[key] = static_image
    image = static_image.copy()
    dx, dy = minigrid.DIR_TO_VEC[direction]
    view_size = self.agent_view_size
    for other_id, other_pos in enumerate(self.agent_pos):
      if other_id == agent_id:
        continue
      relative_x = other_pos[0] - x
      relative_y = other_pos[1] - y
      vx = -dy * relative_x + dx * relative_y + view_size // 2
      vy = view_size - 1 - dx * relative_x - dy * relative_y
      if (0 <= vx < view_size and 0 <= vy < view_size and
          image[vx, vy, 0] != minigrid.OBJECT_TO_IDX['unseen']):
        image[vx, vy] = (minigrid.OBJECT_TO_IDX['agent'], other_id,
                         (self.agent_dir[other_id] - direction - 1) % 4)
    return image, direction


class AdaptiveClutteredMultiGridFixed15x15(ClutteredMultiGrid):
    """A cluttered environment where the walls fixed, but agents positions
    change every block_move_frequency episodes. There are multiple goals
    which move every goal_move_frequency episodes."""

    def __init__(self, block_move_frequency=10, goal_move_frequency=10, **kwargs):
      self.block_move_frequency = block_move_frequency  # Change this as needed
      self.goal_move_frequency = goal_move_frequency # Change this as needed
      self.episode_count = 0
      self.wall_locs = np.array([], dtype=int).reshape(0, 2)  # Empty 2D array for wall locations
      self.goal_locs = np.array([], dtype=int).reshape(0, 2)  # Empty 2D array for goal locations
      super().__init__(n_agents=3, size=15, n_clutter=30, randomize_goal=False,
                      agent_view_size=5, max_steps=5, fixed_environment=True, **kwargs)

    def reset(self):
        """Override parent class reset() to not persist agent locations across episodes when grid is reset.
        Grid is not reset() every time unlike parent class."""
        for i in range(self.n_agents):
          if self.episode_count > 0:
              pos = self.agent_pos[i]
              self.grid.set(pos[0], pos[1], None) # Empty the cell
        return super().reset()

    def _gen_grid(self, width, height):
        """Override parent class reset() method to move a block every block_move_frequency episodes."""
        if self.episode_count == 0:
          # Create a random grid on the first episode
          self.grid = multigrid.Grid(width, height)
          self.grid.wall_rect(0, 0, width, height)

          # Place the goals (n_agents # of goals) in random initial location
          # and store their locations
          for _ in range(self.n_agents):
            goal_loc = self.place_obj(minigrid.Goal(), max_tries=100)
            self.goal_locs = np.append(self.goal_locs, [goal_loc], axis=0)
          
          # Place walls and store their locations
          for _ in range(self.n_clutter):
            wall_loc = self.place_obj(minigrid.Wall(), max_tries=100)
            self.wall_locs = np.append(self.wall_locs, [wall_loc], axis=0)

        for i in range(self.n_agents):
          self.place_one_agent(i)

        if self.episode_count % self.block_move_frequency == 0:
            self.move_random_block()

        if self.goal_move_frequency is None:
          # Goal does not move
          pass
        elif (self.episode_count > 0) and (self.episode_count % self.goal_move_frequency == 0):
          # Move each goal
          for goal_idx in range(self.n_agents):
            self.move_goal(goal_idx)

        self.mission = 'get to the green square'

        self.episode_count += 1

    def move_random_block(self):
        """Move a random block to a random empty cell."""
        wall_idx = random.randint(0, len(self.wall_locs) - 1)
        remove_wall_loc = self.wall_locs[wall_idx]
        self.grid.set(remove_wall_loc[0], remove_wall_loc[1], None) # Empty the cell
        new_wall_loc = self.place_obj(minigrid.Wall(), max_tries=100)
        self.wall_locs[wall_idx] = new_wall_loc

    def move_goal(self, goal_idx):
        """Move each goal by 1 square to a randomly-chosen
        adjacent index."""
        remove_goal_loc = np.copy(self.goal_locs[goal_idx]) # copy to avoid pointer issue
        self.grid.set(remove_goal_loc[0], remove_goal_loc[1], None) # Empty the cell

        # Pick a random direction to move the goal.
        # Possible moves: up, down, left, right
        possible_moves = np.array([
            [remove_goal_loc[0] - 1, remove_goal_loc[1]], # up
            [remove_goal_loc[0] + 1, remove_goal_loc[1]], # down
            [remove_goal_loc[0], remove_goal_loc[1] - 1], # left
            [remove_goal_loc[0], remove_goal_loc[1] + 1]  # right
        ])
        
        # Filter out moves that are out of bounds or occupied
        valid_moves = np.array([
            [x, y] for x, y in possible_moves
            if 0 <= x < self.grid.width and 0 <= y < self.grid.height and self.grid.get(x, y) is None
        ])
        
        if valid_moves.size > 0: # If a valid move exists
            new_goal_loc = random.choice(valid_moves)
            self.grid.set(new_goal_loc[0], new_goal_loc[1], minigrid.Goal())
            self.goal_locs[goal_idx] = new_goal_loc

        if (self.goal_locs[goal_idx] == remove_goal_loc).all():
            # If the goal did not move, notify
            print("Goal did not move. New location is same as old location.")

    def place_one_agent(self,
                      agent_id,
                      top=None,
                      size=None,
                      rand_dir=True,
                      max_tries=math.inf,
                      agent_obj=None):
      """Override this method to place agents at fixed positions."""

      self.agent_pos[agent_id] = None

      if agent_id == 0:
        self.agent_pos[agent_id] = np.array([1, 1])
      elif agent_id == 1:
        self.agent_pos[agent_id] = np.array([1, self.grid.height - 2])
      elif agent_id == 2:
        self.agent_pos[agent_id] = np.array([self.grid.width - 2, 1])

      self.place_agent_at_pos(agent_id, self.agent_pos[agent_id], 
                              agent_obj=agent_obj, rand_dir=rand_dir)

      return self.agent_pos[agent_id]


class ClutteredMultiGridSingle6x6(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, size=6, n_clutter=5, randomize_goal=True,
                     agent_view_size=5, max_steps=50, **kwargs)


class ClutteredMultiGridSingle(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, size=15, n_clutter=25, randomize_goal=True,
                     agent_view_size=5, max_steps=250, **kwargs)


class ClutteredMultiGridSingleFixed(ClutteredMultiGridFixed15x15):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, **kwargs)


class ClutteredMultiGridSingleFixed6x6(ClutteredMultiGridSingleFixed):

  def __init__(self, **kwargs):
    super().__init__(size=6, n_clutter=5, **kwargs)


class Cluttered40Minigrid(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=40, minigrid_mode=True, **kwargs)


class Cluttered10Minigrid(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=10, minigrid_mode=True, **kwargs)


class Cluttered50Minigrid(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=50, minigrid_mode=True, **kwargs)


class Cluttered5Minigrid(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=5, minigrid_mode=True, **kwargs)


class Cluttered1MinigridMini(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=1, minigrid_mode=True, size=6,
                     **kwargs)


class Cluttered6MinigridMini(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=6, minigrid_mode=True, size=6,
                     **kwargs)


class Cluttered7MinigridMini(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=7, minigrid_mode=True, size=6,
                     **kwargs)


class ClutteredMinigridLava(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, walls_are_lava=True, minigrid_mode=True,
                     **kwargs)


class ClutteredMinigridLavaMini(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=4, walls_are_lava=True, size=6,
                     minigrid_mode=True, **kwargs)


class ClutteredMinigridLavaMedium(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=15, walls_are_lava=True, size=10,
                     minigrid_mode=True, **kwargs)


class Cluttered15MinigridMedium(ClutteredMultiGrid):

  def __init__(self, **kwargs):
    super().__init__(n_agents=1, n_clutter=15, minigrid_mode=True, size=10,
                     **kwargs)

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-Cluttered-v0',
    entry_point=module_path + ':ClutteredMultiGrid'
)

register(
    env_id='MultiGrid-Cluttered-Fixed-15x15',
    entry_point=module_path + ':ClutteredMultiGridFixed15x15'
)

# For new multigrid env
register(
    env_id='MultiGrid-AdaptiveCluttered-Fixed-15x15',
    entry_point=module_path + ':AdaptiveClutteredMultiGridFixed15x15'
)

register(
    env_id='MultiGrid-Cluttered-Single-v0',
    entry_point=module_path + ':ClutteredMultiGridSingle'
)

register(
    env_id='MultiGrid-Cluttered-Fixed-Single-v0',
    entry_point=module_path + ':ClutteredMultiGridSingleFixed'
)


register(
    env_id='MultiGrid-Cluttered-Fixed-Single-6x6-v0',
    entry_point=module_path + ':ClutteredMultiGridSingleFixed6x6'
)

register(
    env_id='MultiGrid-Cluttered-Single-6x6-v0',
    entry_point=module_path + ':ClutteredMultiGridSingle6x6'
)

register(
    env_id='MultiGrid-Cluttered40-Minigrid-v0',
    entry_point=module_path + ':Cluttered40Minigrid'
)

register(
    env_id='MultiGrid-Cluttered10-Minigrid-v0',
    entry_point=module_path + ':Cluttered10Minigrid'
)

register(
    env_id='MultiGrid-Cluttered50-Minigrid-v0',
    entry_point=module_path + ':Cluttered50Minigrid'
)

register(
    env_id='MultiGrid-Cluttered5-Minigrid-v0',
    entry_point=module_path + ':Cluttered5Minigrid'
)

register(
    env_id='MultiGrid-MiniCluttered1-Minigrid-v0',
    entry_point=module_path + ':Cluttered1MinigridMini'
)

register(
    env_id='MultiGrid-MiniCluttered6-Minigrid-v0',
    entry_point=module_path + ':Cluttered6MinigridMini'
)

register(
    env_id='MultiGrid-MiniCluttered7-Minigrid-v0',
    entry_point=module_path + ':Cluttered7MinigridMini'
)

register(
    env_id='MultiGrid-Cluttered-Lava-Minigrid-v0',
    entry_point=module_path + ':ClutteredMinigridLava'
)

register(
    env_id='MultiGrid-MiniCluttered-Lava-Minigrid-v0',
    entry_point=module_path + ':ClutteredMinigridLavaMini'
)

register(
    env_id='MultiGrid-MediumCluttered-Lava-Minigrid-v0',
    entry_point=module_path + ':ClutteredMinigridLavaMedium'
)

register(
    env_id='MultiGrid-MediumCluttered15-Minigrid-v0',
    entry_point=module_path + ':Cluttered15MinigridMedium'
)

