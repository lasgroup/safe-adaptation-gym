from typing import Mapping

import numpy as np

import learn2learn_safely.envs.utils as utils
from learn2learn_safely.envs.robot import Robot
from learn2learn_safely.envs.world import World
from objective import Objective

import learn2learn_safely.envs.primitive_objects as po


class GoToGoal(Objective):
  GOAL_SIZE = 0.3
  GOAL_KEEPOUT = 0.305

  def __init__(self):
    pass

  def setup_placements(self) -> Mapping[str, tuple]:
    return {'goal': (None, self.GOAL_KEEPOUT)}

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    return po.get_goal('goal', self.GOAL_SIZE, layout['goal'],
                       utils.random_rot(rs))

  def compute_reward(self, robot: Robot, world: World) -> float:
    # Here a new goal gets resampled?
    pass

  def _resample_goal_position(self, layout: dict, placements: dict,
                              rs: np.random.RandomState):
    layout.pop('goal')
    for _ in range(10000):
      goal_xy = utils.draw_placement(rs, None, self.GOAL_KEEPOUT)
      for other_name, other_xy in layout.items():
        other_keepout = placements[other_name][1]
        dist = np.sqrt(np.sum(np.square(goal_xy - other_xy)))
        if dist >= other_keepout + self.GOAL_KEEPOUT:
          return goal_xy
    raise utils.ResamplingError('Failed to generate goal')
