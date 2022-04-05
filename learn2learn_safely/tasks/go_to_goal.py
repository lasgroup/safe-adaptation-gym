from typing import Dict, Tuple

import numpy as np

import learn2learn_safely.primitive_objects as po
import learn2learn_safely.utils as utils
from learn2learn_safely.mujoco_bridge import MujocoBridge
from learn2learn_safely.tasks.task import Task


class GoToGoal(Task):
  GOAL_SIZE = 0.3
  GOAL_KEEPOUT = 0.305

  def __init__(self):
    self._last_goal_distance = None

  def setup_placements(self) -> Dict[str, tuple]:
    return {'goal': (None, self.GOAL_KEEPOUT)}

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    return {
        'bodies': {
            'goal':
                po.get_goal('goal', self.GOAL_SIZE, layout['goal'],
                            utils.random_rot(rs))
        }
    }

  def compute_reward(self, layout: dict, placements: dict,
                     rs: np.random.RandomState,
                     mujoco_bridge: MujocoBridge) -> Tuple[float, bool, dict]:
    goal_pos = np.asarray(mujoco_bridge.body_pos('goal'))
    robot_pos = mujoco_bridge.body_pos('robot')
    distance = np.linalg.norm(robot_pos - goal_pos)
    reward = self._last_goal_distance - distance
    self._last_goal_distance = distance
    info = {}
    if distance <= self.GOAL_SIZE:
      info['goal_met'] = True
      utils.update_layout(layout, mujoco_bridge)
      self.build(layout, placements, rs, mujoco_bridge)
      reward += 1.
    return reward, False, info

  def set_mocaps(self, mujoco_bridge: MujocoBridge):
    pass

  def build(self, layout: dict, placements: dict, rs: np.random.RandomState,
            mujoco_bridge: MujocoBridge):
    goal_xy = self._resample_goal_position(layout, placements, rs)
    layout['goal'] = goal_xy
    robot_pos = mujoco_bridge.body_pos('robot')[:2]
    self._last_goal_distance = np.linalg.norm(robot_pos - goal_xy)
    mujoco_bridge.set_body_pos('goal', goal_xy)
    mujoco_bridge.physics.forward()

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
