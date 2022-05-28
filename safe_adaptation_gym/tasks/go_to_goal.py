from typing import Dict, Tuple

import numpy as np

import safe_adaptation_gym.primitive_objects as po
import safe_adaptation_gym.utils as utils
from safe_adaptation_gym.mujoco_bridge import MujocoBridge
from safe_adaptation_gym.tasks.task import Task
from safe_adaptation_gym import rewards


class GoToGoal(Task):
  GOAL_SIZE = 0.3
  GOAL_KEEPOUT = 0.4

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
    reward = rewards.tolerance(
        distance,
        bounds=(0, self.GOAL_SIZE),
        sigmoid='linear',
        margin=mujoco_bridge.arena_radius,
        value_at_margin=0.)
    info = {}
    if distance <= self.GOAL_SIZE:
      info['goal_met'] = True
      utils.update_layout(layout, mujoco_bridge)
      self.reset(layout, placements, rs, mujoco_bridge)
    return reward, False, info

  def set_mocaps(self, mujoco_bridge: MujocoBridge):
    pass

  def reset(self, layout: dict, placements: dict, rs: np.random.RandomState,
            mujoco_bridge: MujocoBridge):
    goal_xy = self._resample_goal_position(layout, placements, rs)
    layout['goal'] = goal_xy
    mujoco_bridge.set_body_pos('goal', goal_xy)
    mujoco_bridge.physics.forward()

  def _resample_goal_position(self, layout: dict, placements: dict,
                              rs: np.random.RandomState):
    layout.pop('goal')
    for _ in range(10000):
      goal_xy = utils.draw_placement(rs, None, self.placement_extents,
                                     self.GOAL_KEEPOUT)
      valid_placement = True
      for other_name, other_xy in layout.items():
        other_keepout = placements[other_name][1]
        dist = np.linalg.norm(goal_xy - other_xy)
        if dist < other_keepout + self.GOAL_KEEPOUT:
          valid_placement = False
          break
      if valid_placement:
        return goal_xy
    raise utils.ResamplingError('Failed to generate goal')

  @property
  def obstacles_distribution(self):
    return [0.4, 0.4, 0.1, 0.1]

  @property
  def placement_extents(self) -> Tuple[float, float, float, float]:
    return 2.25, 2.25, 2.5, 2.25
