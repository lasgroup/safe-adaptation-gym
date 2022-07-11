import numpy as np

import safe_adaptation_gym.primitive_objects as po
from safe_adaptation_gym import tasks
from safe_adaptation_gym import utils
from safe_adaptation_gym import consts as c


class FollowTheLeader(tasks.GoToGoal):

  MIN_RADIUS = 0.2
  MAX_RADIUS = 1.0
  SAMPLE_POINTS = 10

  def __init__(self):
    super(FollowTheLeader, self).__init__()
    self.rs = None
    self._current_radius = self.MAX_RADIUS
    self._next_radius = self.MIN_RADIUS
    self._timer = tasks.press_buttons.Timer(self.SAMPLE_POINTS)

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    goal_dict = {
        'name': 'goal',
        'size': [self.GOAL_SIZE, self.GOAL_SIZE / 2.],
        'pos': np.r_[layout['goal'], self.GOAL_SIZE / 2. + 1e-2],
        'quat': utils.rot2quat(utils.random_rot(rs)),
        'type': 'cylinder',
        'contype': 0,
        'conaffinity': 0,
        'user': [c.GROUP_GOAL],
        'rgba': c.GOAL_COLOR * [1, 1, 1, 0.25]
    }
    return {'bodies': {'goal': ([po.mocap_attributes_to_xml(goal_dict)], '')}}

  def set_mocaps(self, mujoco_bridge: tasks.task.MujocoBridge):
    self._timer.tick()
    if self._timer.done:
      self._current_radius = self._next_radius
      self._next_radius = self._random_radius(self.rs)
      self._timer.reset()
    phase = float(mujoco_bridge.time)
    progress = (self.SAMPLE_POINTS - self._timer.time) / self.SAMPLE_POINTS
    radius = progress * (self._next_radius -
                         self._current_radius) + self._current_radius
    target = np.array([np.sin(phase), np.cos(phase)]) * radius
    pos = np.r_[target, [0.]]
    mujoco_bridge.set_mocap_pos('goal', pos)

  def _random_radius(self, rs: np.random.RandomState):
    return rs.uniform(self.MIN_RADIUS, self.MAX_RADIUS)

  def reset(self, layout: dict, placements: dict, rs: np.random.RandomState,
            mujoco_bridge: tasks.task.MujocoBridge):
    self.rs = rs
    super(FollowTheLeader, self).reset(layout, placements, rs, mujoco_bridge)
