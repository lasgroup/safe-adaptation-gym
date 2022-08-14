import numpy as np

from safe_adaptation_gym import tasks


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
    self._origin = None

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
    mujoco_bridge.set_body_pos('goal', self._origin + target)

  def _random_radius(self, rs: np.random.RandomState):
    return rs.uniform(self.MIN_RADIUS, self.MAX_RADIUS)

  def reset(self, layout: dict, placements: dict, rs: np.random.RandomState,
            mujoco_bridge: tasks.task.MujocoBridge):
    self.rs = rs
    super(FollowTheLeader, self).reset(layout, placements, rs, mujoco_bridge)
    self._origin = mujoco_bridge.body_pos('goal')[:2].copy()
