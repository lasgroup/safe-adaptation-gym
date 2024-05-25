from enum import Enum
from typing import Dict, Tuple

import numpy as np

import safe_adaptation_gym.consts as c
import safe_adaptation_gym.utils as utils
from safe_adaptation_gym.primitive_objects import get_button
from safe_adaptation_gym.tasks.task import Task, MujocoBridge


class PressButtons(Task):
  NUM_BUTTONS = 4
  BUTTONS_KEEPOUT = 0.2
  BUTTON_SIZE = 0.1
  BUTTON_COLOR = np.array([1, 105 / 255, 180 / 255, 1])
  BUTTON_TICKING_DELAY = 5

  def __init__(self):
    super(PressButtons, self).__init__()
    self._last_goal_distance = None
    self._goal_button = None
    self._state = State.NORMAL
    self._goal_button_timer = Timer(self.BUTTON_TICKING_DELAY)

  def setup_placements(self) -> Dict[str, tuple]:
    placements = dict()
    for i in range(self.NUM_BUTTONS):
      placements['buttons{}'.format(i)] = (None, self.BUTTONS_KEEPOUT)
    return placements

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    return {
        'bodies': {
            name: get_button(name, self.BUTTON_SIZE, xy, utils.random_rot(rs),
                             self.BUTTON_COLOR)
            for name, xy in layout.items()
            if 'buttons' in name
        }
    }

  def compute_reward(self, layout: dict, placements: dict,
                     rs: np.random.RandomState,
                     mujoco_bridge: MujocoBridge) -> Tuple[float, bool, dict]:
    goal_pos = mujoco_bridge.body_pos(self._goal_button)[:2]
    robot_pos = mujoco_bridge.body_pos('robot')[:2]
    goal_distance = np.linalg.norm(robot_pos - goal_pos)
    reward = self._last_goal_distance - goal_distance
    self._last_goal_distance = goal_distance
    info = {}
    if mujoco_bridge.robot_contacts([self._goal_button]):
      reward += 1.
      info['goal_met'] = True
      self._sample_goal_button(rs, mujoco_bridge)
      self._state = State.BUTTON_CHANGE
    if self._state == State.BUTTON_CHANGE:
      if not self._goal_button_timer.done:
        self._goal_button_timer.tick()
      else:
        self._state = State.NORMAL
        self._goal_button_timer.reset()
    self._update_goal_button(mujoco_bridge)
    return reward, False, info

  def reset(self, layout: dict, placements: dict, rs: np.random.RandomState,
            mujoco_bridge: MujocoBridge):
    self._sample_goal_button(rs, mujoco_bridge)
    self._update_goal_button(mujoco_bridge)

  def _sample_goal_button(self, rs: np.random.RandomState,
                          mujoco_bridge: MujocoBridge):
    self._goal_button = 'buttons{}'.format(rs.choice(self.NUM_BUTTONS))
    self._goal_button_timer.reset()
    goal_pos = mujoco_bridge.body_pos(self._goal_button)[:2]
    robot_pos = mujoco_bridge.body_pos('robot')[:2]
    self._last_goal_distance = np.linalg.norm(robot_pos - goal_pos)

  def _update_goal_button(self, mujoco_bridge: MujocoBridge):
    if self._state == State.BUTTON_CHANGE:
      for i in range(self.NUM_BUTTONS):
        name = 'buttons{}'.format(i)
        mujoco_bridge.geom_rgba[name] = self.BUTTON_COLOR
        mujoco_bridge.user_groups[name] = [c.GROUP_INACTIVE]
    elif self._state == State.NORMAL:
      for i in range(self.NUM_BUTTONS):
        name = 'buttons{}'.format(i)
        if name != self._goal_button:
          mujoco_bridge.user_groups[name] = [c.GROUP_OBJECTS]
        else:
          mujoco_bridge.geom_rgba[self._goal_button] = c.GOAL_COLOR
          mujoco_bridge.user_groups[self._goal_button] = [c.GROUP_GOAL]

  def set_mocaps(self, mujoco_bridge: MujocoBridge):
    pass

  @property
  def obstacles_distribution(self):
    return [0.3, 0.3, 0.1, 0.3]

  @property
  def num_obstacles(self) -> int:
    return 8


class State(Enum):
  """ State of the Buttons class. """
  BUTTON_CHANGE = 0
  NORMAL = 1


class Timer:

  def __init__(self, count):
    self._count = count
    self.time = 0

  def tick(self):
    self.time = max(0, self.time - 1)

  def reset(self):
    self.time = self._count

  @property
  def done(self):
    return self.time == 0
