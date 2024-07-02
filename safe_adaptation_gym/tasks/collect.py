from typing import Dict, Tuple

import numpy as np

import safe_adaptation_gym.consts as c
from safe_adaptation_gym.tasks.press_buttons import PressButtons
from safe_adaptation_gym.tasks.task import MujocoBridge


class Collect(PressButtons):
  NUM_BUTTONS = 6

  def __init__(self):
    super(PressButtons, self).__init__()
    self._active_buttons = set(
        'buttons{}'.format(i) for i in range(self.NUM_BUTTONS))

  def setup_placements(self) -> Dict[str, tuple]:
    placements = dict()
    for i in range(self.NUM_BUTTONS):
      placements['buttons{}'.format(i)] = ([(-1.5, -1.5, 1.5, 1.5)], self.BUTTONS_KEEPOUT)
    return placements

  def compute_reward(self, layout: dict, placements: dict,
                     rs: np.random.RandomState,
                     mujoco_bridge: MujocoBridge) -> Tuple[float, bool, dict]:
    if not self._active_buttons:
      self.reset(layout, placements, rs, mujoco_bridge)
    info = {}
    reward = 0.
    for button in self._active_buttons:
      if mujoco_bridge.robot_contacts([button]):
        reward += 1.
        info['goal_met'] = True
        mujoco_bridge.geom_rgba[button] = self.BUTTON_COLOR
        mujoco_bridge.user_groups[button] = [c.GROUP_INACTIVE]
        self._active_buttons.remove(button)
        break
    return reward, False, info

  def reset(self, layout: dict, placements: dict, rs: np.random.RandomState,
            mujoco_bridge: MujocoBridge):
    for i in range(self.NUM_BUTTONS):
      name = 'buttons{}'.format(i)
      mujoco_bridge.geom_rgba[name] = c.GOAL_COLOR
      mujoco_bridge.user_groups[name] = [c.GROUP_GOAL]
      self._active_buttons.add(name)

  @property
  def placement_extents(self) -> Tuple[float, float, float, float]:
    return [-2.25, -2.25, 2.25, 2.25]