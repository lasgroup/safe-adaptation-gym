from typing import Tuple

import numpy as np

from safe_adaptation_gym.tasks import task
from safe_adaptation_gym.tasks import push_box
from safe_adaptation_gym import utils


class HaulBox(push_box.PushBox):

  def __init__(self):
    super(HaulBox, self).__init__()

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    # Import mjcf here so that rendering with multiple process works.
    from dm_control import mjcf
    layout['box'] = layout['robot'].copy()
    layout['box'][0] += self.BOX_SIZE * 3.
    config = super(HaulBox, self).build_world_config(layout, rs)
    model = mjcf.RootElement()
    string = model.tendon.add(
        'spatial', name='string', limited=True, range='0 0.75', width=0.002)
    string.add('site', site='robot')
    string.add('site', site='box_site')
    box_config = {'others': {'tendon': string.to_xml_string()}}
    utils.merge(config, box_config)
    return config

  def compute_reward(
      self, layout: dict, placements: dict, rs: np.random.RandomState,
      mujoco_bridge: task.MujocoBridge) -> Tuple[float, bool, dict]:
    goal_pos = mujoco_bridge.body_pos('goal')[:2]
    box_pos = mujoco_bridge.body_pos('box')[:2]
    box_goal_distance = np.linalg.norm(box_pos - goal_pos)
    reward = self._last_box_goal_distance - box_goal_distance
    self._last_box_goal_distance = box_goal_distance
    info = {}
    if box_goal_distance <= self.GOAL_SIZE:
      info['goal_met'] = True
      utils.update_layout(layout, mujoco_bridge)
      self.reset(layout, placements, rs, mujoco_bridge)
      reward += 1.
    return reward, False, info
