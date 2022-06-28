from typing import Tuple

import numpy as np

from safe_adaptation_gym.tasks.task import MujocoBridge
from safe_adaptation_gym.tasks.push_box import PushBox, GoToGoal
from safe_adaptation_gym.utils import merge


class HaulBox(PushBox):

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
    merge(config, box_config)
    return config

  def compute_reward(self, layout: dict, placements: dict,
                     rs: np.random.RandomState,
                     mujoco_bridge: MujocoBridge) -> Tuple[float, bool, dict]:
    # No need to reward the agent for getting close to the box as the robot
    # and box are tied.
    return GoToGoal.compute_reward(self, layout, placements, rs, mujoco_bridge)
