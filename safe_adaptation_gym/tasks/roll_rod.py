from typing import Tuple

import numpy as np

import safe_adaptation_gym.consts as c
from safe_adaptation_gym.tasks.push_box import PushBox, GoToGoal
from safe_adaptation_gym.utils import merge


class RollRod(PushBox):
  ROD_LENGTH = 0.3
  ROD_RADIUS = 0.08
  BOX_KEEPOUT = 0.7
  BOX_SIZE = 0.25

  def __init__(self):
    super(RollRod, self).__init__()

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    # Import mjcf here so that rendering with multiple process works.
    from dm_control import mjcf
    config = GoToGoal.build_world_config(self, layout, rs)
    model = mjcf.RootElement()
    rod_root = model.worldbody.add(
        'body',
        name='box',
        euler='90 0 0',
        pos=[layout['box'][0], layout['box'][1], self.ROD_RADIUS])
    rod_root.add('freejoint', name='box')
    rod_root.add(
        'geom',
        name='box',
        type='cylinder',
        size=[self.ROD_RADIUS, self.ROD_LENGTH],
        density=self.BOX_DENSITY / 2.,
        friction=[1.2, 0.001, 0.05],
        rgba=[1, 1, 1, 1],
        material='interactobj',
        user=[c.GROUP_OBJECTS],
        priority=1)
    box_config = {'bodies': {'box': ([rod_root.to_xml_string()], '')}}
    merge(config, box_config)
    return config

  @property
  def placement_extents(self) -> Tuple[float, float, float, float]:
    return -1.75, -1.75, 1.75, 1.75
