from typing import Tuple

import numpy as np

import safe_adaptation_gym.consts as c
from safe_adaptation_gym.tasks.push_box import PushBox, GoToGoal
from safe_adaptation_gym.utils import merge, convert_to_text


class DribbleBall(PushBox):
  SPHERE_RADIUS = 0.14
  BOX_KEEPOUT = 0.2
  BOX_SIZE = SPHERE_RADIUS

  def __init__(self):
    super(DribbleBall, self).__init__()

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    # Import mjcf here so that rendering with multiple process works.
    from dm_control import mjcf
    config = GoToGoal.build_world_config(self, layout, rs)
    model = mjcf.RootElement()
    ball = model.worldbody.add(
        'body', name='box', pos=list(layout['box']) + [self.SPHERE_RADIUS])
    ball.add('freejoint', name='box')
    ball.add(
        'geom',
        name='box',
        type='sphere',
        size=convert_to_text(np.array([self.SPHERE_RADIUS])),
        density=self.BOX_DENSITY / 2.,
        friction=[1.2, 0.003, 0.05],
        user=[c.GROUP_OBJECTS],
        solref=[0.018, 0.2],
        condim=6,
        rgba=np.ones((4,)),
        material='interactobj',
        priority=1)
    box_config = {'bodies': {'box': ([ball.to_xml_string()], '')}}
    merge(config, box_config)
    return config

  @property
  def placement_extents(self) -> Tuple[float, float, float, float]:
    return -1.75, -1.75, 1.75, 1.75
