import os

import numpy as np
from dm_control import mjcf

from safe_adaptation_gym.tasks.push_box import PushBox, GoToGoal
from safe_adaptation_gym.utils import merge, convert_to_text
import safe_adaptation_gym.consts as c


class BallToGoal(PushBox):
  SPHERE_RADIUS = 0.1

  def __init__(self):
    super(PushBox, self).__init__()

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    config = GoToGoal.build_world_config(self, layout, rs)
    model = mjcf.RootElement()
    texture = model.asset.add(
        'texture',
        name='ukraine',
        file=os.path.join(c.BASE_DIR, 'textures', 'ukraine.png'))
    material = model.asset.add('material', name='ukraine', texture=texture)
    ball = model.worldbody.add('body', name='box')
    ball.add('freejoint', name='box')
    ball.add(
        'geom',
        name='box',
        type='sphere',
        size=convert_to_text(np.array([self.SPHERE_RADIUS])),
        density=0.125,
        friction=[0.6, 0.05, 0.04],
        pos=[0, 0, self.SPHERE_RADIUS],
        user=[c.GROUP_OBJECTS],
        solref=[0.02, 0.1],
        condim=6,
        material=material,
        priority=1)
    box_config = {'bodies': {'box': ([ball.to_xml_string()], '')}}
    merge(config, box_config)
    return config
