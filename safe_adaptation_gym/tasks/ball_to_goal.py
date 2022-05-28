import numpy as np
from dm_control import mjcf

from safe_adaptation_gym.tasks.push_box import PushBox, GoToGoal
from safe_adaptation_gym.utils import merge, convert_to_text
import safe_adaptation_gym.consts as c


class BallToGoal(PushBox):
  SPHERE_RADIUS = 0.14
  BOX_KEEPOUT = 0.2
  BOX_SIZE = SPHERE_RADIUS

  def __init__(self):
    super(PushBox, self).__init__()

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
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
        density=0.1,
        friction=[0.8, 0.01, 0.01],
        user=[c.GROUP_OBJECTS],
        solref=[0.02, 0.1],
        condim=6,
        rgba=[1., 1., 1., 1.],
        material='ukraine',
        priority=1)
    box_config = {'bodies': {'box': ([ball.to_xml_string()], '')}}
    merge(config, box_config)
    return config
