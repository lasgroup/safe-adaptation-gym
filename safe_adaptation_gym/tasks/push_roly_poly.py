import numpy as np
from dm_control import mjcf

from safe_adaptation_gym.tasks.push_box import PushBox, GoToGoal
from safe_adaptation_gym.utils import merge, convert_to_text
import safe_adaptation_gym.consts as c


class PushRolyPoly(PushBox):
  # For the interested: https://en.wikipedia.org/wiki/Roly-poly_toy and
  # https://he.wikipedia.org/wiki/%D7%A0%D7%97%D7%95%D7%9D_%D7%AA%D7%A7%D7%95%D7%9D
  SPHERE_SIZE = 0.17
  ROLY_POLY_COLOR = np.array([1., 0.94, 0.68, 0.5])

  def __init__(self):
    super(PushBox, self).__init__()

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    config = GoToGoal.build_world_config(self, layout, rs)
    model = mjcf.RootElement()
    rolypoly = model.worldbody.add('body', name='box')
    rolypoly.add('freejoint', name='box')
    rolypoly.add(
        'geom',
        name='box',
        type='ellipsoid',
        size='0.1 0.1 0.15',
        density='0.0001',
        rgba=convert_to_text(self.ROLY_POLY_COLOR),
        pos='0 0 0.22',
        user=[c.GROUP_OBJECTS])
    rolypoly.add(
        'geom',
        name='box_base',
        type='sphere',
        size=convert_to_text(self.SPHERE_SIZE),
        density='0.0001',
        rgba=convert_to_text(self.ROLY_POLY_COLOR),
        pos=convert_to_text(np.array([0, 0, self.SPHERE_SIZE])),
        friction='.7 .005 .005',
        solref='-100 -60',
        condim='6',
        priority='1',
        user=[c.GROUP_OBJECTS])
    rolypoly.add(
        'geom',
        name='ballast',
        type='sphere',
        size="0.1",
        contype='0',
        conaffinity='0',
        group='3',
        density='0.0008',
        pos='0 0 -0.05')
    box_config = {'bodies': {'box': ([rolypoly.to_xml_string()], '')}}
    merge(config, box_config)
    return config
