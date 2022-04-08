import numpy as np
from dm_control import mjcf

from learn2learn_safely.tasks.push_box import PushBox, GoToGoal
from learn2learn_safely.utils import merge, convert_to_text
import learn2learn_safely.consts as c


class PushRolyPoly(PushBox):
  # For the interested: https://en.wikipedia.org/wiki/Roly-poly_toy and
  # https://he.wikipedia.org/wiki/%D7%A0%D7%97%D7%95%D7%9D_%D7%AA%D7%A7%D7%95%D7%9D
  BOX_SIZE = 0.2
  BOX_KEEPOUT = 0.2
  BOX_COLOR = np.array([1, 1, 0, 0.25])
  BOX_DENSITY = 0.001

  def __init__(self):
    super(PushBox, self).__init__()

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    config = GoToGoal.build_world_config(self, layout, rs)
    model = mjcf.RootElement()
    tippetop_root = model.worldbody.add('body', name='box')
    tippetop_root.add('freejoint', name='box')
    color = np.array([1., 0.94, 0.68, 0.5])
    tippetop_root.add(
        'geom',
        name='box',
        type='ellipsoid',
        size="0.1 0.1 0.15",
        density='0.0001',
        rgba=convert_to_text(color),
        pos='0 0 0.2',
        user=[c.GROUP_OBJECTS])
    tippetop_root.add(
        'geom',
        name='box_base',
        type='sphere',
        size='0.15',
        density='0.0001',
        rgba=convert_to_text(color),
        pos='0 0 0.15',
        friction='.7 .005 .005',
        solref='-100 -60',
        condim='6',
        priority='1')
    tippetop_root.add(
        'geom',
        name='ballast',
        type='sphere',
        size="0.1",
        contype='0',
        conaffinity='0',
        group='3',
        density='0.0008',
        pos='0 0 -0.05')
    box_config = {'bodies': {'box': ([tippetop_root.to_xml_string()], '')}}
    merge(config, box_config)
    return config
