import numpy as np
from dm_control import mjcf

from learn2learn_safely.tasks.push_box import PushBox, GoToGoal
from learn2learn_safely.utils import merge, convert_to_text
import learn2learn_safely.consts as c


class PushRodMass(PushBox):
  ROD_LENGTH = 0.5
  ROD_RADIUS = 0.05
  BOX_KEEPOUT = 0.7
  ROD_COLOR = np.array([0., 0., 0.5, 0.5])

  def __init__(self):
    super(PushBox, self).__init__()

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    config = GoToGoal.build_world_config(self, layout, rs)
    model = mjcf.RootElement()
    rod_root = model.worldbody.add('body', name='box', euler='88 0 0')
    rod_root.add('freejoint', name='box')
    rod_root.add(
        'geom',
        name='box',
        type='cylinder',
        size=convert_to_text(np.array([self.ROD_RADIUS, self.ROD_LENGTH])),
        density='0.001',
        rgba=convert_to_text(self.ROD_COLOR),
        pos=convert_to_text(np.array([0.0, self.ROD_RADIUS, 0.0])),
        user=[c.GROUP_OBJECTS])
    rod_root.add(
        'geom',
        name='box_base',
        type='sphere',
        size='0.07',
        density='0.0001',
        rgba=convert_to_text(self.ROD_COLOR),
        pos='0. 0.05 0.5',
        user=[c.GROUP_OBJECTS])
    box_config = {'bodies': {'box': ([rod_root.to_xml_string()], '')}}
    merge(config, box_config)
    return config
