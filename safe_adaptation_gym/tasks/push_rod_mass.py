import numpy as np
from dm_control import mjcf

from safe_adaptation_gym.tasks.push_box import PushBox, GoToGoal
from safe_adaptation_gym.utils import merge
import safe_adaptation_gym.consts as c


class PushRodMass(PushBox):
  ROD_LENGTH = 0.4
  ROD_RADIUS = 0.08
  BOX_KEEPOUT = 0.7
  BOX_SIZE = 0.25

  def __init__(self):
    super(PushBox, self).__init__()

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    config = GoToGoal.build_world_config(self, layout, rs)
    model = mjcf.RootElement()
    rod_root = model.worldbody.add(
        'body',
        name='box',
        euler='88 0 0',
        pos=[layout['box'][0], layout['box'][1], self.ROD_RADIUS])
    rod_root.add('freejoint', name='box')
    rod_root.add(
        'geom',
        name='box',
        type='cylinder',
        size=[self.ROD_RADIUS, self.ROD_LENGTH],
        density=0.1,
        friction=[0.30, 0.001, 0.05],
        rgba=[1, 1, 1, 1],
        material='pine',
        user=[c.GROUP_OBJECTS],
        condim=6,
        priority=1)
    box_config = {'bodies': {'box': ([rod_root.to_xml_string()], '')}}
    merge(config, box_config)
    return config
