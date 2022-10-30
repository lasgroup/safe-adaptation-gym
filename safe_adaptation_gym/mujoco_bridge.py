from itertools import product
from typing import Iterable, Optional

import numpy as np

from safe_adaptation_gym.tasks import Task


def _load_model():
    from dm_control.suite import quadruped
    from lxml import etree
    from dm_control import mjcf

    xml_string = quadruped.make_model(walls_and_ball=True, rangefinders=True)
    parser = etree.XMLParser(remove_blank_text=True)
    xml_tree = etree.XML(xml_string, parser)
    xml_tree.find('.//map').attrib.pop('znear')
    xml_string = etree.tostring(xml_tree)
    model = mjcf.from_xml_string(xml_string, assets=quadruped.common.ASSETS)
    model.option.timestep = 0.01
    for wall_name, (sign, loc) in zip(quadruped._WALLS, product((-1, 1), (0, 1))):
        wall = model.find('geom', wall_name)
        wall.pos[loc] = sign * 10.
        wall.size[1 - loc] = 10.
    model.find("geom", "floor").size[:2] = 10.0
    model.visual.map.znear = 0.005
    return quadruped.Physics.from_xml_string(model.to_xml_string(), assets=quadruped.common.ASSETS)


class MujocoBridge:
    def __init__(self):
        self.physics = _load_model()
        self.build()

    def get_sensor(self, name: str) -> np.ndarray:
        return self.physics.named.data.sensordata[name]

    def build(self, task: Optional[Task] = None):
        self.physics = _load_model()
        self.physics.forward()

    def robot_contacts(self, group_geom_names: Iterable[str]) -> int:
        """
        Check if any of the geoms in group_geom_names is in contact with any of
        the geom names attached to the robot. Returns the number of contacts with
        the robot
        """
        return 0

    def robot_pos(self) -> np.ndarray:
        """Get the position of the robot in the simulator world reference frame"""
        return self.body_pos("robot")

    def robot_mat(self) -> np.ndarray:
        """Get the rotation matrix of the robot in the simulator world reference
        frame"""
        return self.body_mat("robot")

    def body_pos(self, name: str) -> np.ndarray:
        """Get the position of a named body in the simulator world reference
        frame"""
        return self.physics.named.data.xpos[name]

    def body_mat(self, name: str) -> np.ndarray:
        """Get the rotation matrix of a named body in the simulator world
        reference frame"""
        return self.physics.named.data.xmat[name].reshape([3, 3])

    def set_control(self, action: np.ndarray):
        """Sets the control"""
        self.physics.set_control(action)

    @property
    def time(self):
        return self.physics.data.time  # type: ignore

    @property
    def nu(self):
        return self.physics.model.nu

    @property
    def actuator_ctrlrange(self):
        return self.physics.model.actuator_ctrlrange
