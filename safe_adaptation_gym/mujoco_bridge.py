from typing import Iterable, Optional

import numpy as np

from safe_adaptation_gym.tasks import Task


class MujocoBridge:
    def __init__(self):
        from dm_control.suite import quadruped
        from lxml import etree
        xml_string = quadruped.make_model(walls_and_ball=True, rangefinders=True)
        parser = etree.XMLParser(remove_blank_text=True)
        mjcf = etree.XML(xml_string, parser)
        mjcf.find('option').attrib['timestep'] = '0.01'
        self.physics = quadruped.Physics.from_xml_string(
            etree.tostring(mjcf, pretty_print=True), quadruped.common.ASSETS
        )
        self.build()

    def get_sensor(self, name: str) -> np.ndarray:
        return self.physics.named.data.sensordata[name]

    def build(self, task: Optional[Task] = None):
        from dm_control.suite import quadruped
        from dm_control.suite import quadruped
        from lxml import etree
        xml_string = quadruped.make_model(walls_and_ball=True, rangefinders=True)
        parser = etree.XMLParser(remove_blank_text=True)
        mjcf = etree.XML(xml_string, parser)
        mjcf.find('option').attrib['timestep'] = '0.01'
        self.physics = quadruped.Physics.from_xml_string(
            etree.tostring(mjcf, pretty_print=True), quadruped.common.ASSETS
        )
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
