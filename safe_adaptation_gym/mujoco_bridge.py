from typing import Iterable

import numpy as np
from dm_control.suite import quadruped


class MujocoBridge:
    def __init__(self, addition_render_objects_specs=None):
        self.physics = None
        self._addition_render_objects_specs = addition_render_objects_specs
        self._build()

    def get_sensor(self, name: str) -> np.ndarray:
        return self.physics.named.data.sensordata[name]

    def _build(self):
        xml_string = quadruped.make_model(walls_and_ball=True)
        self.physics = quadruped.Physics.from_xml_string(
            xml_string, quadruped.common.ASSETS
        )
        self.physics.forward()

    def rebuild(self, config):
        """Build a new physics from a model if the model changed"""
        self._build()

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
        return self.physics.data.time

    @property
    def nu(self):
        return self.physics.model.nu

    @property
    def actuator_ctrlrange(self):
        return self.physics.model.actuator_ctrlrange

    @property
    def user_groups(self):
        return self.physics.named.model.geom_user

    @property
    def site_rgba(self):
        return self.physics.named.model.site_rgba
