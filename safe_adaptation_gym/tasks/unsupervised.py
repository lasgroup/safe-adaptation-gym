from typing import Dict, Tuple

import numpy as np

from safe_adaptation_gym import utils
from safe_adaptation_gym.primitive_objects import geom_attributes_to_xml
from safe_adaptation_gym.tasks.go_to_goal import GoToGoal
from safe_adaptation_gym.tasks.task import MujocoBridge, Task
import safe_adaptation_gym.consts as c


class Unsupervised(Task):
    _CIRCLE_RADIUS = 1.5

    def __init__(self):
        super(Unsupervised, self).__init__()
        self.goal = GoToGoal()

    def setup_placements(self) -> Dict[str, tuple]:
        placements = self.goal.setup_placements()
        return placements

    def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
        goal_config = self.goal.build_world_config(layout, rs)
        circle_dict = {
            "name": "circle",
            "size": np.array([self._CIRCLE_RADIUS, 1e-2]),
            "pos": np.array([0, 0, 2e-2]),
            "quat": utils.rot2quat(0),
            "type": "cylinder",
            "contype": 0,
            "conaffinity": 0,
            "rgba": np.array([0, 1, 0, 1]) * [1, 1, 1, 0.1],
            "user": [c.GROUP_GOAL],
        }
        circle_xml = geom_attributes_to_xml(circle_dict)
        circle = {
            "bodies": {
                "circle": (
                    [circle_xml],
                    "",
                )
            }
        }
        utils.merge(goal_config, circle)
        return goal_config

    def compute_reward(
        self,
        layout: dict,
        placements: dict,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ) -> Tuple[float, bool, dict]:
        goal_reward, *_ = self.goal.compute_reward(layout, placements, rs, mujoco_bridge)
        robot_com = mujoco_bridge.body_com("robot")
        robot_vel = mujoco_bridge.robot_vel()
        x, y = robot_com[:2]
        u, v = robot_vel[:2]
        radius = np.sqrt(x**2 + y**2)
        circle_reward = (
            ((-u * y + v * x) / radius) / (1 + np.abs(radius - self._CIRCLE_RADIUS))
        ) * 1e-1
        done = False
        info = {}
        reward = np.stack([circle_reward, goal_reward])
        return reward, done, info

    def reset(
        self,
        layout: dict,
        placements: dict,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ):
        return self.goal.reset(layout, placements, rs, mujoco_bridge)

    @property
    def obstacles(self) -> int:
        return [5, 6, 0, 1]

    def set_mocaps(self, mujoco_bridge: MujocoBridge):
        return super().set_mocaps(mujoco_bridge)
