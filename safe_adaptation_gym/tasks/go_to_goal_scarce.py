from typing import Tuple

import numpy as np

from safe_adaptation_gym import utils
from safe_adaptation_gym.tasks.go_to_goal import GoToGoal
from safe_adaptation_gym.tasks.task import MujocoBridge


class GoToGoalScarce(GoToGoal):
    def __init__(self):
        super(GoToGoalScarce, self).__init__()

    def compute_reward(
        self,
        layout: dict,
        placements: dict,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ) -> Tuple[float, bool, dict]:
        goal_pos = np.asarray(mujoco_bridge.body_pos("goal"))
        robot_pos = mujoco_bridge.body_pos("robot")
        distance = np.linalg.norm(robot_pos - goal_pos)
        info = {}
        reward = 0
        if distance <= self.GOAL_SIZE:
            info["goal_met"] = True
            utils.update_layout(layout, mujoco_bridge)
            self.reset(layout, placements, rs, mujoco_bridge)
            reward += 1.0
        return reward, False, info
