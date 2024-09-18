from typing import Dict, Tuple

import numpy as np

from dm_control.utils.rewards import tolerance
from safe_adaptation_gym import utils
from safe_adaptation_gym.tasks.push_box import PushBox
from safe_adaptation_gym.tasks.task import MujocoBridge


class PushBoxScarce(PushBox):
    def __init__(self):
        super(PushBoxScarce, self).__init__()

    def setup_placements(self) -> Dict[str, tuple]:
        placements = super(PushBox, self).setup_placements()
        placements.update(
            {"box": ([(-2.25, -2.25, 2.25, 2.25)], self.BOX_KEEPOUT * 1.1)}
        )
        return placements

    def compute_reward(
        self,
        layout: dict,
        placements: dict,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ) -> Tuple[float, bool, dict]:
        goal_pos = mujoco_bridge.body_pos("goal")[:2]
        robot_pos = mujoco_bridge.body_pos("robot")[:2]
        box_pos = mujoco_bridge.body_pos("box")[:2]
        box_distance = np.linalg.norm(robot_pos - box_pos)
        reward = tolerance(
            box_distance,
            (0, self.GOAL_SIZE * 1.80),
            margin=0.0,
            value_at_margin=0.0,
            sigmoid="linear",
        ) * (self._last_box_distance - box_distance)
        self._last_box_distance = box_distance
        box_goal_distance = np.linalg.norm(box_pos - goal_pos)
        reward += self._last_box_goal_distance - box_goal_distance
        self._last_box_goal_distance = box_goal_distance
        info = {}
        if box_goal_distance <= self.GOAL_SIZE:
            info["goal_met"] = True
            utils.update_layout(layout, mujoco_bridge)
            self.reset(layout, placements, rs, mujoco_bridge)
            reward += 1.0
        return reward, False, info
