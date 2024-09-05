from typing import Tuple

import numpy as np
from dm_control.utils.rewards import tolerance

from safe_adaptation_gym.tasks.press_buttons import PressButtons, State
from safe_adaptation_gym.tasks.task import MujocoBridge


class PressButtonsScarce(PressButtons):
    def __init__(self):
        super(PressButtonsScarce, self).__init__()

    def compute_reward(
        self,
        layout: dict,
        placements: dict,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ) -> Tuple[float, bool, dict]:
        goal_pos = mujoco_bridge.body_pos(self._goal_button)[:2]
        robot_pos = mujoco_bridge.body_pos("robot")[:2]
        distance = np.linalg.norm(robot_pos - goal_pos)
        self._last_goal_distance = distance
        reward = tolerance(
            distance,
            (0, 0.30),
            margin=0.0,
            value_at_margin=0.0,
            sigmoid="linear",
        ) * (self._last_goal_distance - distance)
        info = {}
        if mujoco_bridge.robot_contacts([self._goal_button]):
            reward += 1.0
            info["goal_met"] = True
            self._sample_goal_button(rs, mujoco_bridge)
            self._state = State.BUTTON_CHANGE
        elif mujoco_bridge.robot_contacts(
            [
                f"button{i}"
                for i in self.NUM_BUTTONS
                if f"button{i}" != self._goal_button
            ]
        ):
            reward -= 1.0
        if self._state == State.BUTTON_CHANGE:
            if not self._goal_button_timer.done:
                self._goal_button_timer.tick()
            else:
                self._state = State.NORMAL
                self._goal_button_timer.reset()
        self._update_goal_button(mujoco_bridge)
        return reward, False, info
