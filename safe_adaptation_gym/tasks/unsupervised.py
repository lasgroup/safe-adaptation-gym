from typing import Dict, Tuple

import numpy as np

from safe_adaptation_gym.tasks.go_to_goal import GoToGoal
from safe_adaptation_gym.tasks.push_box import PushBox
from safe_adaptation_gym.tasks.task import MujocoBridge, Task


class Unsupervised(Task):
    def __init__(self):
        super(Unsupervised, self).__init__()
        self.box = PushBox()

    def setup_placements(self) -> Dict[str, tuple]:
        box_placements = self.box.setup_placements()
        return box_placements

    def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
        box_config = self.box.build_world_config(layout, rs)
        return box_config

    def compute_reward(
        self,
        layout: dict,
        placements: dict,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ) -> Tuple[float, bool, dict]:
        box_reward, box_done, box_info = self.box.compute_reward(
            layout, placements, rs, mujoco_bridge
        )
        goal_reward, *_ = GoToGoal.compute_reward(
            self.box, layout, placements, rs, mujoco_bridge
        )
        reward = np.stack([goal_reward, box_reward])
        done = box_done
        return reward, done, box_info

    def reset(
        self,
        layout: dict,
        placements: dict,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ):
        return self.box.reset(layout, placements, rs, mujoco_bridge)

    def set_mocaps(self, mujoco_bridge: MujocoBridge):
        pass

    @property
    def obstacles(self) -> int:
        return [5, 6, 0, 1]
