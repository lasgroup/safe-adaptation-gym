from typing import Dict, Tuple

import numpy as np

from safe_adaptation_gym import utils
from safe_adaptation_gym.tasks.collect import Collect
from safe_adaptation_gym.tasks.push_box import PushBox
from safe_adaptation_gym.tasks.task import MujocoBridge, Task


class Unsupervised(Task):
    def __init__(self):
        super(Unsupervised, self).__init__()
        self.buttons = Collect()
        self.box = PushBox()

    def setup_placements(self) -> Dict[str, tuple]:
        buttons_placements = self.buttons.setup_placements()
        box_placements = self.box.setup_placements()
        utils.merge(box_placements, buttons_placements)
        return box_placements

    def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
        buttons_config = self.buttons.build_world_config(layout, rs)
        box_config = self.box.build_world_config(layout, rs)
        utils.merge(box_config, buttons_config)
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
        collect_reward, collect_done, collect_info = self.buttons.compute_reward(
            layout, placements, rs, mujoco_bridge
        )
        reward = box_reward + collect_reward
        done = box_done or collect_done
        info = utils.merge(box_info, collect_info)
        return reward, done, info

    def reset(
        self,
        layout: dict,
        placements: dict,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ):
        self.buttons.reset(layout, placements, rs, mujoco_bridge)
        return self.box.reset(layout, placements, rs, mujoco_bridge)

    def set_mocaps(self, mujoco_bridge: MujocoBridge):
        self.buttons.set_mocaps(mujoco_bridge)

    @property
    def obstacles(self) -> int:
        return [5, 6, 0, 1]
