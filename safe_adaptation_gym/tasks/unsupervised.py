from typing import Dict, Tuple

import numpy as np

from safe_adaptation_gym import utils
from safe_adaptation_gym.tasks.press_buttons import PressButtons
from safe_adaptation_gym.tasks.push_box import PushBox
from safe_adaptation_gym.tasks.task import MujocoBridge, Task


class Unsupervised(Task):
    def __init__(self):
        super(Unsupervised, self).__init__()
        self.buttons = PressButtons()
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
        return 0.0, False, {}

    def reset(
        self,
        layout: dict,
        placements: dict,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ):
        return self.box.reset(layout, placements, rs, mujoco_bridge)

    def set_mocaps(self, mujoco_bridge: MujocoBridge):
        self.buttons.set_mocaps(mujoco_bridge)

    @property
    def num_obstacles(self) -> int:
        return 12
