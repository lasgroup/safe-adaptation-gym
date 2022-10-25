from typing import Tuple, List

import numpy as np

from dm_control.utils import rewards

from safe_adaptation_gym.tasks.task import MujocoBridge, Task, upright_reward


class Fetch(Task):
    def __init__(self, rs: np.random.RandomState, max_bound: int = 25):
        super().__init__(rs, max_bound)

    def body_positions(self, mujoco_bridge: MujocoBridge) -> Tuple[List, List, List]:
        return [], [], []

    # Copyright 2019 The dm_control Authors.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #    http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ============================================================================
    def compute_reward(
        self,
        mujoco_bridge: MujocoBridge,
    ) -> Tuple[float, bool, dict]:
        physics = mujoco_bridge.physics
        arena_radius = physics.named.model.geom_size["floor", 0] * np.sqrt(2)
        workspace_radius = physics.named.model.site_size["workspace", 0]
        ball_radius = physics.named.model.geom_size["ball", 0]
        reach_reward = rewards.tolerance(
            physics.self_to_ball_distance(),
            bounds=(0, workspace_radius + ball_radius),
            sigmoid="linear",
            margin=arena_radius,
            value_at_margin=0,
        )
        # Reward for bringing the ball to the target.
        target_radius = physics.named.model.site_size["target", 0]
        fetch_reward = rewards.tolerance(
            physics.ball_to_target_distance(),
            bounds=(0, target_radius),
            sigmoid="linear",
            margin=arena_radius,
            value_at_margin=0,
        )
        reach_then_fetch = reach_reward * (0.5 + 0.5 * fetch_reward)
        return upright_reward(physics) * reach_then_fetch
