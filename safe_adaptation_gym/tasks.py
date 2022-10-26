from __future__ import annotations
import abc
from typing import TYPE_CHECKING, List, Tuple

# Define as generic type instead of import the actual mujoco_bridge as it
# loads resources (e.g. GPU pointers) that should not exist on a parent process.
if TYPE_CHECKING:
    from safe_adaptation_gym.mujoco_bridge import MujocoBridge

import numpy as np
from dm_control.suite.quadruped import _find_non_contacting_height
from dm_control.utils import rewards

from safe_adaptation_gym import consts


class Task(abc.ABC):
    def __init__(self, rs: np.random.RandomState, max_bound: int = 25):
        self.obstacle_scales = rs.standard_cauchy(len(consts.OBSTACLES))
        # if max_joints_to_disable == 0:
        #     self._joints = []
        # else:
        #     num_joints_to_disable = rs.randint(max_joints_to_disable + 1)
        #     self._joints = doggo_joints_sampler.disable_joints(
        #         rs, num_joints_to_disable
        #     )
        # if cripple_leg:
        #     self._joints += doggo_joints_sampler.cripple_leg(rs)
        self.joints = None
        self.bound = rs.uniform(0.0, max_bound)

    @abc.abstractmethod
    def compute_reward(
        self,
        mujoco_bridge: MujocoBridge,
    ) -> float:
        """
        Computes the task-specific reward. Exposes the layout and placements for
        goal/object position resampling
        """

    def compute_cost(self, mujoco_bridge: MujocoBridge) -> float:
        """
        Computes the task specific costs.
        """
        return 0.0

    @abc.abstractmethod
    def body_positions(
        self, mujoco_bridge: MujocoBridge
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Queries mujoco_bridge for current positions of task-relevant bodies.
        """

    def reset(
        self,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ):
        """
        Not sure about this yet. But should allow an interface to build to task
        specific stuff
        """

        azimuth = rs.uniform(0, 2 * np.pi)
        orientation = np.array((np.cos(azimuth / 2), 0, 0, np.sin(azimuth / 2)))
        spawn_radius = 0.9 * mujoco_bridge.physics.named.model.geom_size["floor", 0]
        x_pos, y_pos = rs.uniform(-spawn_radius, spawn_radius, size=(2,))
        _find_non_contacting_height(mujoco_bridge.physics, orientation, x_pos, y_pos)

    @property
    def obstacles_distribution(self) -> List[float]:
        """
        How many obstacle instances are sampled for each obstacle type.
        Type order follows: 'hazards', 'vases', 'gremlins', 'pillars',
        as in consts.OBSTACLES
        """
        return [1 / len(consts.OBSTACLES)] * len(consts.OBSTACLES)

    @property
    def num_obstacles(self) -> int:
        return 20


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
def upright_reward(physics, deviation_angle=0):
    """Returns a reward proportional to how upright the torso is.
    Args:
      physics: an instance of `Physics`.
      deviation_angle: A float, in degrees. The reward is 0 when the torso is
        exactly upside-down and 1 when the torso's z-axis is less than
        `deviation_angle` away from the global z-axis.
    """
    deviation = np.cos(np.deg2rad(deviation_angle))
    return rewards.tolerance(
        physics.torso_upright(),
        bounds=(deviation, float("inf")),
        sigmoid="linear",
        margin=1 + deviation,
        value_at_margin=0,
    )


class Fetch(Task):

    def body_positions(
        self, mujoco_bridge: MujocoBridge
    ) -> Tuple[np.ndarray, np.ndarray]:
        ball_state = mujoco_bridge.physics.ball_state()
        target_position = mujoco_bridge.physics.target_position()
        return ball_state, target_position

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
    ) -> float:
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
        return float(upright_reward(physics) * reach_then_fetch)
