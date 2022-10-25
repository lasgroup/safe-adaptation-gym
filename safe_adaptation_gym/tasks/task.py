import abc
from typing import List, Tuple, TypeVar

import numpy as np

from dm_control.utils import rewards

from safe_adaptation_gym import consts

# Define as generic type instead of import the actual mujoco_bridge as it
# loads resources (e.g. GPU pointers) that should not exist on a parent process.
MujocoBridge = TypeVar("MujocoBridge")


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
    ) -> Tuple[float, bool, dict]:
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
    def body_positions(self, mujoco_bridge: MujocoBridge) -> Tuple[List, List, List]:
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
        from dm_control.suite.quadruped import _find_non_contacting_height

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
