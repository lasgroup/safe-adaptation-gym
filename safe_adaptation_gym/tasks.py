from __future__ import annotations
import abc
from typing import TYPE_CHECKING, List

# Define as generic type instead of import the actual mujoco_bridge as it
# loads resources (e.g. GPU pointers) that should not exist on a parent process.
if TYPE_CHECKING:
    from safe_adaptation_gym.mujoco_bridge import MujocoBridge

import numpy as np
from dm_control.suite import quadruped

from safe_adaptation_gym import consts


class Task(abc.ABC):
    def __init__(self, rs: np.random.RandomState, max_bound: float):
        self.obstacle_scales = rs.standard_cauchy(len(consts.OBSTACLES))
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
    def observation(self, mujoco_bridge: MujocoBridge) -> np.ndarray:
        """
        Queries mujoco_bridge for current positions of task-relevant bodies.
        """

    @abc.abstractmethod
    def reset(
        self,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ):
        """
        An interface for reseting tasks's internal state.
        """

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


class Fetch(Task):
    def __init__(self, rs: np.random.RandomState, max_bound: float):
        super().__init__(rs, max_bound)
        self.task = quadruped.Fetch(rs)
        self._last_target_ball_distance = None
        self._last_robot_ball_distance = None

    def observation(self, mujoco_bridge: MujocoBridge) -> np.ndarray:
        obs_dict = self.task.get_observation(mujoco_bridge.physics)
        obs = np.concatenate(
            [value if value.ndim == 1 else value[None] for value in obs_dict.values()]
        )
        return obs

    def reset(self, rs: np.random.RandomState, mujoco_bridge: MujocoBridge):
        self.task.initialize_episode(mujoco_bridge.physics)
        self._last_robot_ball_distance = mujoco_bridge.physics.self_to_ball_distance()
        self._last_target_ball_distance = (
            mujoco_bridge.physics.ball_to_target_distance()
        )

    def compute_reward(
        self,
        mujoco_bridge: MujocoBridge,
    ) -> float:
        physics = mujoco_bridge.physics
        assert (
            self._last_target_ball_distance is not None
            and self._last_robot_ball_distance is not None
        )
        self_to_ball = physics.self_to_ball_distance()
        reach_reward = self._last_robot_ball_distance - self_to_ball
        target_to_ball = physics.ball_to_target_distance()
        fetch_reward = self._last_target_ball_distance - target_to_ball
        reward = fetch_reward + reach_reward
        self._last_target_ball_distance = target_to_ball
        self._last_robot_ball_distance = self_to_ball
        if target_to_ball <= physics.named.model.site_size["target", 0]:
            reward += 1.0
        return float(quadruped._upright_reward(physics) * reward)
