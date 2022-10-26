from typing import Callable, Optional, Tuple, Union

import dm_control.rl.control
import gym
from gym import spaces
import numpy as np

from safe_adaptation_gym.mujoco_bridge import MujocoBridge
from safe_adaptation_gym.tasks import Task

Action = np.ndarray
Observation = np.ndarray


class SafeAdaptationGym(gym.Env):
    def __init__(
        self,
        rgb_observation: bool = False,
        max_bound: float = 25,
        render_options=dict(),
    ):
        self.max_bound = max_bound
        self.render_options = render_options
        self._rgb_observation = rgb_observation
        self.mujoco_bridge = MujocoBridge()
        self.task: Optional[Task] = None
        self._seed = np.random.randint(2**32)
        self.rs = np.random.RandomState(self._seed)
        self._action_space = spaces.Box(
            -1, 1, (self.mujoco_bridge.nu,), dtype=np.float32
        )
        if self._rgb_observation:
            self.observation_space = spaces.Box(0, 255, (64, 64, 3), np.float32)
        else:
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(90,), dtype=np.float32
            )

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Take a step and return observation, reward, done, and info"""
        assert (
            self.task is not None
        ), "Should set a task and reset before stepping in the environment."
        action = np.array(action, copy=True)
        action_range = self.mujoco_bridge.actuator_ctrlrange
        self.mujoco_bridge.set_control(
            np.clip(action, action_range[:, 0], action_range[:, 1])
        )
        try:
            self.mujoco_bridge.physics.step(nstep=5)
        except dm_control.rl.control.PhysicsError as er:
            print("PhysicsError", er)
            return self.observation, -10.0, True, {"cost": 0.0}
        self.mujoco_bridge.physics.forward()
        reward = self.task.compute_reward(self.mujoco_bridge)
        cost = self.task.compute_cost(self.mujoco_bridge)
        info = {"cost": cost, "bound": self.task.bound}
        observation = self.observation
        return observation, reward, False, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[Observation, Tuple[Observation, dict]]:
        """Reset the physics simulation and return observation"""
        assert self.task is not None or (
            options is not None and "task" in options
        ), "A task should be first set before reset."
        if seed is not None:
            self._seed = seed
        else:
            self._seed += 1
        self.rs = np.random.RandomState(self._seed)
        if options is not None and "task" in options:
            self.set_task(options["task"])
            return self.observation
        assert self.task is not None, "Set task before reset."
        self.task.reset(self.rs, self.mujoco_bridge)
        return self.observation

    def render(self, mode="human"):
        """Renders the mujoco simulator"""
        return self.mujoco_bridge.physics.render(**self.render_options)

    def seed(self, seed=None):
        """Set internal random state seeds"""
        self._seed = np.random.randint(2**32) if seed is None else seed
        self.rs = np.random.RandomState(self._seed)

    @property
    def observation(self) -> np.ndarray:
        if self._rgb_observation:
            image = self.mujoco_bridge.physics.render(
                height=64, width=64, camera_id="egocentric"  # type: ignore
            )
            image = np.clip(image, 0, 255).astype(np.uint8)
            return image
        ego = self.mujoco_bridge.physics.egocentric_state()
        torso_velocity = self.mujoco_bridge.physics.torso_velocity()
        torso_upright = self.mujoco_bridge.physics.torso_upright()
        imu = self.mujoco_bridge.physics.imu()
        force_torque = self.mujoco_bridge.physics.force_torque()
        proprio = np.concatenate(
            [ego, torso_velocity, torso_upright[None], imu, force_torque]
        )
        task_observation = self.task_observations
        obs = np.concatenate([task_observation, proprio])
        return obs

    @property
    def task_observations(self) -> np.ndarray:
        assert self.task is not None
        ball, goal = self.task.body_positions(self.mujoco_bridge)
        return np.concatenate([ball, goal])

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def set_task(self, task_ctor: Callable[[np.random.RandomState, float], Task]):
        """Sets a new task to be solved"""
        self.task = task_ctor(self.rs, self.max_bound)
        self.mujoco_bridge.build(self.task)
