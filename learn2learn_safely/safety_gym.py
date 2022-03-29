from typing import Tuple, Union, Optional

import dm_control.rl.control
import gym
import numpy as np
from gym.core import ActType, ObsType

from learn2learn_safely.mujoco_bridge import MujocoBridge
from learn2learn_safely.tasks.task import Task
from learn2learn_safely.world import World


# TODO (yarden): make sure that the environment is wrapped with a timelimit
class SafetyGym(gym.Env):

  def __init__(self, robot_base, config=None):
    self._world: Optional[World] = None
    self.base_config = config
    self.robot_base = robot_base
    self.mujoco_bridge = MujocoBridge()
    self._seed = np.random.randint(2**32)
    self.rs = np.random.RandomState(self._seed)

  def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
    """ Take a step and return observation, reward, done, and info """
    action = np.array(action, copy=False)
    action_range = self.mujoco_bridge.model.actuator_ctrlrange
    # Action space is in [-1, 1] by definition. Each time we sample a new
    # world (a.k.a. CMDP), we sample a new scale (see _build_world_config) to
    # accomodate variability in the dynamics.
    action = (
        action * action_range + self._world.config.action_noise *
        self.rs.normal(size=self.mujoco_bridge.model.nu))
    self.mujoco_bridge.data.ctrl[:] = np.clip(action, action_range[:, 0],
                                              action_range[:, 1])
    # Keeping magic 10 from Safety-Gym
    # https://github.com/openai/safety-gym/blob
    # /f31042f2f9ee61b9034dd6a416955972911544f5/safety_gym/envs/engine.py#L1253
    for _ in range(10):
      try:
        self._world.set_mocaps()
        self.mujoco_bridge.physics.step()
      except dm_control.rl.control.PhysicsError as er:
        print('PhysicsError', er)
        return self.observation, -10., True, {'cost': 0.}
    self.mujoco_bridge.physics.forward()
    reward, terminal, info = self._world.compute_reward(self.mujoco_bridge)
    info = {'cost': self._world.compute_cost()}
    return self.observation, reward, terminal, info

  def reset(
      self,
      *,
      seed: Optional[int] = None,
      return_info: bool = False,
      options: Optional[dict] = None,
  ) -> Union[ObsType, Tuple[ObsType, dict]]:
    """ Reset the physics simulation and return observation """
    assert self._world is not None, 'A task should be first set before reset.'
    self._seed += 1
    self.rs = np.random.RandomState(self._seed)
    self._world.rs = self.rs
    self.mujoco_bridge.rebuild(self._world.sample_layout())
    return self.observation

  def render(self, mode="human"):
    """ Renders the mujoco simulator """
    pass

  def seed(self, seed=None):
    """ Set internal random state seeds """
    self._seed = np.random.randint(2**32) if seed is None else seed
    self.rs = np.random.RandomState(self._seed)

  @property
  def observation(self):
    pass

  def set_task(self, task: Task):
    """ Sets a new task to be solved """
    self._world = World(self.rs, task, self.robot_base, self.base_config)
    self.mujoco_bridge.rebuild(self._world.sample_layout())
