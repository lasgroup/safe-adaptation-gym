from typing import Tuple, Union, Optional
import gym
from gym.core import ActType, ObsType

import numpy as np

from learn2learn_safely.world import World
from learn2learn_safely.tasks import GoToGoal
from learn2learn_safely.mujoco_bridge import MujocoBridge


# TODO (yarden): what is the best interface to sample and load new tasks?
class SafetyGym(gym.Env):

  def __init__(self):
    pass

  def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
    action = np.array(action, copy=False)
    action_range = self.world.model.actuator_ctrlrange
    self.data.ctrl[:] = np.clip(action, action_range[:, 0],
                                action_range[:, 1])  # np.clip(
    # action * 2 / action_scale, -1, 1)
    if self.action_noise:
      self.data.ctrl[:] += self.action_noise * self.rs.randn(self.model.nu)

  def reset(
      self,
      *,
      seed: Optional[int] = None,
      return_info: bool = False,
      options: Optional[dict] = None,
  ) -> Union[ObsType, Tuple[ObsType, dict]]:
    pass

  def render(self, mode="human"):
    pass
