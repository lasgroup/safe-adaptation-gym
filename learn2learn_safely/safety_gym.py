from typing import Tuple, Union, Optional
import gym
from gym.core import ActType, ObsType

from task import Task


# TODO (yarden): what is the best interface to sample and load new tasks?
class SafetyGym(gym.Env):

  def __init__(self):
    pass

  def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
    pass

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
