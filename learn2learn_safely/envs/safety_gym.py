import gym


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
  ) -> Union[ObsType, tuple[ObsType, dict]]:
    pass

  def render(self, mode="human"):
    pass
