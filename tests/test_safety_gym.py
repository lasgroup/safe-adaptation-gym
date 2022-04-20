import numpy as np
import pytest
from dm_control import viewer
from dm_env import specs, TimeStep, StepType
from gym.wrappers import TimeLimit

from safe_adaptation_gym import tasks
from safe_adaptation_gym.safety_gym import SafeAdaptationGym


@pytest.fixture(
    params=[
            tasks.PushRodMass()])
def safety_gym(request):
  arena = TimeLimit(
      SafeAdaptationGym('xmls/doggo.xml', render_lidars_and_collision=True), 1000)
  arena.seed(2)
  arena.set_task(request.param)
  return arena


def test_viewer(safety_gym):

  class ViewerWrapper:

    def __init__(self, env):
      self.env = env
      self._action_spec = specs.BoundedArray(
          shape=env.action_space.shape,
          dtype=np.float32,
          minimum=env.action_space.low,
          maximum=env.action_space.high)

    @property
    def physics(self):
      return self.env.mujoco_bridge.physics

    def reset(self):
      self.env.reset()

    def action_spec(self):
      return self._action_spec

    def step(self, action):
      obs, reward, terminal, info = self.env.step(action)
      return TimeStep(StepType.MID, reward, 1.0, obs)

  def sample(*args, **kwargs):
    return np.zeros_like(safety_gym.action_space.sample())

  viewer.launch(ViewerWrapper(safety_gym), sample)


def test_episode(safety_gym):
  safety_gym.reset()
  done = False
  while not done:
    *_, done, _ = safety_gym.step(safety_gym.action_space.sample())
  assert True
