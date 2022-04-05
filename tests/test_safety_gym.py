import numpy as np
import pytest

from dm_control import viewer
from dm_env import specs
from gym.wrappers import TimeLimit

from learn2learn_safely import tasks
from learn2learn_safely.safety_gym import SafetyGym


@pytest.fixture(params=[tasks.GoToGoal(), tasks.PushBox()])
def safety_gym(request):
  arena = TimeLimit(SafetyGym('xmls/doggo.xml'), 1000)
  arena.seed(2)
  arena.set_task(request.param)
  return arena


def test_episode(safety_gym):
  safety_gym.reset()
  done = False
  while not done:
    *_, done, _ = safety_gym.step(safety_gym.action_space.sample())
  assert True


def test_viewer(safety_gym):

  class ViewerWrapper:

    def __init__(self, env):
      self.physics = env.mujoco_bridge.physics
      self.env = env
      self._action_spec = specs.BoundedArray(
          shape=env.action_space.shape,
          dtype=np.float32,
          minimum=env.action_space.low,
          maximum=env.action_space.high)

    def reset(self):
      self.env.reset()

    def action_spec(self):
      return self._action_spec

  viewer.launch(ViewerWrapper(safety_gym))
