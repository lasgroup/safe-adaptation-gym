import pytest

from gym.wrappers import TimeLimit

from learn2learn_safely import tasks
from learn2learn_safely.safety_gym import SafetyGym


@pytest.fixture(params=[tasks.GoToGoal(), tasks.PushBox()])
def safety_gym(request):
  arena = TimeLimit(SafetyGym('xmls/doggo.xml'), 1000)
  arena.set_task(request.param)
  return arena


def test_episode(safety_gym):
  safety_gym.reset()
  done = False
  while not done:
    *_, done, _ = safety_gym.step(safety_gym.action_space.sample())
  assert True

