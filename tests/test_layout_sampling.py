import numpy as np
import pytest

from learn2learn_safely import tasks
from learn2learn_safely.mujoco_bridge import MujocoBridge
from learn2learn_safely.utils import ResamplingError
from learn2learn_safely.world import World


@pytest.fixture(params=[tasks.GoToGoal()])
def world(request):
  world = World(np.random.RandomState(0), request.param, 'xmls/doggo.xml')
  return world


def test_layout_sampling(world):
  mujoco_bridge = MujocoBridge(world.sample_layout())
  fails = 0
  for _ in range(1000):
    try:
      mujoco_bridge.rebuild(world.sample_layout(), state=False)
    except ResamplingError:
      fails += 1
  # Make sure that the fail rate is smaller than 0.5%
  assert fails <= 5
