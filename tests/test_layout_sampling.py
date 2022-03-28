import numpy as np
import pytest

from learn2learn_safely import tasks
from learn2learn_safely.world import World
from learn2learn_safely.utils import ResamplingError
from learn2learn_safely.mujoco_bridge import MujocoBridge


@pytest.fixture(params=[tasks.GoToGoal()])
def world(request):
  config = {
      'obstacles_size_noise': 0.00,
      'keepout_noise': 0.00,
      'num_obstacles': 10
  }
  world = World(
      np.random.RandomState(0), request.param, 'xmls/doggo.xml', config)
  return world


def test_layout_sampling(world):
  world_config = world.build()
  mujoco_bridge = MujocoBridge(world_config)
  mujoco_bridge.build()
  fails = 0
  for _ in range(1000):
    try:
      world_config = world.build()
      mujoco_bridge.rebuild(world_config, state=False)
    except ResamplingError:
      fails += 1
  # Make sure that the fail rate is smaller than 0.5%
  assert fails <= 5
