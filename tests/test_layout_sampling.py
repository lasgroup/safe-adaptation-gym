import numpy as np
import pytest

from learn2learn_safely import objectives
from learn2learn_safely.task import Task
from learn2learn_safely.utils import ResamplingError
from learn2learn_safely.world import World


@pytest.fixture(params=[objectives.GoToGoal()])
def task(request):
  config = {
      'obstacles_size_noise': 0.00,
      'keepout_noise': 0.00,
      'num_obstacles': 10
  }
  task = Task(np.random.RandomState(0), request.param, 'xmls/doggo.xml', config)
  return task


def test_layout_sampling(task):
  world_config = task.build()
  world = World(world_config)
  world.build()
  fails = 0
  for _ in range(1000):
    try:
      world_config = task.build()
      world.rebuild(world_config, state=False)
    except ResamplingError:
      fails += 1
  # Make sure that the fail rate is smaller than 0.5%
  assert fails <= 5
