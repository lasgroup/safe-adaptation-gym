import numpy as np
import pytest

import matplotlib.pyplot as plt

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
      mujoco_bridge.reset(world.sample_layout())
    except ResamplingError:
      fails += 1
  # Make sure that the fail rate is smaller than 0.5%
  assert fails <= 5


def test_reset(world):
  mujoco_bridge = MujocoBridge(world.sample_layout())
  plt.figure()
  plt.imshow(mujoco_bridge.physics.render(camera_id='fixedfar'))
  init_state = mujoco_bridge.physics.get_state().copy()
  mujoco_bridge.reset(world.sample_layout())
  plt.figure()
  plt.imshow(mujoco_bridge.physics.render(camera_id='fixedfar'))
  mujoco_bridge.physics.reset()
  mujoco_bridge.physics.forward()
  plt.figure()
  plt.imshow(mujoco_bridge.physics.render(camera_id='fixedfar'))
  plt.show()
  assert (init_state == mujoco_bridge.physics.get_state()).all() # noqa
