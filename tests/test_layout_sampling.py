import numpy as np
import pytest

import matplotlib.pyplot as plt

from safe_adaptation_gym import benchmark
from safe_adaptation_gym.mujoco_bridge import MujocoBridge
from safe_adaptation_gym.utils import ResamplingError
from safe_adaptation_gym.world import World
from safe_adaptation_gym.robot import Robot


@pytest.fixture(params=benchmark.TASKS)
def good_world(request):
  world = World(
      np.random.RandomState(0), request.param[1](), Robot('xmls/doggo.xml'))
  return world


@pytest.fixture(params=benchmark.TASKS)
def bad_world(request):
  config = {
      'hazards_size': 2.0,
      'vases_size': 2.0,
      'pillars_size': 2.0,
      'gremlins_size': 2.0
  }
  world = World(
      np.random.RandomState(0), request.param[1](), Robot('xmls/doggo.xml'),
      config)
  return world


def layout_sampling(world):
  mujoco_bridge = MujocoBridge(Robot('xmls/doggo.xml'))
  fails = 0
  for _ in range(1000):
    try:
      mujoco_bridge.rebuild(world.sample_layout())
    except ResamplingError:
      fails += 1
  # Make sure that the fail rate is smaller than 0.5%
  return fails


def test_layout_sampling(good_world):
  fails = layout_sampling(good_world)
  # Make sure that the fail rate is smaller than 0.5%
  assert fails <= 5


def test_layout_impossible(bad_world):
  ok = False
  try:
    MujocoBridge(Robot('xmls/doggo.xml'))
    bad_world.sample_layout()
  except ResamplingError:
    ok = True
  assert ok
