import numpy as np
import pytest

import matplotlib.pyplot as plt

from learn2learn_safely import tasks
from learn2learn_safely.mujoco_bridge import MujocoBridge
from learn2learn_safely.utils import ResamplingError
from learn2learn_safely.world import World
from learn2learn_safely.robot import Robot


@pytest.fixture(params=[tasks.GoToGoal(), tasks.PushBox()])
def good_world(request):
  world = World(
      np.random.RandomState(0), request.param, Robot('xmls/doggo.xml'))
  return world


@pytest.fixture(params=[tasks.GoToGoal(), tasks.PushBox()])
def bad_world(request):
  config = {
      'hazards_size': 2.0,
      'vases_size': 2.0,
      'pillars_size': 2.0,
      'gremlins_size': 2.0
  }
  world = World(
      np.random.RandomState(0), request.param, Robot('xmls/doggo.xml'), config)
  return world


def layout_sampling(world):
  mujoco_bridge = MujocoBridge('xmls/doggo.xml', world.sample_layout())
  fails = 0
  for _ in range(1000):
    try:
      mujoco_bridge.reset(world.sample_layout())
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
    MujocoBridge(Robot('xmls/doggo.xml'), bad_world.sample_layout())
  except ResamplingError:
    ok = True
  assert ok


def test_reset(good_world):
  mujoco_bridge = MujocoBridge(
      Robot('xmls/doggo.xml'), good_world.sample_layout())
  plt.figure()
  plt.imshow(mujoco_bridge.physics.render(camera_id='fixedfar'))
  mujoco_bridge.reset(good_world.sample_layout())
  plt.figure()
  plt.imshow(mujoco_bridge.physics.render(camera_id='fixedfar'))
  plt.pause(4.0)
  assert True
