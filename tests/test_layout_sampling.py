import numpy as np
import pytest


from safe_adaptation_gym import tasks
from safe_adaptation_gym.mujoco_bridge import MujocoBridge
from safe_adaptation_gym.utils import ResamplingError
from safe_adaptation_gym.world import World
from safe_adaptation_gym.robot import Robot


@pytest.fixture(params=[
        tasks.CatchGoal(),
        tasks.HaulBox(),
        tasks.Collect(),
        tasks.PushBox(),
        tasks.RollRod(),
        tasks.DribbleBall(),
        tasks.PressButtons(),
        tasks.GoToGoal(),
        tasks.Unsupervised(),
    ], ids=lambda x: x.__class__.__name__)
def good_world(request):
  world = World(
      np.random.RandomState(0), request.param, Robot('xmls/doggo.xml'))
  return world


@pytest.fixture(params=[
        tasks.CatchGoal(),
        tasks.HaulBox(),
        tasks.Collect(),
        tasks.PushBox(),
        tasks.RollRod(),
        tasks.DribbleBall(),
        tasks.PressButtons(),
        tasks.GoToGoal(),
        tasks.Unsupervised(),
    ], ids=lambda x: x.__class__.__name__)
def bad_world(request):
  config = {
      'hazards_size': 2.0,
      'vases_size': 2.0,
      'pillars_size': 2.0,
      'gremlins_size': 2.0
  }
  world = World(
      np.random.RandomState(0), request.param, Robot('xmls/doggo.xml'),
      config)
  return world


def layout_sampling(world):
  mujoco_bridge = MujocoBridge(Robot('xmls/doggo.xml'))
  fails = 0
  for _ in range(200):
    try:
      mujoco_bridge.rebuild(world.sample_layout())
    except ResamplingError:
      fails += 1
  # Make sure that the fail rate is smaller than 0.5%
  return fails


def test_layout_sampling(good_world):
  fails = layout_sampling(good_world)
  # Make sure that the fail rate is smaller than 0.5%
  assert fails <= 1


def test_layout_impossible(bad_world):
  ok = False
  try:
    MujocoBridge(Robot('xmls/doggo.xml'))
    bad_world.sample_layout()
  except ResamplingError:
    ok = True
  assert ok
