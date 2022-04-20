import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pytest

from safe_adaptation_gym.mujoco_bridge import MujocoBridge
from safe_adaptation_gym.tasks import GoToGoal
from safe_adaptation_gym.world import World
from safe_adaptation_gym.robot import Robot


@pytest.fixture
def world():
  _world = World(np.random.RandomState(0), GoToGoal(), Robot('xmls/doggo.xml'))
  return _world


def test_manually(world):
  mujoco_bridge = MujocoBridge(Robot('xmls/doggo.xml'), world.sample_layout())
  duration = 7
  framerate = 60
  frames = []
  while mujoco_bridge.physics.data.time < duration:
    mujoco_bridge.physics.step()
    if len(frames) < mujoco_bridge.physics.data.time * framerate:
      pixels = mujoco_bridge.physics.render(camera_id='fixedfar')
      frames.append(pixels)
  display_video(frames)
  assert True


def display_video(frames, framerate=30):
  height, width, _ = frames[0].shape
  dpi = 70
  fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
  ax.set_axis_off()
  ax.set_aspect('equal')
  ax.set_position([0, 0, 1, 1])
  im = ax.imshow(frames[0])

  def update(frame):
    im.set_data(frame)
    return [im]

  interval = 1000 / framerate
  anim = animation.FuncAnimation(
      fig=fig,
      func=update,
      frames=frames,
      interval=interval,
      blit=True,
      repeat=False)
  plt.show()
