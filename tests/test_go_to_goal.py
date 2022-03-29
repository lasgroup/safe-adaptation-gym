import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pytest

from learn2learn_safely.mujoco_bridge import MujocoBridge
from learn2learn_safely.tasks import GoToGoal
from learn2learn_safely.world import World


@pytest.fixture
def world():
  _world = World(np.random.RandomState(0), GoToGoal(), 'xmls/doggo.xml')
  return _world


def test_manually(world):
  mujoco_bridge = MujocoBridge(world.sample_layout())
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
