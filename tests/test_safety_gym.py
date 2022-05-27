import numpy as np
import pytest
import glfw
from dm_control import viewer
from dm_env import specs, TimeStep, StepType
from gym.wrappers import TimeLimit

from safe_adaptation_gym import tasks
from safe_adaptation_gym.safe_adaptation_gym import SafeAdaptationGym


def controller(*args, **kwargs):
  action = np.zeros((2,))
  if not glfw.joystick_present(glfw.JOYSTICK_1):
    return action
  cmd = glfw.get_joystick_axes(glfw.JOYSTICK_1)[0]
  y = -cmd[1] if np.abs(cmd[1]) > 0.05 else 0.0
  x = -cmd[0] if np.abs(cmd[0]) > 0.05 else 0.0
  action[0] = np.clip(y, -1.0, 1.0)
  action[1] = x
  return action


class ViewerWrapper:

  def __init__(self, env):
    self.env = env
    self._action_spec = specs.BoundedArray(
        shape=env.action_space.shape,
        dtype=np.float32,
        minimum=env.action_space.low,
        maximum=env.action_space.high)
    self.sum_rewards = None

  @property
  def physics(self):
    return self.env.mujoco_bridge.physics

  def reset(self):
    if self.sum_rewards is not None:
      print('Sum rewards: {}'.format(self.sum_rewards))
    self.env.reset()
    self.sum_rewards = 0.

  def action_spec(self):
    return self._action_spec

  def step(self, action):
    obs, reward, terminal, info = self.env.step(action)
    t = StepType.LAST if terminal else StepType.MID
    self.sum_rewards += reward
    return TimeStep(t, reward, 1.0, obs)


@pytest.fixture(params=[
    tasks.PushBox(),
    tasks.PushRodMass(),
    tasks.BallToGoal(),
    tasks.PressButtons(),
    tasks.GoToGoal()
])
def safety_gym(request):
  arena = TimeLimit(
      SafeAdaptationGym('xmls/point.xml', render_lidars_and_collision=True),
      1000)
  arena.seed(10)
  arena.set_task(request.param)
  arena.seed(10)
  return arena


@pytest.mark.interactive
def test_viewer(safety_gym):
  viewer.launch(ViewerWrapper(safety_gym), controller)


def test_episode(safety_gym):
  safety_gym.reset()
  done = False
  while not done:
    *_, done, _ = safety_gym.step(safety_gym.action_space.sample())
  assert True
