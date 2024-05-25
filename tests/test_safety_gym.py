import numpy as np
import pytest
import glfw
from dm_control import viewer
from dm_env import specs, TimeStep, StepType
from gym.wrappers import TimeLimit

from safe_adaptation_gym import tasks
from safe_adaptation_gym.safe_adaptation_gym import SafeAdaptationGym

ROBOT = "doggo"


def controller(action_space):
    if ROBOT == "car":
        return car_controller
    elif ROBOT == "point":
        return point_controller
    else:
        return lambda *_: action_space.sample()


def point_controller(*args, **kwargs):
    action = np.zeros((2,))
    if not glfw.joystick_present(glfw.JOYSTICK_1):
        return action
    cmd = glfw.get_joystick_axes(glfw.JOYSTICK_1)[0]
    y = -cmd[1] if np.abs(cmd[1]) > 0.05 else 0.0
    x = -cmd[0] if np.abs(cmd[0]) > 0.05 else 0.0
    action[0] = np.clip(y, -1.0, 1.0)
    action[1] = x
    return action


def car_controller(*args, **kwargs):
    action = np.zeros((2,))
    if not glfw.joystick_present(glfw.JOYSTICK_1):
        return action
    cmd = glfw.get_joystick_axes(glfw.JOYSTICK_1)[0]
    x = -cmd[1] if np.abs(cmd[1]) > 0.05 else 0.0
    y = -cmd[0] if np.abs(cmd[0]) > 0.05 else 0.0
    action[0] = (x + y) / 2.0
    action[1] = (x - y) / 2.0
    return action


class ViewerWrapper:
    def __init__(self, env):
        self.env = env
        self._action_spec = specs.BoundedArray(
            shape=env.action_space.shape,
            dtype=np.float32,
            minimum=env.action_space.low,
            maximum=env.action_space.high,
        )
        self.sum_rewards = None

    @property
    def physics(self):
        return self.env.mujoco_bridge.physics

    def reset(self):
        if self.sum_rewards is not None:
            print("Episode Return: {}".format(self.sum_rewards))
        self.env.reset()
        self.sum_rewards = 0.0

    def action_spec(self):
        return self._action_spec

    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)
        t = StepType.LAST if terminal else StepType.MID
        self.sum_rewards += reward
        return TimeStep(t, reward, 1.0, obs)


@pytest.fixture(
    params=[
        tasks.CatchGoal(),
        tasks.HaulBox(),
        tasks.Collect(),
        tasks.PushBox(),
        tasks.RollRod(),
        tasks.DribbleBall(),
        tasks.PressButtons(),
        tasks.GoToGoal(),
        tasks.Unsupervised(),
    ]
)
def safety_gym(request):
    arena = TimeLimit(
        SafeAdaptationGym(
            f"xmls/{ROBOT}.xml",
            render_lidars_and_collision=False,
            render_options={"camera_id": "fixedfar", "height": 320, "width": 320},
            config={"obstacles_size_noise_scale": 1.0},
        ),
        1000,
    )
    seeds = {
        "Collect",
        "GoToGoal",
        "HaulBox",
        "PushBox",
        "DribbleBall",
        "PressButtons",
        "CatchGoal",
        "RollRod",
        "Unsupervised",
    }
    seeds = {name: num for name, num in zip(seeds, range(len(seeds)))}
    name = type(request.param).__name__
    arena.seed(seeds[name])
    arena.set_task(request.param)
    arena.seed(seeds[name])
    return arena


@pytest.mark.interactive
def test_viewer(safety_gym):
    policy = (
        (lambda *_: safety_gym.action_space.sample())
        if not glfw.joystick_present(glfw.JOYSTICK_1)
        else controller(safety_gym.action_space)
    )
    viewer.launch(ViewerWrapper(safety_gym), policy)


def test_print(safety_gym):
    import matplotlib.pyplot as plt
    import re

    safety_gym.reset()
    name = type(safety_gym.unwrapped._world.task).__name__
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    plt.imsave(name + ".png", safety_gym.render())
    assert True


def test_episode(safety_gym):
    import matplotlib.pyplot as plt

    policy = (
        (lambda *_: safety_gym.action_space.sample())
        if not glfw.joystick_present(glfw.JOYSTICK_1)
        else controller(safety_gym.action_space)
    )
    safety_gym.reset()
    done = False
    while not done:
        *_, done, _ = safety_gym.step(policy())
        plt.imshow(safety_gym.render())
        plt.pause(0.00001)
    assert True
