import pytest
from dm_control import mujoco, viewer
from dm_env import StepType, TimeStep
from gym.wrappers import TimeLimit

from safe_adaptation_gym import tasks
from safe_adaptation_gym.safe_adaptation_gym import SafeAdaptationGym


class ViewerWrapper:
    def __init__(self, env: SafeAdaptationGym):
        self.env = env
        self._action_spec = mujoco.action_spec(env.mujoco_bridge.physics)
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
        obs, reward, terminal, _ = self.env.step(action)
        t = StepType.LAST if terminal else StepType.MID
        self.sum_rewards += reward
        return TimeStep(t, reward, 1.0, obs)


@pytest.fixture(
    params=[
        tasks.Fetch,
    ]
)
def safety_gym(request):
    arena = TimeLimit(
        SafeAdaptationGym(
            render_lidars_and_collision=False,
            render_options={"camera_id": "fixedfar", "height": 320, "width": 320},
            config={"obstacles_size_noise_scale": 1.0, "max_joints_to_disable": 0},
        ),
        1000,
    )
    arena.set_task(request.param)
    return arena


@pytest.mark.interactive
def test_viewer(safety_gym):
    policy = lambda *_: safety_gym.action_space.sample()
    viewer.launch(ViewerWrapper(safety_gym), policy)


def test_print(safety_gym):
    import re

    import matplotlib.pyplot as plt

    safety_gym.reset()
    name = type(safety_gym.unwrapped._world.task).__name__
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    plt.imsave(name + ".png", safety_gym.render())
    assert True


def test_episode(safety_gym):
    import matplotlib.pyplot as plt

    policy = lambda *_: safety_gym.action_space.sample()
    safety_gym.reset()
    done = False
    while not done:
        *_, done, _ = safety_gym.step(policy())
        plt.imshow(safety_gym.render())
        plt.pause(0.00001)
    assert True
