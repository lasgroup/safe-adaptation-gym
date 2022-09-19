# Safe Adaptation Gym
The `safe-adaptation-gym` is a benchmark suite for testing how agents perform in a multitask and adaptation settings where safety is crucial.
It is based on [Safety Gym](https://github.com/openai/safety-gym) and extends its 3 tasks to 8 different task prototypes.

To find more details about evaluation metrics and formulation, please see our research paper.

## Install & Run
1. Open a new terminal and `git clone` the repo.
2. Create a new environment with your favorite environment manager (venv, conda). Make sure to have it set up with `python >= 3.8.13`.
3. Install dependencies with `cd safe-adaptation-gym && pip install .`
4. The following snippet demostrates how to use the `safe-adapatation-gym` no a specific task:
```python
import safe-adaptation-gym
from safe_adaptation_gym import tasks

robot = 'point'
seed = 666
task_name = 'go_to_goal'
config = {'obstacles_size_noise_scale': 1.})
rgb_observation = False  # use first-person-view observations, or only pseudo-lidar
render_options = {'camera_id': 'fixedfar', 'height': 320, 'width': 320}
render_lidar_and_collision = True  # render human-supportive visualization (slight computation slowdown)

env = safe_adaptation_gym.make(robot,
  task,
  seed,
  config,
  rbg_observation,
  render_options,
  render_lidar_and_collision
  )

o = env.reset()
*_ = env.step(env.action_space.sample()

env.set_task(tasks.HaulBox())

o = env.reset()
*_ = env.step(env.action_space.sample()
```

## Benchmark
