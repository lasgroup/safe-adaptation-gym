# Safe Adaptation Gym
The `safe-adaptation-gym` is a benchmark suite for testing how agents perform in a multitask and adaptation settings where safety is crucial.
It is based on [Safety Gym](https://github.com/openai/safety-gym) and extends its 3 tasks to 8 different task prototypes.

To find more details about evaluation metrics and formulation, please see our research paper.

## Install & Run
1. Open a new terminal and `git clone` the repo.
2. Create a new environment with your favorite environment manager (venv, conda). Make sure to have it set up with `python >= 3.8.13`.
3. Install dependencies with `cd safe-adaptation-gym && pip install .`
4. The following snippet demostrates how to use the `safe-adapatation-gym` on a specific task:
```python
import safe-adaptation-gym
from safe_adaptation_gym import tasks
# Define important parameters.
robot = 'point'
seed = 666
task_name = 'go_to_goal'
config = {'obstacles_size_noise_scale': 1.})
rgb_observation = False  # use first-person-view observations, or only pseudo-lidar
render_options = {'camera_id': 'fixedfar', 'height': 320, 'width': 320}
render_lidar_and_collision = True  # render human-supportive visualization (slight computation slowdown)
# Make a new simulation environment.
env = safe_adaptation_gym.make(robot,
  task,
  seed,
  config,
  rbg_observation,
  render_options,
  render_lidar_and_collision
  )
policy = lambda obs: env.action_space.sample()  # define a uniformly random policy
# One RL interaction.
observation = env.reset()
action = policy(observation)
next_observation, reward, done, info  = env.step(action)
cost = info.get('cost', 0.)
# Set a new task.
env.set_task(tasks.HaulBox())
# One RL interaction.
observation = env.reset()
action = policy(observation)
next_observation, reward, done, info = env.step(action)
```

## Benchmark
In order to reproduce our results, define a task sampler:
```python
import safety_gym
from safe_adaptation_gym import benchmark
# Define important parameters.
robot = 'doggo'
seed = 666
env = safe_adaptation_gym.make(robot, seed)
policy = lambda obs: env.action_space.sample()  # define a uniformly random policy
benchmark_name = 'multitask'  # can also be 'task_adaptation', in which case, some tasks prototypes are held out.
batch_size = 30
task_sampler = benchmark.make(benchmark_name, batch_size, seed)
# Iterate over training task prototypes
for task_name, task in task_sampler.train_tasks:
  observation = env.reset(options={'task': task})  # We can also set new tasks as we reset to a new episode
  action = policy(observation)
  next_observation, reward, done, info  = env.step(action)
  cost = info.get('cost', 0.)
# Iterate over held-out task prototypes
for task_name, task in task_sampler.test_tasks:
  env.reset(options={'task': task})
  action = policy(observation)
  next_observation, reward, done, info  = env.step(action)
  cost = info.get('cost', 0.)
```

