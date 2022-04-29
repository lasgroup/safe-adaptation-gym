from safe_adaptation_gym.benchmark import TASKS, ROBOTS_BASENAMES
from safe_adaptation_gym.safe_adaptation_gym import SafeAdaptationGym


def make(task_name: str,
         robot_name: str,
         seed: int = 666,
         rgb_observation: bool = False) -> SafeAdaptationGym:
  env = SafeAdaptationGym(
      ROBOTS_BASENAMES[robot_name.lower()],
      rgb_observation=rgb_observation,
      render_lidars_and_collision=True)
  env.seed(seed)
  task = TASKS[task_name.lower()]()
  env.set_task(task)
  return env
