from safe_adaptation_gym.benchmark import _TASKS, _ROBOTS_NAMES_TO_BASENAMES
from safe_adaptation_gym.safe_adaptation_gym import SafeAdaptationGym


def make(task_name: str,
         robot_name: str,
         seed: int = 666,
         rgb_observation: bool = False) -> SafeAdaptationGym:
  env = SafeAdaptationGym(
      _ROBOTS_NAMES_TO_BASENAMES[robot_name.lower()],
      rgb_observation=rgb_observation,
      render_lidars_and_collision=True)
  env.seed(seed)
  task = _TASKS(task_name.lower())()
  env.set_task(task)
  return env
