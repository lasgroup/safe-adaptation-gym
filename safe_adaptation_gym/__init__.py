from typing import Optional, Dict

from safe_adaptation_gym.benchmark import TASKS, ROBOTS_BASENAMES
from safe_adaptation_gym.safe_adaptation_gym import SafeAdaptationGym


def make(task_name: str,
         robot_name: str,
         seed: int = 666,
         rgb_observation: bool = False,
         render_options: Optional[Dict] = None) -> SafeAdaptationGym:
  env = SafeAdaptationGym(
      ROBOTS_BASENAMES[robot_name.lower()],
      rgb_observation=rgb_observation,
      render_lidars_and_collision=True,
      render_options=render_options)
  env.seed(seed)
  task = TASKS[task_name.lower()]()
  env.set_task(task)
  return env
