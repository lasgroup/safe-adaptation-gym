import re
import inspect
from dataclasses import dataclass

from typing import Iterator, Mapping, Optional, Type

import numpy as np

from safe_adaptation_gym import tasks
from safe_adaptation_gym.tasks.task import Task
from safe_adaptation_gym.safe_adaptation_gym import SafeAdaptationGym

BENCHMARKS = {'no_adaptation'}
ROBOTS = {'point', 'car', 'doggo'}

pattern = re.compile(r'(?<!^)(?=[A-Z])')

_TASKS = {
    pattern.sub('_', name).lower(): task
    for name, task in inspect.getmembers(tasks, inspect.isclass)
}

_BASE = 'xmls/'
_ROBOTS_NAMES_TO_BASENAMES = {
    name: _BASE + name + '.xml' for name in ['point', 'car', 'doggo']
}

_BENCHMARKS = {'no_adaptation': (_TASKS, {})}


def make(benchmark_name: str,
         robot_name: str,
         seed: int = 666,
         rgb_observation: bool = False):
  if benchmark_name == 'no_adaptation':
    rs = np.random.RandomState(seed)
    train_sampler = OneRunTaskSampler(rs, _TASKS)
    test_sampler = TaskSampler(rs, {})
    return Benchmark(train_sampler, test_sampler, seed,
                     _ROBOTS_NAMES_TO_BASENAMES[robot_name], rgb_observation)


@dataclass
class TaskSampler:
  rs: np.random.RandomState
  tasks: Mapping[str, Type[Task]]

  def sample(self) -> Optional[Task]:
    if len(self.tasks) == 0:
      return None
    return self.rs.choice(list(self.tasks.values()))()


class OneRunTaskSampler(TaskSampler):

  def __init__(self, rs, tasks):
    super(OneRunTaskSampler, self).__init__(rs, tasks)
    self._n_samples = len(tasks)

    def _sample():
      all_tasks = list(self.tasks.values())
      self.rs.shuffle(all_tasks)
      for task in all_tasks:
        yield task()

    self._sample = _sample()

  def sample(self) -> Optional[Task]:
    try:
      return next(self._sample)
    except StopIteration:
      return None


class Benchmark:

  def __init__(self, train_sampler: TaskSampler, test_sampler: TaskSampler,
               seed: int, robot_base: str, rgb_observation: bool):
    self._gym = SafeAdaptationGym(
        robot_base,
        rgb_observation=rgb_observation,
        render_lidars_and_collision=True)
    self._gym.seed(seed)
    self._train_tasks_sampler = train_sampler
    self._test_tasks_sampler = test_sampler

  def train_tasks(self) -> Iterator[SafeAdaptationGym]:
    """
    Genereates the next task to train on. The user is in charge of (and has
    the flexibility to) calling this function after enough episodes per task.
    """
    while True:
      next_task = self._train_tasks_sampler.sample()
      if next_task is None:
        return None
      self._gym.set_task(next_task)
      yield self._gym

  def test_tasks(self) -> Iterator[SafeAdaptationGym]:
    """
    Genereates the next task to test on. The user is in charge of (and has
    the flexibility to) calling this function after enough episodes per task.
    """
    while True:
      next_task = self._test_tasks_sampler.sample()
      if next_task is None:
        return None
      self._gym.set_task(next_task)
      yield self._gym
