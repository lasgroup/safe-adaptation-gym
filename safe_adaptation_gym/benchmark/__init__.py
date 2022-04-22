import re
import inspect
from copy import deepcopy

from typing import Iterator

import numpy as np

from safe_adaptation_gym import tasks
from safe_adaptation_gym.safe_adaptation_gym import SafeAdaptationGym
from safe_adaptation_gym.benchmark import samplers

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


def make(benchmark_name: str,
         robot_name: str,
         seed: int = 666,
         rgb_observation: bool = False):
  rs = np.random.RandomState(seed)
  if benchmark_name == 'no_adaptation':
    train_sampler = samplers.OneRunTaskSampler(rs, _TASKS)
    test_sampler = samplers.TaskSampler(rs, {})
    return Benchmark(train_sampler, test_sampler, seed,
                     _ROBOTS_NAMES_TO_BASENAMES[robot_name], rgb_observation)


class Benchmark:

  def __init__(self, train_sampler: samplers.TaskSampler,
               test_sampler: samplers.TaskSampler, seed: int, robot_base: str,
               rgb_observation: bool):
    self._gym = SafeAdaptationGym(
        robot_base,
        rgb_observation=rgb_observation,
        render_lidars_and_collision=True)
    self._gym.seed(seed)
    self._seed = seed
    self._train_tasks_sampler = train_sampler
    self._test_tasks_sampler = test_sampler

  def train_tasks(self) -> Iterator[SafeAdaptationGym]:
    """
    Genereates the next task to train on. The user is in charge of (and has
    the flexibility to) calling this function after enough episodes per task.
    """
    while True:
      sample = self._train_tasks_sampler.sample()
      if sample is None:
        return None
      self._seed += 1
      self._gym.seed(self._seed)
      self._gym.set_task(sample[1])
      yield sample[0], deepcopy(self._gym)

  def test_tasks(self) -> Iterator[SafeAdaptationGym]:
    """
    Genereates the next task to test on. The user is in charge of (and has
    the flexibility to) calling this function after enough episodes per task.
    """
    while True:
      sample = self._test_tasks_sampler.sample()
      if sample is None:
        return None
      self._gym.set_task(sample[1])
      yield sample[0], deepcopy(self._gym)
