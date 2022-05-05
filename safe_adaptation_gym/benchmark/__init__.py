import re
import inspect
from copy import deepcopy

from typing import Iterator, Callable, Optional

import numpy as np

from gym import Env

from safe_adaptation_gym import tasks
from safe_adaptation_gym.safe_adaptation_gym import SafeAdaptationGym
from safe_adaptation_gym.benchmark import samplers

BENCHMARKS = {'no_adaptation'}
ROBOTS = {'point', 'car', 'doggo'}

pattern = re.compile(r'(?<!^)(?=[A-Z])')

TASKS = {
    pattern.sub('_', name).lower(): task
    for name, task in inspect.getmembers(tasks, inspect.isclass)
}

_BASE = 'xmls/'
ROBOTS_BASENAMES = {
    name: _BASE + name + '.xml' for name in ['point', 'car', 'doggo']
}


class Benchmark:

  def __init__(self,
               train_sampler: samplers.TaskSampler,
               test_sampler: samplers.TaskSampler,
               seed: int,
               robot_base: str,
               rgb_observation: bool,
               wrappers: Optional[Callable[[SafeAdaptationGym], Env]] = None):
    self._gym = SafeAdaptationGym(
        robot_base,
        rgb_observation=rgb_observation,
        render_lidars_and_collision=True)
    if wrappers:
      self._gym = wrappers(self._gym)
    self._gym.seed(seed)
    self._seed = seed
    self._train_tasks_sampler = train_sampler
    self._test_tasks_sampler = test_sampler

  @property
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
      self._gym.unwrapped.set_task(sample[1])  # noqa
      yield sample[0], deepcopy(self._gym)

  @property
  def test_tasks(self) -> Iterator[SafeAdaptationGym]:
    """
    Genereates the next task to test on. The user is in charge of (and has
    the flexibility to) calling this function after enough episodes per task.
    """
    while True:
      sample = self._test_tasks_sampler.sample()
      if sample is None:
        return None
      self._gym.unwrapped.set_task(sample[1])  # noqa
      yield sample[0], deepcopy(self._gym)


def make(
    benchmark_name: str,
    robot_name: str,
    seed: int = 666,
    rgb_observation: bool = False,
    wrappers: Optional[Callable[[SafeAdaptationGym], Env]] = None) -> Benchmark:
  """
  Creates a new benchmark based on name. Can wrap with gym.Wrappers via the
  wrappers function (which takes a SafeAdaptationGym instance and returns a
  wrapped Gym instance).
  """
  assert benchmark_name in BENCHMARKS, 'Supplied a wrong benchmark name.'
  rs = np.random.RandomState(seed)
  if benchmark_name == 'no_adaptation':
    train_sampler = samplers.OneRunTaskSampler(rs, TASKS)
    test_sampler = samplers.TaskSampler(rs, {})
    return Benchmark(train_sampler, test_sampler, seed,
                     ROBOTS_BASENAMES[robot_name], rgb_observation, wrappers)
