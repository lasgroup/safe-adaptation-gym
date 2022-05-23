import re
import inspect

from typing import Iterator, Tuple

import numpy as np

from safe_adaptation_gym import tasks
from safe_adaptation_gym.benchmark import samplers

BENCHMARKS = {'no_adaptation', 'domain_randomization'}
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

  def __init__(self, train_sampler: samplers.TaskSampler,
               test_sampler: samplers.TaskSampler, batch_size: int):
    self._train_tasks_sampler = train_sampler
    self._test_tasks_sampler = test_sampler
    self._batch_size = batch_size

  @property
  def train_tasks(self) -> Iterator[Tuple[str, tasks.Task]]:
    """
    Genereates the next task to train on. The user is in charge of (and has
    the flexibility to) calling this function after enough episodes per task.
    """
    for _ in range(self._batch_size):
      sample = self._train_tasks_sampler.sample()
      if sample is None:
        return
      yield sample

  @property
  def test_tasks(self) -> Iterator[Tuple[str, tasks.Task]]:
    """
    Genereates the next task to test on. The user is in charge of (and has
    the flexibility to) calling this function after enough episodes per task.
    """
    for _ in range(self._batch_size):
      sample = self._test_tasks_sampler.sample()
      if sample is None:
        return
      yield sample


def make(benchmark_name: str,
         batch_size: int = 16,
         seed: int = 666) -> Benchmark:
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
    return Benchmark(train_sampler, test_sampler, len(TASKS))
  if benchmark_name == 'domain_randomization':
    # Observe all tasks, radomize the parameters of the MDP (action scale,
    # size of obstacles, etc.).
    train_sampler = samplers.TaskSampler(rs, TASKS)
    test_sampler = samplers.TaskSampler(rs, TASKS)
    return Benchmark(train_sampler, test_sampler, batch_size)
