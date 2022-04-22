from dataclasses import dataclass

from typing import Iterator, Mapping, Optional, Type

import numpy as np

from safe_adaptation_gym.tasks.task import Task


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

    def _sample() -> Iterator[Task]:
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
