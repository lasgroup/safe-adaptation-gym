from dataclasses import dataclass

from typing import Iterator, Mapping, Optional, Type, Tuple

import numpy as np

from safe_adaptation_gym.tasks.task import Task


@dataclass
class TaskSampler:
  rs: np.random.RandomState
  tasks: Mapping[str, Type[Task]]

  def sample(self) -> Optional[Tuple[str, Task]]:
    if len(self.tasks) == 0:
      return
    task_name, task = self.rs.permutation(list(self.tasks.items()))[0]
    return task_name, task()


class OneRunTaskSampler(TaskSampler):

  def __init__(self, rs, tasks):
    super(OneRunTaskSampler, self).__init__(rs, tasks)
    self._n_samples = len(tasks)

    def _sample() -> Iterator[Tuple[str, Task]]:
      all_tasks = list(self.tasks.items())
      self.rs.shuffle(all_tasks)
      for task_name, task in all_tasks:
        yield task_name, task()

    self._sample = _sample()

  def sample(self) -> Optional[Tuple[str, Task]]:
    try:
      return next(self._sample)
    except StopIteration:
      return
