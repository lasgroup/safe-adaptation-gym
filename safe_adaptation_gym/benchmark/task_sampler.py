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
