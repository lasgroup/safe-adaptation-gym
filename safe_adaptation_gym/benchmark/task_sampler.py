from dataclasses import dataclass

from typing import Mapping, Optional, Tuple

import numpy as np

from safe_adaptation_gym.tasks import Task


@dataclass
class TaskSampler:
    rs: np.random.RandomState
    tasks: Mapping[str, Task]

    def sample(self) -> Optional[Tuple[str, Task]]:
        if len(self.tasks) == 0:
            return
        task_name = self.rs.permutation(list(self.tasks.keys()))[0]
        task = self.tasks[task_name]
        return task_name, task
