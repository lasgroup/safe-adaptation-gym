from __future__ import annotations
from dataclasses import dataclass

from typing import TYPE_CHECKING, Callable, Mapping, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from safe_adaptation_gym.tasks import Task
    TaskFactory = Callable[[np.random.RandomState, float], Task]


@dataclass
class TaskSampler:
    rs: np.random.RandomState
    tasks: Mapping[str, TaskFactory]

    def sample(self) -> Optional[Tuple[str, TaskFactory]]:
        if len(self.tasks) == 0:
            return
        task_name = self.rs.permutation(list(self.tasks.keys()))[0]
        task = self.tasks[task_name]
        return task_name, task
