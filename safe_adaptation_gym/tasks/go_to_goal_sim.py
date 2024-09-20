
from matplotlib.pylab import RandomState
import numpy as np

from safe_adaptation_gym.tasks.go_to_goal import GoToGoal


class GoToGoalSim(GoToGoal):
    def __init__(self):
        super(GoToGoalSim, self).__init__()

    def modify_tree(self, rs: RandomState, min_damping: float):
        self._damping = np.ones((2,)) * min_damping * 0.1
        return [
            (("joint", "y"), ("damping", self._damping[0])),
            (("joint", "x"), ("damping", self._damping[1])),
        ]
