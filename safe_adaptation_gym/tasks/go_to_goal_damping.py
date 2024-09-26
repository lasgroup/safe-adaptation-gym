import numpy as np

from safe_adaptation_gym.tasks.go_to_goal import GoToGoal

_DEFAULT_DAMPING = 0.01


class GoToGoalDamping(GoToGoal):
    def __init__(self):
        super(GoToGoalDamping, self).__init__()

    def modify_tree(self, rs: np.random.RandomState):
        damping = np.ones((2,)) * _DEFAULT_DAMPING * 0.1
        return [
            (("joint", "y"), ("damping", damping[0])),
            (("joint", "x"), ("damping", damping[1])),
        ]
