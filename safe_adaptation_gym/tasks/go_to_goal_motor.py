import numpy as np

from safe_adaptation_gym.tasks.go_to_goal import GoToGoal

_DEFAULT_GEAR = 0.3


class GoToGoalMotor(GoToGoal):
    def __init__(self):
        super(GoToGoalMotor, self).__init__()

    def modify_tree(self, rs: np.random.RandomState):
        motor = np.ones((2,)) * _DEFAULT_GEAR * 10.
        return [
            (("actuator", "x"), ("gear", f"{motor[0]} 0 0 0 0 0")),
        ]
