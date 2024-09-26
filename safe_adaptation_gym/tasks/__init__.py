from safe_adaptation_gym.tasks.dribble_ball import DribbleBall
from safe_adaptation_gym.tasks.collect import Collect
from safe_adaptation_gym.tasks.go_to_goal import GoToGoal
from safe_adaptation_gym.tasks.haul_box import HaulBox
from safe_adaptation_gym.tasks.press_buttons import PressButtons
from safe_adaptation_gym.tasks.push_box import PushBox
from safe_adaptation_gym.tasks.roll_rod import RollRod
from safe_adaptation_gym.tasks.catch_goal import CatchGoal
from safe_adaptation_gym.tasks.unsupervised import Unsupervised
from safe_adaptation_gym.tasks.go_to_goal_scarce import GoToGoalScarce
from safe_adaptation_gym.tasks.press_buttons_scarce import PressButtonsScarce
from safe_adaptation_gym.tasks.push_box_scarce import PushBoxScarce
from safe_adaptation_gym.tasks.go_to_goal_damping import GoToGoalDamping
from safe_adaptation_gym.tasks.go_to_goal_motor import GoToGoalMotor
from safe_adaptation_gym.tasks.task import Task

__all__ = [
    "GoToGoal",
    "PushBox",
    "PressButtons",
    "RollRod",
    "DribbleBall",
    "Collect",
    "HaulBox",
    "CatchGoal",
    "Unsupervised",
    "GoToGoalScarce",
    "PressButtonsScarce",
    "PushBoxScarce",
    "GoToGoalDamping",
    "GoToGoalMotor",
]
