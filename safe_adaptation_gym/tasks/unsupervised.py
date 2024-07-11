from typing import Dict, Tuple

import numpy as np

from safe_adaptation_gym import utils
import safe_adaptation_gym.primitive_objects as po
from safe_adaptation_gym.tasks.go_to_goal import _GOAL_PLACEMENT, GoToGoal
from safe_adaptation_gym.tasks.task import MujocoBridge


class Unsupervised(GoToGoal):
    def __init__(self):
        super(Unsupervised, self).__init__()

    def setup_placements(self) -> Dict[str, tuple]:
        placements = super(Unsupervised, self).setup_placements()
        placements.update({"secondary_goal": (_GOAL_PLACEMENT, self.GOAL_KEEPOUT)})
        return placements

    def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
        goal_world_config = super(Unsupervised, self).build_world_config(layout, rs)
        secondary_goal = {
            "bodies": {
                "secondary_goal": po.get_goal(
                    "secondary_goal",
                    self.GOAL_SIZE,
                    layout["goal"],
                    utils.random_rot(rs),
                    color=np.array([1, 0, 0, 0.25]),
                )
            }
        }
        utils.merge(goal_world_config, secondary_goal)
        return goal_world_config

    def compute_reward(
        self,
        layout: dict,
        placements: dict,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ) -> Tuple[float, bool, dict]:
        goal_pos = np.asarray(mujoco_bridge.body_pos("secondary_goal"))
        robot_pos = mujoco_bridge.body_pos("robot")
        distance = np.linalg.norm(robot_pos - goal_pos)
        reward = self._last_goal_distance - distance
        self._last_goal_distance = distance
        info = {}
        if distance <= self.GOAL_SIZE:
            info["goal_met"] = True
            utils.update_layout(layout, mujoco_bridge)
            self.reset(layout, placements, rs, mujoco_bridge)
            reward += 1.0
        return reward, False, info

    def _resample_goal_position(self, layout: dict, placements: dict,
                                rs: np.random.RandomState):
        layout.pop('secondary_goal')
        goal_placement = _GOAL_PLACEMENT
        for i in range(50):
            for _ in range(10000):
                goal_xy = utils.draw_placement(rs, goal_placement, self.placement_extents,
                                            self.GOAL_KEEPOUT)
                valid_placement = True
                for other_name, other_xy in layout.items():
                    other_keepout = placements[other_name][1]
                    dist = np.linalg.norm(goal_xy - other_xy)
                    if dist < other_keepout + self.GOAL_KEEPOUT:
                        valid_placement = False
                        break
                if valid_placement:
                    return goal_xy
                if i == 48:
                    goal_placement = None
                else:
                    goal_placement = [utils.increase_extents(goal_placement[0])]
        raise utils.ResamplingError('Failed to generate goal')


    def reset(
        self,
        layout: dict,
        placements: dict,
        rs: np.random.RandomState,
        mujoco_bridge: MujocoBridge,
    ):
        goal_xy = self._resample_goal_position(layout, placements, rs)
        layout["secondary_goal"] = goal_xy
        robot_pos = mujoco_bridge.body_pos("robot")[:2]
        self._last_goal_distance = np.linalg.norm(robot_pos - goal_xy)
        mujoco_bridge.set_body_pos("secondary_goal", goal_xy)
        mujoco_bridge.physics.forward()

    @property
    def obstacles(self) -> int:
        return [5, 6, 0, 1]
