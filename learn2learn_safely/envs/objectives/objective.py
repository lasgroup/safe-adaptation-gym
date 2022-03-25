import abc

from typing import Mapping

import numpy as np

from learn2learn_safely.envs.robot import Robot
from learn2learn_safely.envs.world import World


class Objective(abc.ABC):

  def setup_placements(self) -> Mapping[str, tuple]:
    raise NotImplementedError

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    raise NotImplementedError

  def compute_reward(self, robot: Robot, world: World) -> float:
    raise NotImplementedError
