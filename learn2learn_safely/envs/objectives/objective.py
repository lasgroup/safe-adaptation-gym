import abc

from typing import Mapping

from learn2learn_safely.envs.robot import Robot
from learn2learn_safely.envs.world import World


class Objective(abc.ABC):

  def __init__(self):
    pass

  def setup_placements(self) -> Mapping[str, tuple]:
    raise NotImplementedError

  def build_world_config(self) -> dict:
    raise NotImplementedError

  def compute_reward(self, robot: Robot, world: World) -> float:
    raise NotImplementedError
