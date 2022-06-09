import abc
from typing import Dict, Tuple, List

import numpy as np

from safe_adaptation_gym.mujoco_bridge import MujocoBridge
from safe_adaptation_gym import consts


class Task(abc.ABC):

  @abc.abstractmethod
  def setup_placements(self) -> Dict[str, tuple]:
    """
    Setups the task specific placements (e.g., goals or other obstacles).
    """

  @abc.abstractmethod
  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    """
    Defines the placements physical and visual attributes to be loaded by
    mujoco.
    """

  @abc.abstractmethod
  def compute_reward(self, layout: dict, placements: dict,
                     rs: np.random.RandomState,
                     mujoco_bridge: MujocoBridge) -> Tuple[float, bool, dict]:
    """
    Computes the task-specific reward. Exposes the layout and placements for
    goal/object position resampling
    """

  def compute_cost(self, mujoco_bridge: MujocoBridge) -> float:
    """
    Computes the task specific costs.
    """
    return 0.

  @abc.abstractmethod
  def set_mocaps(self, mujoco_bridge: MujocoBridge):
    """
    Sets up the task specific mocap objects.
    """

  @abc.abstractmethod
  def reset(self, layout: dict, placements: dict, rs: np.random.RandomState,
            mujoco_bridge: MujocoBridge):
    """
    Not sure about this yet. But should allow an interface to build to task
    specific stuff
    """

  @property
  def obstacles_distribution(self) -> List[float]:
    """
    How many obstacle instances are sampled for each obstacle type.
    Type order follows: 'hazards', 'vases', 'gremlins', 'pillars',
    as in consts.OBSTACLES
    """
    return [1 / len(consts.OBSTACLES)] * len(consts.OBSTACLES)

  @property
  def num_obstacles(self) -> int:
    return 20

  @property
  def placement_extents(self) -> Tuple[float, float, float, float]:
    return consts.PLACEMENT_EXTENTS

  @property
  def arena_radius(self):
    return (self.placement_extents[2]) * np.sqrt(2.)
