import abc
from typing import Dict, Tuple, List, TypeVar

import numpy as np

from safe_adaptation_gym import consts

# Define as generic type instead of import the actual mujoco_bridge as it
# loads resources (e.g. GPU pointers) that should not exist on a parent process.
MujocoBridge = TypeVar("MujocoBridge")


class Task(abc.ABC):

  def __init__(self):
    self._obstacle_scales = None
    self._ctrl_scale = None
    self._bound = None

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

  def obstacle_scales(self, rs: np.random.RandomState):
    if self._obstacle_scales is None:
      self._obstacle_scales = rs.standard_cauchy(len(consts.OBSTACLES))
    return self._obstacle_scales

  def ctrl_scale(self, rs: np.random.RandomState, control_size: int):
    if self._ctrl_scale is None:
      # https://keisan.casio.com/exec/system/1180573169
      self._ctrl_scale = rs.standard_cauchy(control_size)
    return self._ctrl_scale

  def constraint_bound(self, rs: np.random.RandomState, max_bound: float):
    if self._bound is None:
      self._bound = rs.uniform(0., max_bound)
    return self._bound
