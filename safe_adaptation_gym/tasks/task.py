import abc
from typing import Dict, Tuple, List, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation

from safe_adaptation_gym import consts

# Define as generic type instead of import the actual mujoco_bridge as it
# loads resources (e.g. GPU pointers) that should not exist on a parent process.
MujocoBridge = TypeVar("MujocoBridge")

_DEFAULT_DAMPING = 0.01


class Task(abc.ABC):

  def __init__(self, train: bool):
    self._obstacle_scales = None
    self._damping = None
    self._bound = None
    self._gravity = None
    self.train = train

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
  def compute_reward(
      self,
      layout: dict,
      placements: dict,
      rs: np.random.RandomState,
      mujoco_bridge: MujocoBridge,
    ) -> Tuple[float, bool, dict]:
      """
      Computes the task-specific reward. Exposes the layout and placements for
      goal/object position resampling
      """

  def compute_cost(self, mujoco_bridge: MujocoBridge) -> float:
      """
      Computes the task specific costs.
      """
      return 0.0

  @abc.abstractmethod
  def set_mocaps(self, mujoco_bridge: MujocoBridge):
      """
      Sets up the task specific mocap objects.
      """

  @abc.abstractmethod
  def reset(
      self,
      layout: dict,
      placements: dict,
      rs: np.random.RandomState,
      mujoco_bridge: MujocoBridge,
    ):
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

  def num_obstacles(self, rs: np.random.RandomState) -> int:
    if self.train:
      return rs.randint(10, 15)
    else:
      return 20

  @property
  def placement_extents(self) -> Tuple[float, float, float, float]:
    return consts.PLACEMENT_EXTENTS

  @property
  def arena_radius(self):
    return (self.placement_extents[2]) * np.sqrt(2.0)

  def obstacle_scales(self, rs: np.random.RandomState):
    if self._obstacle_scales is None:
      # https://keisan.casio.com/exec/system/1180573169
        self._obstacle_scales = rs.standard_cauchy(len(consts.OBSTACLES))
    return self._obstacle_scales

  def gravity(self, rs: np.random.RandomState, max_angle: float) -> np.ndarray:
    if self._gravity is None:
        x, y = rs.uniform(0.0, max_angle, 2)
        self._gravity = Rotation.from_euler("xy", [x, y], degrees=True)
    return self._gravity.apply([0., 0., -9.81])

  def constraint_bound(self, rs: np.random.RandomState, max_bound: float):
    if self._bound is None:
        self._bound = rs.uniform(0.0, max_bound)
    return self._bound

  def modify_tree(self, rs: np.random.RandomState, min_damping: float):
    if self._damping is None:
        self._damping = rs.uniform(
            min(_DEFAULT_DAMPING, min_damping), _DEFAULT_DAMPING, 2
            )
    return [
            (("joint", "y"), ("damping", self._damping[0])),
            (("joint", "x"), ("damping", self._damping[1])),
        ]
