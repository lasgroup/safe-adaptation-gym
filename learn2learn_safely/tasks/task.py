import abc
from typing import Mapping, Tuple

import numpy as np

from learn2learn_safely.mujoco_bridge import MujocoBridge


class Task(abc.ABC):

  @abc.abstractmethod
  def setup_placements(self) -> Mapping[str, tuple]:
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
    Computes the task-specific reward.
    """

  @abc.abstractmethod
  def build(self, layout: dict, placements: dict, rs: np.random.RandomState,
            mujoco_bridge: MujocoBridge):
    """
    Not sure about this yet. But should allow an interface to build to task
    specific stuff
    """
