import abc

from typing import Mapping, Tuple

import numpy as np

from learn2learn_safely.world import World


class Objective(abc.ABC):

  def setup_placements(self) -> Mapping[str, tuple]:
    raise NotImplementedError

  def build_world_config(self, layout: dict, rs: np.random.RandomState) -> dict:
    raise NotImplementedError

  def compute_reward(self, layout: dict, placements: dict,
                     rs: np.random.RandomState,
                     world: World) -> Tuple[float, dict]:
    raise NotImplementedError

  def build(self, layout: dict, placements: dict, rs: np.random.RandomState,
            world: World):
    raise NotImplementedError
