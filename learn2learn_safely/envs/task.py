from copy import deepcopy
from types import SimpleNamespace
from typing import List, Mapping

import numpy as np

import consts as c
import primitive_objects as po
import utils
from objectives.objective import Objective
from robot import Robot
from world import World


class Task:
  DEFAULT = {
      'placements_margin': 0.0,
      'robot_keepout': 0.4,
      'num_obstacles': 10,
      'hazards_size': 0.3,
      'vases_size': 0.1,
      'pillars_size': 0.2,
      'gremlins_size': 0.1,
      'obstacles_size_noise': 0.05
  }

  def __init__(self, seed: int, objective: Objective, config=None):
    if config is None:
      config = {}
    tmp_config = deepcopy(self.DEFAULT)
    tmp_config.update(config)
    self.config = SimpleNamespace(**tmp_config)
    self.objective = objective
    self._seed = seed
    self.rs = np.random.RandomState(seed)
    self._placements = self.setup_placements()
    self._obstacle_sizes = self.rs.normal([
        self.config.hazards_size, self.config.vases_size,
        self.config.pillars_size, self.config.gremlins_size
    ], self.config.obstacles_size_noise * np.ones(len(c.OBSTACLES)))

  def setup_placements(self):
    """ Build a dict of placements. """
    # TODO (yarden): Should the probability of different obstacle types be
    #  different?
    obstacle_samples = self.rs.multinomial(
        self.config.num_obstacles, [1 / len(c.OBSTACLES)] * len(c.OBSTACLES))
    placements = {
        **self.placement_dict_from_object('robot', 1),
    }
    for obstacle, num in zip(c.OBSTACLES, obstacle_samples):
      placements.update(self.placement_dict_from_object(obstacle, num))
    placements.update(self.objective.setup_placements())
    return placements

  def placement_dict_from_object(self, object_name: str,
                                 num_placements) -> Mapping[str, tuple]:
    """ Get the placements' dict subset just for a given object name """
    placements_dict = {}
    name = object_name + 's' if num_placements > 1 else object_name
    object_fmt = object_name + '{i}' if num_placements > 1 else object_name
    object_keepout = getattr(self.config, name + '_keepout')
    for i in range(num_placements):
      # Set placements to None so that obstacles positions are sampled randomly.
      placements = None
      placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
    return placements_dict

  def build_world_config(self, layout: dict, robot_base_path: str):
    """ Create a world_config from our own config """
    world_config = {
        'robot_path': robot_base_path,
        'robot_xy': layout['robot'],
        'robot_rot': utils.random_rot(self.rs)
    }
    for name, xy in layout.items():
      if 'vase' in name:
        _object = po.get_vase(name, self._obstacle_sizes[0], xy,
                              utils.random_rot(self.rs))
        world_config['objects'][name] = _object
      elif 'gremlin' in name:
        rot = utils.random_rot(self.rs)
        _object = po.get_gremlin(name + 'obj', self._obstacle_sizes[1], xy, rot)
        mocap = po.get_gremlin(name + 'mocap', self._obstacle_sizes[1], xy, rot)
        world_config['objects'][name + 'obj'] = _object
        world_config['mocaps'][name + 'mocap'] = mocap
      elif 'hazard' in name:
        _object = po.get_hazard(name, self._obstacle_sizes[2], xy,
                                utils.random_rot(self.rs))
        world_config['objects'][name] = _object
      elif 'pillar' in name:
        _object = po.get_pillar(name, self._obstacle_sizes[3], xy,
                                utils.random_rot(self.rs))
        world_config['objects'][name] = _object
    world_config.update(self.objective.build_world_config())
    return world_config

  def compute_reward(self, robot: Robot, world: World) -> float:
    raise NotImplementedError

  def reset(self):
    """ Resets the task. Allows the concrete implementation to perform
    specialized reset """
    self._seed += 1
    self.rs = np.random.RandomState(self._seed)

  def generate_new_layout(self) -> dict:
    """ Rejection sample a placement of objects to find a layout. """
    for _ in range(10000):
      new_layout = self._sample_layout()
      if new_layout is not None:
        return new_layout
    else:
      raise utils.ResamplingError('Failed to sample layout of objects')

  def _sample_layout(self) -> dict:
    """ Sample a single layout, returning True if successful, else False. """

    def placement_is_valid(xy: np.ndarray, layout: dict):
      for other_name, other_xy in layout.items():
        other_keepout = self._placements[other_name][1]
        dist = np.sqrt(np.sum(np.square(xy - other_xy)))
        if dist < other_keepout + self.config.placements_margin + keepout:
          return False
      return True

    layout = {}
    for name, (placements, keepout) in self._placements.items():
      conflicted = True
      for _ in range(100):
        xy = utils.draw_placement(self.rs, placements, keepout)
        if placement_is_valid(xy, layout):
          conflicted = False
          break
      if conflicted:
        return {}
      layout[name] = xy
    return layout

  @property
  def obstacles(self) -> List[np.ndarray]:
    pass

  @property
  def goal(self) -> np.ndarray:
    pass

  @property
  def object(self) -> np.ndarray:
    pass
