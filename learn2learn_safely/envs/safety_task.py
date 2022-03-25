from types import SimpleNamespace
from typing import List, Mapping, Tuple

from copy import deepcopy

import numpy as np

import consts as c
import obstacle_utils as ou
from objectives.objective import Objective
from robot import Robot
from world import World


class ResamplingError(AssertionError):
  """ Raised when we fail to sample a valid distribution of objects or goals """
  pass


class Task:
  DEFAULT = {
      'placement_extents': [-2, -2, 2, 2],
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
    self._placements = None
    self.objective = objective
    self._seed = seed
    self.rs = np.random.RandomState(seed)
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
        'robot_rot': self.random_rot()
    }
    for name, xy in layout.items():
      if 'vase' in name:
        _object = ou.get_vase(name, self._obstacle_sizes[0], xy,
                              self.random_rot())
        world_config['objects'][name] = _object
      elif 'gremlin' in name:
        rot = self.random_rot()
        _object = ou.get_gremlin(name + 'obj', self._obstacle_sizes[1], xy, rot)
        mocap = ou.get_gremlin(name + 'mocap', self._obstacle_sizes[1], xy, rot)
        world_config['objects'][name + 'obj'] = _object
        world_config['mocaps'][name + 'mocap'] = mocap
      elif 'hazard' in name:
        _object = ou.get_hazard(name, self._obstacle_sizes[2], xy,
                                self.random_rot())
        world_config['objects'][name] = _object
      elif 'pillar' in name:
        _object = ou.get_pillar(name, self._obstacle_sizes[3], xy,
                                self.random_rot())
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
      raise ResamplingError('Failed to sample layout of objects')

  def _sample_layout(self) -> dict:
    """ Sample a single layout, returning True if successful, else False. """
    if self._placements is None:
      self._placements = self.setup_placements()

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
        xy = self._draw_placement(placements, keepout)
        if placement_is_valid(xy, layout):
          conflicted = False
          break
      if conflicted:
        return {}
      layout[name] = xy
    return layout

  def _draw_placement(self, placements, keepout):
    """
    Sample an (x,y) location, based on potential placement areas.

    Summary of behavior:

    'placements' is a list of (xmin, xmax, ymin, ymax) tuples that specify
    rectangles in the XY-plane where an object could be placed.

    'keepout' describes how much space an object is required to have
    around it, where that keepout space overlaps with the placement rectangle.

    To sample an (x,y) pair, first randomly select which placement rectangle
    to sample from, where the probability of a rectangle is weighted by its
    area. If the rectangles are disjoint, there's an equal chance the (x,y)
    location will wind up anywhere in the placement space. If they overlap, then
    overlap areas are double-counted and will have higher density. This allows
    the user some flexibility in building placement distributions. Finally,
    randomly draw a uniform point within the selected rectangle.

    """

    def constrain_placement(placement, keepout):
      """ Helper function to constrain a single placement by the keepout
      radius """
      xmin, ymin, xmax, ymax = placement
      return xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout

    if placements is None:
      choice = constrain_placement(self.config.placements_extents, keepout)
    else:
      # Draw from placements according to placeable area
      constrained = []
      for placement in placements:
        xmin, ymin, xmax, ymax = constrain_placement(placement, keepout)
        if xmin > xmax or ymin > ymax:
          continue
        constrained.append((xmin, ymin, xmax, ymax))
      assert len(
          constrained), 'Failed to find any placements with satisfy keepout'
      if len(constrained) == 1:
        choice = constrained[0]
      else:
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in constrained]
        probs = np.array(areas) / np.sum(areas)
        choice = constrained[self.rs.choice(len(constrained), p=probs)]
    xmin, ymin, xmax, ymax = choice
    return np.array([self.rs.uniform(xmin, xmax), self.rs.uniform(ymin, ymax)])

  @property
  def obstacles(self) -> List[np.ndarray]:
    pass

  @property
  def goal(self) -> np.ndarray:
    pass

  @property
  def object(self) -> np.ndarray:
    pass

  def random_rot(self):
    """ Use internal random state to get a random rotation in radians """
    return self.rs.uniform(0, 2 * np.pi)
