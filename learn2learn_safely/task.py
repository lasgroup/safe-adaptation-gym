from copy import deepcopy
from types import SimpleNamespace
from typing import List, Mapping, Tuple, Union

import numpy as np

import learn2learn_safely.consts as c
import learn2learn_safely.primitive_objects as po
import learn2learn_safely.utils as utils
from learn2learn_safely.objectives.objective import Objective
from learn2learn_safely.robot import Robot
from learn2learn_safely.world import World


class Task:
  DEFAULT = {
      'placements_margin': 0.0,
      'robot_keepout': 0.4,
      'num_obstacles': 10,
      'hazards_size': 0.3,
      'vases_size': 0.1,
      'pillars_size': 0.2,
      'gremlins_size': 0.1,
      'hazards_keepout': 0.18,
      'gremlins_keepout': 0.4,
      'vases_keepout': 0.15,
      'pillars_keepout': 0.3,
      'obstacles_size_noise': 0.025,
      'keepout_noise': 0.025
  }

  def __init__(self,
               rs: np.random.RandomState,
               objective: Objective,
               robot_base: str,
               config=None):
    if config is None:
      config = {}
    tmp_config = deepcopy(self.DEFAULT)
    tmp_config.update(config)
    self.config = SimpleNamespace(**tmp_config)
    self.objective = objective
    self.rs = rs
    self.robot_base = robot_base
    self._obstacle_sizes = self.rs.normal([
        self.config.hazards_size, self.config.vases_size,
        self.config.pillars_size, self.config.gremlins_size
    ], self.config.obstacles_size_noise * np.ones(len(c.OBSTACLES)))
    self._obstacle_keepouts = self.rs.normal([
        self.config.hazards_keepout, self.config.vases_keepout,
        self.config.pillars_keepout, self.config.gremlins_keepout
    ], self.config.keepout_noise * np.ones(len(c.OBSTACLES)))
    self._placements = self._setup_placements()
    self._layout = None

  def _setup_placements(self):
    """ Build a dict of placements. """
    # TODO (yarden): Should the probability of different obstacle types be
    #  different?
    obstacle_samples = self.rs.multinomial(
        self.config.num_obstacles, [1 / len(c.OBSTACLES)] * len(c.OBSTACLES))
    placements = {
        **self._placement_dict_from_object('robot', 1),
    }
    for obstacle, num in zip(c.OBSTACLES, obstacle_samples):
      placements.update(self._placement_dict_from_object(obstacle, num))
    utils.merge(placements, self.objective.setup_placements())
    return placements

  def _placement_dict_from_object(self, name: str,
                                  num_placements) -> Mapping[str, tuple]:
    """ Get the placements' dict subset just for a given object name """
    placements_dict = {}
    object_fmt = name + '{i}' if name != 'robot' else name
    object_keepout = {
        'hazards': self._obstacle_keepouts[0],
        'vases': self._obstacle_keepouts[1],
        'pillars': self._obstacle_keepouts[2],
        'gremlins': self._obstacle_keepouts[3],
        'robot': self.config.robot_keepout
    }[name]
    for i in range(num_placements):
      # Set placements to None so that obstacles positions are sampled randomly.
      placements = None
      placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
    return placements_dict

  def build(self):
    # TODO (yarden): might need to clear layout upon reset?
    self._layout = self._generate_new_layout()
    return self._build_world_config()

  def _build_world_config(self):
    """ Create a world_config from our own config """
    world_config = {
        'robot_base': self.robot_base,
        'robot_xy': self._layout['robot'],
        'robot_rot': utils.random_rot(self.rs),
        'objects': {},
        'geoms': {},
        'mocaps': {}
    }
    for name, xy in self._layout.items():
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
        geom = po.get_hazard(name, self._obstacle_sizes[2], xy,
                             utils.random_rot(self.rs))
        world_config['geoms'][name] = geom
      elif 'pillar' in name:
        geom = po.get_pillar(name, self._obstacle_sizes[3], xy,
                             utils.random_rot(self.rs))
        world_config['geoms'][name] = geom
    utils.merge(world_config,
                self.objective.build_world_config(self._layout, self.rs))
    return world_config

  def compute_reward(self, world: World) -> Tuple[float, dict]:
    return self.objective.compute_reward(self._layout, self._placements,
                                         self.rs, world)

  def reset(self):
    """ Resets the task. Allows the concrete implementation to perform
    specialized reset """
    # TODO (yarden): How is this function to be used?
    pass

  def _generate_new_layout(self):
    """ Rejection sample a placement of objects to find a layout. """
    for _ in range(10000):
      new_layout = self._sample_layout()
      if new_layout is not None:
        return new_layout
    else:
      raise utils.ResamplingError('Failed to sample layout of objects')

  def _sample_layout(self) -> Union[dict, None]:
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
        return None
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
