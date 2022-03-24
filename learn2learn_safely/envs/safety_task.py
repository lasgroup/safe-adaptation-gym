import abc
from types import SimpleNamespace
from typing import List

import numpy as np
from robot import Robot
from world import World


class ResamplingError(AssertionError):
  ''' Raised when we fail to sample a valid distribution of objects or goals '''
  pass


OBSTACLES = ['hazard', 'vase', 'gremlin', 'pillar']


class Task(abc.ABC):
  DEFAULT = SimpleNamespace(
      **{
          'placement_extents': [-2, -2, 2, 2],
          'placements_margin': 0.0,
          'robot_keepout': 0.4,
          'num_obstacles': 10
      })

  def __init__(self, seed: int, robot: Robot, world: World):
    self.world = world
    self.robot = robot
    self.placements = None
    self._seed = seed
    self.rs = np.random.RandomState(seed)

  def setup_placements(self):
    ''' Build a dict of placements. '''
    # TODO (yarden): Should the probability of different obstacle types be
    #  different?
    obstacle_samples = self.rs.multinomial(
        self.DEFAULT.num_obstacles, [1 / len(OBSTACLES)] * len(OBSTACLES))
    placements = {
        **self.placement_dict_from_object('robot', 1),
    }
    for obstacle, num in zip(OBSTACLES, obstacle_samples):
      placements.update(self.placement_dict_from_object(obstacle, num))
    return placements

  def placement_dict_from_object(self, object_name: str,
                                 num_placements) -> dict:
    ''' Get the placements dict subset just for a given object name '''
    placements_dict = {}
    name = object_name + 's' if num_placements > 1 else object_name
    object_fmt = object_name + '{i}' if num_placements > 1 else object_name
    object_num = getattr(self, name + '_num', None)
    object_locations = getattr(self, name + '_locations', [])
    object_placements = getattr(self, name + '_placements', None)
    object_keepout = getattr(self, name + '_keepout')
    for i in range(num_placements):
      if i < len(object_locations):
        x, y = object_locations[i]
        k = object_keepout + 1e-9  # Epsilon to account for numerical issues
        placements = [(x - k, y - k, x + k, y + k)]
      else:
        placements = object_placements
      placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
    return placements_dict

  def compute_reward(self) -> float:
    raise NotImplementedError

  def reset(self):
    ''' Resets the task. Allows the concrete implementation to perform
    specialized reset '''
    self._seed += 1
    self.rs = np.random.RandomState(self._seed)

  def generate_new_layout(self):
    ''' Rejection sample a placement of objects to find a layout. '''
    for _ in range(10000):
      new_layout = self._sample_layout()
      if new_layout is not None:
        return new_layout
    else:
      raise ResamplingError('Failed to sample layout of objects')

  def _sample_layout(self):
    ''' Sample a single layout, returning True if successful, else False. '''
    if self.placements is None:
      self.placements = self.setup_placements()

    def placement_is_valid(xy, layout):
      for other_name, other_xy in layout.items():
        other_keepout = self.placements[other_name][1]
        dist = np.sqrt(np.sum(np.square(xy - other_xy)))
        if dist < other_keepout + self.DEFAULT.placements_margin + keepout:
          return False
      return True

    layout = {}
    for name, (placements, keepout) in self.placements.items():
      conflicted = True
      for _ in range(100):
        xy = self._draw_placement(placements, keepout)
        if placement_is_valid(xy, layout):
          conflicted = False
          break
      if conflicted:
        return None
      layout[name] = xy
    return layout

  def _draw_placement(self, placements, keepout):
    '''
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

    '''

    def constrain_placement(placement, keepout):
      ''' Helper function to constrain a single placement by the keepout
      radius '''
      xmin, ymin, xmax, ymax = placement
      return (xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout)

    if placements is None:
      choice = constrain_placement(self.DEFAULT.placements_extents, keepout)
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
