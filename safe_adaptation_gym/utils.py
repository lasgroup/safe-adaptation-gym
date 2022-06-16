from typing import Union, Tuple

import numpy as np


class ResamplingError(AssertionError):
  """ Raised when we fail to sample a valid distribution of objects or goals """
  pass


def random_rot(rs: np.random.RandomState) -> float:
  """ Use internal random state to get a random rotation in radians """
  return rs.uniform(0, 2 * np.pi)


def update_layout(layout: dict, mujoco_bridge):
  """ Update layout dictionary with new places of objects """
  for k in layout.keys():
    layout[k] = mujoco_bridge.body_pos(k)[:2].copy()


def constrain_placement(placement: tuple, keepout: float) -> tuple:
  """ Helper function to constrain a single placement by the keepout radius """
  xmin, ymin, xmax, ymax = placement
  return xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout


def draw_placement(rs: np.random.RandomState, placements: Union[dict, None],
                   placement_extents: Tuple[float, float, float, float],
                   keepout: float) -> np.ndarray:
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
  if placements is None:
    choice = constrain_placement(placement_extents, keepout)
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
      choice = constrained[rs.choice(len(constrained), p=probs)]
  xmin, ymin, xmax, ymax = choice
  return np.array([rs.uniform(xmin, xmax), rs.uniform(ymin, ymax)])


# https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of
# -dictionaries/24088493#24088493
def merge(a, b, path=None):
  """ merges b into a """
  if path is None:
    path = []
  for key in b:
    if key in a:
      if isinstance(a[key], dict) and isinstance(b[key], dict):
        merge(a[key], b[key], path + [str(key)])
      elif a[key] == b[key]:
        pass  # same leaf value
      else:
        raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
    else:
      a[key] = b[key]
  return a


def rot2quat(theta):
  """ Get a quaternion rotated only about the Z axis """
  return np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)], dtype='float64')


def convert_to_text(v: Union[int, float, str, np.ndarray]) -> str:
  """ Convert a value into a string for mujoco XML """
  if isinstance(v, (int, float, str)):
    return str(v)
  # Numpy arrays and lists
  return ' '.join(str(i) for i in np.asarray(v))


def convert_from_text(txt: str) -> np.ndarray:
  """ Convert a string vector into a numpy array """
  return np.array(txt.split(), dtype=np.float32)


def quat2mat(quat):
  """ Convert Quaternion to a 3x3 Rotation Matrix using mujoco """
  from dm_control import mujoco
  q = np.array(quat, dtype='float64')
  m = np.zeros(9, dtype='float64')
  mujoco.mju_quat2Mat(m, q)
  return m.reshape((3, 3))

