import numpy as np

import learn2learn_safely.consts as c


def get_vase(name: str,
             size: float,
             xy: np.ndarray,
             rot: float,
             color: np.ndarray = c.VASES_COLOR) -> dict:
  return {
      'name': name,
      'size': np.ones(3) * size,
      'type': 'box',
      'density': c.VASES_DENSITY,
      'pos': np.r_[xy, size - c.VASES_SINK],
      'rot': rot,
      'group': c.GROUP_OBSTACLES,
      'rgba': color
  }


def get_gremlin(name: str,
                size: float,
                xy: np.ndarray,
                rot: float,
                color: np.ndarray = c.GREMLINS_COLOR) -> dict:
  return {
      'name': name,
      'size': np.ones(3) * size,
      'type': 'box',
      'density': c.GREMLINS_DENSITY,
      'pos': np.r_[xy, size],
      'rot': rot,
      'group': c.GROUP_OBSTACLES,
      'rgba': color
  }


def get_hazard(name: str,
               size: float,
               xy: np.ndarray,
               rot: float,
               color: np.ndarray = c.HAZARDS_COLOR * [1, 1, 1, 0.25]) -> dict:
  return {
      'name': name,
      'size': [size, c.HAZARDS_HEIGHT],
      'pos': np.r_[xy, 2. * c.HAZARDS_HEIGHT],
      'rot': rot,
      'type': 'cylinder',
      'contype': 0,
      'conaffinity': 0,
      'group': c.GROUP_OBSTACLES,
      'rgba': color
  }


def get_pillar(name: str,
               size: float,
               xy: np.ndarray,
               rot: float,
               color: np.ndarray = c.PILLARS_COLOR) -> dict:
  return {
      'name': name,
      'size': [size, c.PILLARS_HEIGHT],
      'pos': np.r_[xy, c.PILLARS_HEIGHT],
      'rot': rot,
      'type': 'cylinder',
      'group': c.GROUP_OBSTACLES,
      'rgba': color
  }


def get_goal(name: str,
             size: float,
             xy: np.ndarray,
             rot: float,
             color: np.ndarray = c.GOAL_COLOR * [1, 1, 1, 0.25]) -> dict:
  return {
      'name': name,
      'size': [size, size / 2.],
      'pos': np.r_[xy, size / 2. + 1e-2],
      'rot': rot,
      'type': 'cylinder',
      'contype': 0,
      'conaffinity': 0,
      'group': c.GROUP_GOAL,
      'rgba': color
  }
