from typing import Tuple, Iterable

import numpy as np

import learn2learn_safely.consts as c
import learn2learn_safely.utils as utils


def object_attributes_to_xml(attributes_dict: dict) -> str:
  return """<body name="{name}" pos="{pos}" quat="{quat}">
                        <freejoint name="{name}"/>
                        <geom name="{name}" type="{type}" size="{size}" 
                        density="{density}" rgba="{rgba}" group="{group}"/>
            </body>""".format(
      **{k: utils.convert_to_text(v) for k, v in attributes_dict.items()})


def geom_attributes_to_xml(attributes_dict: dict) -> str:
  return """<body name="{name}" pos="{pos}" quat="{quat}">
                    <geom name="{name}" type="{type}" size="{size}"
                     rgba="{rgba}" group="{group}" contype="{contype}" 
                     conaffinity="{conaffinity}"/>
            </body>""".format(
      **{k: utils.convert_to_text(v) for k, v in attributes_dict.items()})


def mocap_attributes_to_xml(attributes_dict: dict) -> str:
  return """<body name="{name}" mocap="true">
                      <geom name="{name}" type="{type}" size="{size}" 
                      rgba="{rgba}" pos="{pos}" quat="{quat}" 
                      contype="{contype}" conaffinity="{conaffinity}" 
                      group="{group}"/>
            </body>""".format(
      **{k: utils.convert_to_text(v) for k, v in attributes_dict.items()})


def get_vase(name: str,
             size: float,
             xy: np.ndarray,
             rot: float,
             color: np.ndarray = c.VASES_COLOR) -> Tuple[Iterable[str], str]:
  vase_dict = {
      'name': name,
      'size': np.ones(3) * size,
      'type': 'box',
      'density': c.VASES_DENSITY,
      'pos': np.r_[xy, size - c.VASES_SINK],
      'quat': utils.rot2quat(rot),
      'group': c.GROUP_OBSTACLES,
      'rgba': color
  }
  return [object_attributes_to_xml(vase_dict)], ''


def get_gremlin(
    name: str,
    size: float,
    xy: np.ndarray,
    rot: float,
    color: np.ndarray = c.GREMLINS_COLOR) -> Tuple[Iterable[str], str]:
  gremlin_obj_dict = {
      'name': name,
      'size': np.ones(3) * size,
      'type': 'box',
      'density': c.GREMLINS_DENSITY,
      'pos': np.r_[xy, size],
      'quat': utils.rot2quat(rot),
      'group': c.GROUP_OBSTACLES,
      'rgba': color
  }
  gremlin_mocap_dict = gremlin_obj_dict.copy()
  gremlin_mocap_dict.update({
      'contype': 0,
      'conaffinity': 0,
  })
  gremlin_mocap_dict['name'] = name + 'mocap'
  weld_constraint = """<weld name="{name}" body1="{body1}" body2="{body2}" 
    solref=".02 1.5"/>""".format(
      name=name, body1=name + 'mocap', body2=name)
  return [
      object_attributes_to_xml(gremlin_obj_dict),
      mocap_attributes_to_xml(gremlin_mocap_dict)
  ], weld_constraint


def get_hazard(
    name: str,
    size: float,
    xy: np.ndarray,
    rot: float,
    color: np.ndarray = c.HAZARDS_COLOR * [1, 1, 1, 0.25]
) -> Tuple[Iterable[str], str]:
  hazard_dict = {
      'name': name,
      'size': [size, c.HAZARDS_HEIGHT],
      'pos': np.r_[xy, 2. * c.HAZARDS_HEIGHT],
      'quat': utils.rot2quat(rot),
      'type': 'cylinder',
      'contype': 0,
      'conaffinity': 0,
      'group': c.GROUP_OBSTACLES,
      'rgba': color
  }
  return [geom_attributes_to_xml(hazard_dict)], ''


def get_pillar(
    name: str,
    size: float,
    xy: np.ndarray,
    rot: float,
    color: np.ndarray = c.PILLARS_COLOR) -> Tuple[Iterable[str], str]:
  pillar_dict = {
      'name': name,
      'size': [size, c.PILLARS_HEIGHT],
      'pos': np.r_[xy, c.PILLARS_HEIGHT],
      'quat': utils.rot2quat(rot),
      'type': 'cylinder',
      'contype': 1,
      'conaffinity': 1,
      'group': c.GROUP_OBSTACLES,
      'rgba': color
  }
  return [geom_attributes_to_xml(pillar_dict)], ''


def get_goal(
    name: str,
    size: float,
    xy: np.ndarray,
    rot: float,
    color: np.ndarray = c.GOAL_COLOR * [1, 1, 1, 0.25]
) -> Tuple[Iterable[str], str]:
  goal_dict = {
      'name': name,
      'size': [size, size / 2.],
      'pos': np.r_[xy, size / 2. + 1e-2],
      'quat': utils.rot2quat(rot),
      'type': 'cylinder',
      'contype': 0,
      'conaffinity': 0,
      'group': c.GROUP_GOAL,
      'rgba': color
  }
  return [geom_attributes_to_xml(goal_dict)], ''
