from typing import List, Tuple, Union, Iterable

import numpy as np

from safe_adaptation_gym import utils

_NUM_JOINTS = 12

_JOINTS = [
    "hip_1_z", "hip_2_z", "hip_3_z", "hip_4_z", "hip_1_y", "hip_2_y", "hip_3_y",
    "hip_4_y", "ankle_1", "ankle_2", "ankle_3", "ankle_4"
]

_LEGS_DEFAULT_GEOMS = {
    'hip_1': np.array([0.098, 0.0566, -.05]),
    'ankle_1': np.array([-0.1176, -0.0679, -.1]),
    'hip_4': np.array([0.098, -0.0566, -.05]),
    'ankle_4': np.array([-0.1176, 0.0679, -.1]),
    'hip_2': np.array([0.098, 0.0566, -.05]),
    'ankle_2': np.array([-0.1176, -0.0679, -.1]),
    'hip_3': np.array([0.098, -0.0566, -.05]),
    'ankle_3': np.array([-0.1176, 0.0679, -.1])
}

_CRIPPLED_LEG_COLOR = utils.convert_to_text(np.array([1., 165 / 255, 0, 1]))


def disable_joints(
    rs: np.random.RandomState, num_joints_to_disable: int
) -> List[Tuple[Tuple[str, str], Tuple[str, Union[float, Iterable[float]]]]]:
  joints = rs.choice(_NUM_JOINTS, size=num_joints_to_disable, replace=False)
  return [
      (('actuator', _JOINTS[joint_id]), ('gear', [0.])) for joint_id in joints
  ]


def cripple_leg(rs: np.random.RandomState, disable_motors: bool = False):
  leg_id_to_cripple = rs.randint(5) + 1
  # Randomly don't cripple leg.
  if leg_id_to_cripple > 4:
    return []
  hip_id = f'hip_{leg_id_to_cripple}'
  ankle_id = f'ankle_{leg_id_to_cripple}'
  hip = _LEGS_DEFAULT_GEOMS[hip_id] * rs.uniform(0.3, 1.)
  ankle = _LEGS_DEFAULT_GEOMS[ankle_id] * rs.uniform(0.3, 1.)
  append_zeros = lambda x: f'0.0 0.0 0.0 {x}'
  hip = utils.convert_to_text(hip)
  ankle = utils.convert_to_text(ankle)
  hip_zeros = append_zeros(hip)
  ankle_zeros = append_zeros(ankle)
  mujoco_commands = [(('geom', hip_id), ('fromto', hip_zeros)),
                     (('geom', hip_id), ('rgba', _CRIPPLED_LEG_COLOR)),
                     (('body', hip_id), ('pos', hip)),
                     (('geom', ankle_id), ('fromto', ankle_zeros)),
                     (('geom', ankle_id), ('rgba', _CRIPPLED_LEG_COLOR)),
                     (('site', ankle_id + 'b'), ('pos', ankle))]
  if disable_motors:
    mujoco_commands += [(('actuator', hip_id + '_z'), ('gear', [0.])),
                        (('actuator', hip_id + '_y'), ('gear', [0.])),
                        (('actuator', ankle_id), ('gear', [0.]))]
  return mujoco_commands
