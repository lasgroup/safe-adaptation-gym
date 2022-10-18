from typing import List, Tuple, Union, Iterable

import numpy as np

_NUM_JOINTS = 12

_JOINTS = [
    "hip_1_z", "hip_2_z", "hip_3_z", "hip_4_z", "hip_1_y", "hip_2_y", "hip_3_y",
    "hip_4_y", "ankle_1", "ankle_2", "ankle_3", "ankle_4"
]


def disable_joints(
    rs: np.random.RandomState, num_joints_to_disable: int
) -> List[Tuple[Tuple[str, str], Tuple[str, Union[float, Iterable[float]]]]]:
  joints = rs.choice(_NUM_JOINTS, size=num_joints_to_disable, replace=False)
  return [
      (('actuator', _JOINTS[joint_id]), ('gear', [0.])) for joint_id in joints
  ]
