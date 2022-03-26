import os

from dm_control import mujoco

import learn2learn_safely.consts as c


class Robot:
  """ Simple utility class for getting mujoco-specific info about a robot """

  def __init__(self, path):
    base_path = os.path.join(c.BASE_DIR, path)
    self.sim = mujoco.Physics.from_xml_path(base_path)
    self.sim.forward()

    # Needed to figure out z-height of free joint of offset body
    self.z_height = self.sim.named.data.xpos['robot'][2]
    # Get a list of geoms in the robot
    self.geom_names = [
        n for n in self.sim.named.data.geom_xpos.axes.row.names if n != 'floor'
    ]
    # Needed to figure out the observation spaces
    self.nq = self.sim.model.nq
    self.nv = self.sim.model.nv
    # Needed to figure out action space
    self.nu = self.sim.model.nu
    # Needed to figure out observation space
    # See engine.py for an explanation for why we treat these separately
    self.hinge_pos_names = []
    self.hinge_vel_names = []
    self.ballquat_names = []
    self.ballangvel_names = []
    self.sensor_dim = {}

    for name in self.sim.named.model.sensor_objtype.axes.row.names:
      self.sensor_dim[name] = self.sim.named.model.sensor_dim[name]
      sensor_type = self.sim.named.model.sensor_type[name]
      if self.sim.named.model.sensor_objtype[name] == mujoco.mjtObj.mjOBJ_JOINT:
        joint_id = self.sim.named.model.sensor_objid[name]
        joint_type = self.sim.model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
          if sensor_type == mujoco.mjtSensor.mjSENS_JOINTPOS:
            self.hinge_pos_names.append(name)
          elif sensor_type == mujoco.mjtSensor.mjSENS_JOINTVEL:
            self.hinge_vel_names.append(name)
          else:
            raise ValueError(
                'Unrecognized sensor type {} for joint'.format(sensor_type))
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
          if sensor_type == mujoco.mjtSensor.mjSENS_BALLQUAT:
            self.ballquat_names.append(name)
          elif sensor_type == mujoco.mjtSensor.mjSENS_BALLANGVEL:
            self.ballangvel_names.append(name)
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
          # Adding slide joints is trivially easy in code,
          # but this removes one of the good properties about our observations.
          # (That we are invariant to relative whole-world transforms)
          # If slide joints are added we sould ensure this stays true!
          raise ValueError('Slide joints in robots not currently supported')
