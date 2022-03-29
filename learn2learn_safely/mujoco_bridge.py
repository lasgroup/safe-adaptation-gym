import os
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

from typing import Optional

import numpy as np
import xmltodict
from dm_control import mujoco

import learn2learn_safely.consts as c
import learn2learn_safely.utils as utils


class MujocoBridge:
  DEFAULT = {
      'robot_base': 'car.xml',  # Which robot XML to use as the base
      'robot_xy': np.zeros(2),  # Robot XY location
      'robot_z_height': 0.22,
      'robot_rot': 0,  # Robot rotation about Z axis
      'floor_size': [3.5, 3.5, .1],  # Used for displaying the floor
      'bodies': {}
  }

  def __init__(self, config=None):
    """ config - JSON string or dict of configuration. """
    if config is None:
      config = {}
    self.config = deepcopy(self.DEFAULT)
    self.config.update(deepcopy(config))
    self.config = SimpleNamespace(**self.config)
    self.physics = None
    self._build()

  def get_sensor(self, name: str) -> np.ndarray:
    return self.physics.named.sensordata[name].copy()

  def _build(self):
    """ Build a world, including generating XML and moving objects """
    # Read in the base XML (contains robot, camera, floor, etc)
    robot_base_path = os.path.join(c.BASE_DIR, self.config.robot_base)
    with open(robot_base_path) as f:
      robot_base_xml = f.read()
    xml = xmltodict.parse(robot_base_xml)

    # Convenience accessor for xml dictionary
    worldbody = xml['mujoco']['worldbody']

    # Move robot position to starting position
    worldbody['body']['@pos'] = utils.convert_to_text(
        np.r_[self.config.robot_xy, self.config.robot_z_height])
    worldbody['body']['@quat'] = utils.convert_to_text(
        utils.rot2quat(self.config.robot_rot))

    # We need this because xmltodict skips over single-item lists in the tree
    worldbody['body'] = [worldbody['body']]
    if 'geom' in worldbody:
      worldbody['geom'] = [worldbody['geom']]
    else:
      worldbody['geom'] = []

    # Add equality section if missing
    if 'equality' not in xml['mujoco']:
      xml['mujoco']['equality'] = OrderedDict()
    equality = xml['mujoco']['equality']
    if 'weld' not in equality:
      equality['weld'] = []

    # Add asset section if missing
    if 'asset' not in xml['mujoco']:
      # old default rgb1: ".4 .5 .6"
      # old default rgb2: "0 0 0"
      # light pink: "1 0.44 .81"
      # light blue: "0.004 0.804 .996"
      # light purple: ".676 .547 .996"
      # med blue: "0.527 0.582 0.906"
      # indigo: "0.293 0 0.508"
      asset = xmltodict.parse("""
                <asset>
                    <texture type="skybox" builtin="gradient" rgb1="0.527 
                    0.582 0.906" rgb2="0.1 0.1 0.35"
                        width="800" height="800" markrgb="1 1 1" 
                        mark="random" random="0.001"/>
                    <texture name="texplane" builtin="checker" height="100" 
                    width="100"
                        rgb1="0.7 0.7 0.7" rgb2="0.8 0.8 0.8" type="2d"/>
                    <material name="MatPlane" reflectance="0.1" 
                    shininess="0.1" specular="0.1"
                        texrepeat="10 10" texture="texplane"/>
                </asset>
                """)
      xml['mujoco']['asset'] = asset['asset']

    # Add light to the XML dictionary
    light = xmltodict.parse("""<b>
            <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true"
                exponent="1" pos="0 0 0.5" specular="0 0 0" castshadow="false"/>
            </b>""")
    worldbody['light'] = light['b']['light']

    # Make sure floor renders the same for every world
    for g in worldbody['geom']:
      if g['@name'] == 'floor':
        g.update({
            '@size': utils.convert_to_text(self.config.floor_size),
            '@rgba': '1 1 1 1',
            '@material': 'MatPlane'
        })

    # Add cameras to the XML dictionary
    cameras = xmltodict.parse("""<b>
            <camera name="fixednear" pos="0 -2 2" zaxis="0 -1 1"/>
            <camera name="fixedfar" pos="0 -5 5" zaxis="0 -1 1"/>
            </b>""")
    worldbody['camera'] = cameras['b']['camera']

    # Build and add a tracking camera (logic needed to ensure orientation
    # correct)
    theta = self.config.robot_rot
    xyaxes = dict(
        x1=np.cos(theta),
        x2=-np.sin(theta),
        x3=0,
        y1=np.sin(theta),
        y2=np.cos(theta),
        y3=1)
    pos = dict(
        xp=0 * np.cos(theta) + (-2) * np.sin(theta),
        yp=0 * (-np.sin(theta)) + (-2) * np.cos(theta),
        zp=2)
    track_camera = xmltodict.parse("""<b>
            <camera name="track" mode="track" pos="{xp} {yp} {zp}" 
            xyaxes="{x1} {x2} {x3} {y1} {y2} {y3}"/>
            </b>""".format(**pos, **xyaxes))
    worldbody['body'][0]['camera'] = [
        worldbody['body'][0]['camera'], track_camera['b']['camera']
    ]

    for name, (body_strings, weld_str) in self.config.bodies.items():
      for body_str in body_strings:
        worldbody['body'].append(xmltodict.parse(body_str)['body'])
      if weld_str:
        equality['weld'].append(xmltodict.parse(weld_str)['weld'])
    # Instantiate simulator
    xml_string = xmltodict.unparse(xml)
    self.physics = mujoco.Physics.from_xml_string(xml_string)
    self.physics.model.actuator_ctrlrange[:] *= (
        self.config.robot_ctrl_range_scale[:, None])
    # Recompute simulation intrinsics from new position
    self.physics.forward()

  def rebuild(self, config):
    """ Build a new physics from a model if the model changed """
    self.config = deepcopy(self.DEFAULT)
    self.config.update(deepcopy(config))
    self.config = SimpleNamespace(**self.config)
    self._build()

  def reset(self, config: dict):
    """ Sets the new positions of the resampled bodies. """
    assert config['bodies'].keys() == self.config.bodies.keys(), (
        'Some bodies were added or discarded')
    with self.physics.reset_context():
      for name, (body_strings, _) in config['bodies'].items():
        for xml_string in body_strings:
          body_xml = xmltodict.parse(xml_string)['body']
          if body_xml.get('@mocap', False):
            continue
          quat = utils.convert_from_text(body_xml['@quat'])
          pos = utils.convert_from_text(body_xml['@pos'])
          self.set_body_pos(body_xml['@name'], pos)
          self.set_body_quat(body_xml['@name'], quat)
      self.set_body_pos('robot', config['robot_xy'])
      self.set_body_quat('robot', utils.rot2quat(config['robot_rot']))

  def robot_com(self) -> np.ndarray:
    """ Get the position of the robot center of mass in the simulator world
    reference frame """
    return self.body_com('robot')

  def robot_pos(self) -> np.ndarray:
    """ Get the position of the robot in the simulator world reference frame """
    return self.body_pos('robot')

  def robot_mat(self) -> np.ndarray:
    """ Get the rotation matrix of the robot in the simulator world reference
    frame """
    return self.body_mat('robot')

  def robot_vel(self) -> np.ndarray:
    """ Get the velocity of the robot in the simulator world reference frame """
    return self.body_vel('robot')

  def body_com(self, name: str) -> np.ndarray:
    """ Get the center of mass of a named body in the simulator world
    reference frame """
    return self.physics.named.subtree_com[name].copy()

  def body_pos(self, name: str) -> np.ndarray:
    """ Get the position of a named body in the simulator world reference
    frame """
    return self.physics.named.xpos[name].copy()

  def body_mat(self, name: str) -> np.ndarray:
    """ Get the rotation matrix of a named body in the simulator world
    reference frame """
    return self.physics.named.xmat[name].reshape([3, 3]).copy()

  def body_vel(self, name: str) -> np.ndarray:
    """ Get the velocity of a named body in the simulator world reference
    frame """
    vel = self.physics.named.object_velocity(name, 'body').copy()
    return vel[0]

  def set_body_pos(self, name: str, pos: np.ndarray):
    """ Sets position for a given body name """
    dim = np.prod(pos.shape)
    self.physics.named.model.body_pos[name][:dim] = pos

  def set_body_quat(self, name: str, quat: np.ndarray):
    """ Sets quaternion for a given body name """
    self.physics.named.model.body_quat[name] = quat

  def set_control(self, action: np.ndarray):
    """ Sets the control """
    self.physics.set_control(action)
