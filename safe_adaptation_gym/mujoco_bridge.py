import os
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import xmltodict
from dm_control import mujoco, mjcf

import safe_adaptation_gym.consts as c
import safe_adaptation_gym.utils as utils


class MujocoBridge:
  DEFAULT = {
      'robot_xy': np.zeros(2),  # Robot XY location
      'robot_rot': 0,  # Robot rotation about Z axis
      'floor_size': [3.5, 3.5, .1],  # Used for displaying the floor
      'bodies': {},
      'others': {},
      'robot_ctrl_range_scale': None,
      'modify_tree': [],
  }

  def __init__(self, robot, addition_render_objects_specs=None, config=None):
    """ config - JSON string or dict of configuration. """
    if config is None:
      config = {}
    self.config = deepcopy(self.DEFAULT)
    self.config.update(deepcopy(config))
    self.config = SimpleNamespace(**self.config)
    self.robot = robot
    self.physics = None
    self._addition_render_objects_specs = addition_render_objects_specs
    self._build()

  def get_sensor(self, name: str) -> np.ndarray:
    return self.physics.named.data.sensordata[name]

  def _build(self):
    """ Build a world, including generating XML and moving objects """
    # Read in the base XML (contains robot, camera, floor, etc)
    robot_base_path = os.path.join(c.BASE_DIR, self.robot.base_path)
    arena_mjcf = mjcf.from_path(robot_base_path)
    if self._addition_render_objects_specs is not None:
      robot_site = arena_mjcf.find('site', 'robot')
      for creat_fn, specs in self._addition_render_objects_specs:
        robot_site.attach(creat_fn(**specs))
    if self.config.modify_tree is not None:
      for (namespace, id_), (attribute, value) in self.config.modify_tree:
        setattr(arena_mjcf.find(namespace, id_), attribute, value)
    robot_base_xml = arena_mjcf.to_xml_string()
    xml = xmltodict.parse(robot_base_xml)

    # Convenience accessor for xml dictionary
    worldbody = xml['mujoco']['worldbody']

    # Move robot position to starting position
    worldbody['body']['@pos'] = utils.convert_to_text(
        np.r_[self.config.robot_xy, self.robot.z_height])
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
      # TODO (yarden): use dm_control mjcf for this stuff!
      asset = xmltodict.parse("""
                <asset>
                    <texture type="skybox" builtin="gradient" rgb1="0.527 
                    0.582 0.906" rgb2="0.1 0.1 0.35"
                        width="800" height="800" markrgb="1 1 1" 
                        mark="random" random="0.001"/>
                        <texture name="interactobj" builtin="checker" width="150" height="150"
             rgb1="0.87 0.87 0" rgb2="1 1 0" markrgb="0.7 0.7 0"/>
                    <texture name="texplane" builtin="checker" height="100" 
                    width="100"
                        rgb1="0.7 0.7 0.7" rgb2="0.8 0.8 0.8" type="2d"/>
                    <material name="MatPlane" reflectance="0.1" 
                    shininess="0.1" specular="0.1"
                        texrepeat="10 10" texture="texplane"/>
                    <material name="interactobj" reflectance="0.1" 
                    shininess="0.1" specular="0.1"
                        texrepeat="10 10" texture="interactobj"/>
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
    for name, xml_str in self.config.others.items():
      xml['mujoco'][name] = xmltodict.parse(xml_str)
    # Instantiate simulator
    xml_string = xmltodict.unparse(xml)
    self.physics = mujoco.Physics.from_xml_string(xml_string)
    if self.config.robot_ctrl_range_scale is not None:
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

  def robot_contacts(self, group_geom_names: Iterable[str]) -> int:
    """
    Check if any of the geoms in group_geom_names is in contact with any of
    the geom names attached to the robot. Returns the number of contacts with
    the robot
    """
    part_of_robot = lambda name: name in self.robot.geom_names  # noqa
    in_group = lambda name: any(
        name.startswith(geom)  # noqa
        for geom in group_geom_names)
    count = 0
    for geom1, geom2 in self.contacts:
      count += int((part_of_robot(geom1) or part_of_robot(geom2)) and
                   (in_group(geom1) or in_group(geom2)))
    return count

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
    return self.physics.named.data.subtree_com[name]

  def body_pos(self, name: str) -> np.ndarray:
    """ Get the position of a named body in the simulator world reference
    frame """
    return self.physics.named.data.xpos[name]

  def body_mat(self, name: str) -> np.ndarray:
    """ Get the rotation matrix of a named body in the simulator world
    reference frame """
    return self.physics.named.data.xmat[name].reshape([3, 3])

  def body_vel(self, name: str) -> np.ndarray:
    """ Get the velocity of a named body in the simulator world reference
    frame """
    vel = self.physics.named.data.subtree_linvel[name]
    return vel

  def set_body_pos(self, name: str, pos: np.ndarray):
    """ Sets position for a given body name """
    dim = np.prod(pos.shape)
    self.physics.named.model.body_pos[name][:dim] = pos

  def set_mocap_pos(self, name: str, pos: np.ndarray):
    self.physics.named.data.mocap_pos[name] = pos

  def set_body_quat(self, name: str, quat: np.ndarray):
    """ Sets quaternion for a given body name """
    self.physics.named.model.body_quat[name] = quat

  def set_control(self, action: np.ndarray):
    """ Sets the control """
    self.physics.set_control(action)

  @property
  def time(self):
    return self.physics.data.time

  @property
  def nu(self):
    return self.physics.model.nu

  @property
  def contacts(self):
    contacts = self.physics.data.contact
    geom1, geom2 = contacts.geom1, contacts.geom2
    geom1 = map(lambda id_: self.physics.model.id2name(id_, 'geom'), geom1)
    geom2 = map(lambda id_: self.physics.model.id2name(id_, 'geom'), geom2)
    return list(zip(geom1, geom2))

  @property
  def actuator_ctrlrange(self):
    return self.physics.model.actuator_ctrlrange

  @property
  def user_groups(self):
    return self.physics.named.model.geom_user

  @property
  def site_rgba(self):
    return self.physics.named.model.site_rgba

  @property
  def geom_rgba(self):
    return self.physics.named.model.geom_rgba

  @property
  def robot_in_floor(self) -> bool:
    robot_x, robot_y = self.body_pos('robot')[:2]
    # Assumes square floor.
    floor_x, floor_y = self.physics.named.model.geom_size['floor', :2]
    return abs(robot_x) < floor_x and abs(robot_y) < floor_y
