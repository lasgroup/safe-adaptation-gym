from typing import Tuple, Union, Optional, List, Dict

import dm_control.rl.control
import gym
import numpy as np
from gym.core import ActType, ObsType

from safe_adaptation_gym.mujoco_bridge import MujocoBridge
from safe_adaptation_gym.tasks.task import Task
from safe_adaptation_gym.world import World
from safe_adaptation_gym.robot import Robot
import safe_adaptation_gym.utils as utils
from safe_adaptation_gym.render import make_additional_render_objects


class SafeAdaptationGym(gym.Env):
  NUM_LIDAR_BINS = 16
  LIDAR_MAX_DIST = 5.
  BASE_SENSORS = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
  DOGGO_EXTRA_SENSORS = [
      'touch_ankle_1a', 'touch_ankle_2a', 'touch_ankle_3a', 'touch_ankle_4a',
      'touch_ankle_1b', 'touch_ankle_2b', 'touch_ankle_3b', 'touch_ankle_4b'
  ]

  def __init__(self,
               robot_base: str,
               rgb_observation: bool = False,
               config: Optional[Dict] = None,
               render_lidars_and_collision: bool = False,
               render_options: Optional[Dict] = None):
    self._world: Optional[World] = None
    self.base_config = config
    self._rgb_observation = rgb_observation
    self.robot = Robot(robot_base)
    self._render_lidars_and_collision = render_lidars_and_collision
    self._render_options = render_options if render_options is not None else {}
    visualization_objects = None
    if render_lidars_and_collision:
      visualization_objects = make_additional_render_objects(
          self.NUM_LIDAR_BINS)
    self.mujoco_bridge = MujocoBridge(self.robot, visualization_objects)
    self._seed = np.random.randint(2**32)
    self.rs = np.random.RandomState(self._seed)
    self._action_space = gym.spaces.Box(
        -1, 1, (self.mujoco_bridge.nu,), dtype=np.float32)
    self._observation_space = None
    self._sensors_names = self.BASE_SENSORS
    if 'doggo' in robot_base:
      self._sensors_names += self.DOGGO_EXTRA_SENSORS

  def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
    """ Take a step and return observation, reward, done, and info """
    action = np.array(action, copy=False)
    action_range = self.mujoco_bridge.actuator_ctrlrange
    # Action space is in [-1, 1] by definition. Each time we sample a new
    # world (a.k.a. CMDP), we sample a new scale (see _build_world_config) to
    # accomodate variability in the dynamics.
    action = (
        action * action_range[:, 1] + self._world.config.action_noise *
        self.rs.normal(size=self.mujoco_bridge.nu))
    self.mujoco_bridge.set_control(
        np.clip(action, action_range[:, 0], action_range[:, 1]))
    try:
      self._world.set_mocaps(self.mujoco_bridge)
      self.mujoco_bridge.physics.step(nstep=5 if 'point' in self.robot.base_path else 10)
    except dm_control.rl.control.PhysicsError as er:
      print('PhysicsError', er)
      return self.observation, -10., True, {'cost': 0.}
    self.mujoco_bridge.physics.forward()
    reward, terminal, info = self._world.compute_reward(self.mujoco_bridge)
    cost = self._world.compute_cost(self.mujoco_bridge)
    info = {'cost': cost, 'bound': self._world.bound}
    observation = self.observation
    if self._render_lidars_and_collision:
      self._update_lidars_and_collision(self.lidar_observations, cost)
    return observation, reward, terminal, info

  def reset(
      self,
      *,
      seed: Optional[int] = None,
      return_info: bool = False,
      options: Optional[dict] = None,
  ) -> Union[ObsType, Tuple[ObsType, dict]]:
    """ Reset the physics simulation and return observation """
    assert self._world is not None or (options is not None and
                                       'task' in options), ('A task should be '
                                                            'first set before '
                                                            'reset.')
    if seed is not None:
      self._seed = seed
    else:
      self._seed += 1
    self.rs = np.random.RandomState(self._seed)
    if options is not None and 'task' in options:
      self.set_task(options['task'])
      return self.observation
    self._world.rs = self.rs
    self._build_world()
    return self.observation

  def render(self, mode='human'):
    """ Renders the mujoco simulator """
    return self.mujoco_bridge.physics.render(**self._render_options)

  def seed(self, seed=None):
    """ Set internal random state seeds """
    self._seed = np.random.randint(2**32) if seed is None else seed
    self.rs = np.random.RandomState(self._seed)
    if self._world is not None:
      self._world.rs = self.rs

  @property
  def observation(self) -> np.ndarray:
    if self._rgb_observation:
      image = self.mujoco_bridge.physics.render(
          height=64, width=64, camera_id='vision')
      image = np.clip(image, 0, 255).astype(np.uint8)
      return image
    sensors = self._sensors()
    sensors = np.concatenate([s[:, None] for s in sensors]).squeeze()
    lidars = self.lidar_observations
    obs = np.concatenate([lidars, sensors])
    return obs

  @property
  def lidar_observations(self) -> np.ndarray:
    obstacles, objects, goal = self._world.body_positions(self.mujoco_bridge)
    obstacles_lidar = self._lidar(obstacles)
    goal_lidar = self._lidar(goal)
    objects_lidar = self._lidar(objects)
    return np.concatenate([obstacles_lidar, objects_lidar, goal_lidar])

  @property
  def action_space(self) -> gym.spaces.Box:
    return self._action_space

  @property
  def observation_space(self) -> gym.spaces.Box:
    if self._observation_space is None:
      if self._rgb_observation:
        self._observation_space = gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
      else:
        sensors = self._sensors()
        # Lidar for (1) obstacles, (2) objects and (3) goal.
        lidar_size = 3 * self.NUM_LIDAR_BINS
        sensors_flat_size = int(
            sum(np.prod(sensor.shape) for sensor in sensors))
        low = np.array([0.] * lidar_size + [-np.inf] * sensors_flat_size)
        high = np.array([1.] * lidar_size + [np.inf] * sensors_flat_size)
        self._observation_space = gym.spaces.Box(
            low,
            high,
            shape=(lidar_size + sensors_flat_size,),
            dtype=np.float32)
    return self._observation_space

  def set_task(self, task: Task):
    """ Sets a new task to be solved """
    self._world = World(self.rs, task, self.robot, self.base_config)
    self._build_world()

  def _build_world(self):
    self.mujoco_bridge.rebuild(self._world.sample_layout())
    self._world.reset(self.mujoco_bridge)

  def _lidar(self, positions: List[np.ndarray]) -> np.ndarray:
    """
    Return a robot-centric lidar observation of a list of positions.

    Lidar is a set of bins around the robot (divided evenly in a circle).
    The detection directions are exclusive and exhaustive for a full 360 view.
    Each bin reads 0 if there are no objects in that direction.
    If there are multiple objects, the distance to the closest one is used.
    Otherwise, the bin reads the fraction of the distance towards the robot.

    E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
    and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
    (The reading can be thought of as "closeness" or inverse distance)

    This encoding has some desirable properties:
        - bins read 0 when empty
        - bins smoothly increase as objects get close
        - maximum reading is 1.0 (where the object overlaps the robot)
        - close objects occlude far objects
        - constant size observation with variable numbers of objects
    """
    obs = np.zeros(self.NUM_LIDAR_BINS)

    def ego_xy(pos):
      robot_3vec = self.mujoco_bridge.robot_pos()
      robot_mat = self.mujoco_bridge.robot_mat()
      pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
      world_3vec = pos_3vec - robot_3vec
      return np.matmul(world_3vec, robot_mat)[:2]

    for pos in positions:
      pos = np.asarray(pos)
      if pos.shape == (3,):
        pos = pos[:2]  # Truncate Z coordinate
      z = complex(*ego_xy(pos))  # X, Y as real, imaginary components
      dist = np.abs(z)
      angle = np.angle(z) % (np.pi * 2)
      bin_size = (np.pi * 2) / self.NUM_LIDAR_BINS
      bin_ = int(angle / bin_size)
      bin_angle = bin_size * bin_
      sensor = max(0, self.LIDAR_MAX_DIST - dist) / self.LIDAR_MAX_DIST
      obs[bin_] = max(obs[bin_], sensor)
      alias = (angle - bin_angle) / bin_size
      assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle ' \
                              f'{angle}, bin {bin_}'
      bin_plus = (bin_ + 1) % self.NUM_LIDAR_BINS
      bin_minus = (bin_ - 1) % self.NUM_LIDAR_BINS
      obs[bin_plus] = max(obs[bin_plus], alias * sensor)
      obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
    return obs

  def _sensors(self) -> List[np.ndarray]:
    """ Returns all robot-attached sensors """
    sensors = []
    for sensor in (self._sensors_names + self.robot.hinge_vel_names +
                   self.robot.ballangvel_names):
      sensors.append(self.mujoco_bridge.get_sensor(sensor))
    for sensor in self.robot.hinge_pos_names:
      theta = float(self.mujoco_bridge.get_sensor(sensor))
      sensors.append(np.array([np.sin(theta), np.cos(theta)]))
    for sensor in self.robot.ballquat_names:
      quat = self.mujoco_bridge.get_sensor(sensor)
      sensors.append(utils.quat2mat(quat).ravel())
    return sensors

  def _update_lidars_and_collision(self, observations, cost):
    obstacles_lidar, objects_lidar, goal_lidar = np.split(observations, 3)

    def update_lidar_alpha(name, lidar, color):
      for i, value in enumerate(lidar):
        alpha = min(1., value + .1)
        self.mujoco_bridge.site_rgba['{}lidar/{}'.format(name,
                                                         i)] = color * alpha

    for i, (name, lidar) in enumerate(
        zip(['obstacles', 'goal', 'objects'],
            [obstacles_lidar, goal_lidar, objects_lidar])):
      color = np.array([0., 0., 0., 1.])
      color[i] = 1.
      update_lidar_alpha(name, lidar, color)
    if cost > 0.:
      self.mujoco_bridge.site_rgba['collision/indicator'][-1] = 0.5
    else:
      self.mujoco_bridge.site_rgba['collision/indicator'][-1] = 0.
