from typing import Tuple, Union, Optional, List

import dm_control.rl.control
import gym
import numpy as np
from gym.core import ActType, ObsType

from learn2learn_safely.mujoco_bridge import MujocoBridge
from learn2learn_safely.tasks.task import Task
from learn2learn_safely.world import World
from learn2learn_safely.robot import Robot
import learn2learn_safely.utils as utils


# TODO (yarden): make sure that the environment is wrapped with a timelimit
class SafetyGym(gym.Env):
  NUM_LIDAR_BINS = 16
  LIDAR_MAX_DIST = 3.
  BASE_SENSORS = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
  DOGGO_EXTRA_SENSORS = [
      'touch_ankle_1a', 'touch_ankle_2a', 'touch_ankle_3a', 'touch_ankle_4a',
      'touch_ankle_1b', 'touch_ankle_2b', 'touch_ankle_3b', 'touch_ankle_4b'
  ]

  def __init__(self, robot_base, rgb_observation=False, config=None):
    self._world: Optional[World] = None
    self.base_config = config
    self._rgb_observation = rgb_observation
    self.robot = Robot(robot_base)
    self.mujoco_bridge = MujocoBridge(robot_base)
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
    # Keeping magic 10 from Safety-Gym
    # https://github.com/openai/safety-gym/blob
    # /f31042f2f9ee61b9034dd6a416955972911544f5/safety_gym/envs/engine.py#L1253
    for _ in range(10):
      try:
        self._world.set_mocaps(self.mujoco_bridge)
        self.mujoco_bridge.physics.step()
      except dm_control.rl.control.PhysicsError as er:
        print('PhysicsError', er)
        return self.observation, -10., True, {'cost': 0.}
    self.mujoco_bridge.physics.forward()
    reward, terminal, info = self._world.compute_reward(self.mujoco_bridge)
    info = {'cost': self._world.compute_cost(self.mujoco_bridge)}
    return self.observation, reward, terminal, info

  def reset(
      self,
      *,
      seed: Optional[int] = None,
      return_info: bool = False,
      options: Optional[dict] = None,
  ) -> Union[ObsType, Tuple[ObsType, dict]]:
    """ Reset the physics simulation and return observation """
    assert self._world is not None, 'A task should be first set before reset.'
    if seed:
      self._seed = seed
    else:
      self._seed += 1
    self.rs = np.random.RandomState(self._seed)
    self._world.rs = self.rs
    self._build_world()
    return self.observation

  def render(self, mode="human"):
    """ Renders the mujoco simulator """
    pass

  def seed(self, seed=None):
    """ Set internal random state seeds """
    self._seed = np.random.randint(2**32) if seed is None else seed
    self.rs = np.random.RandomState(self._seed)

  @property
  def observation(self) -> np.ndarray:
    if self._rgb_observation:
      image = self.mujoco_bridge.physics.render(
          height=64, width=64, camera_id='vision')
      image = np.clip(image, 0, 255).astype(np.float32)
      return image / 255.0
    obstacles, objects, goal = self._world.body_positions(self.mujoco_bridge)
    obstacles_lidar = self._lidar(obstacles)
    goal_lidar = self._lidar(goal)
    objects_lidar = self._lidar(objects)
    sensors = self._sensors()
    obs = np.zeros(self.observation_space.shape, self._observation_space.dtype)
    offset = 0
    for observation in [obstacles_lidar, objects_lidar, goal_lidar] + sensors:
      size = np.prod(observation.shape)
      obs[offset:offset + size] = observation.flat
      offset += size
    return obs

  @property
  def action_space(self) -> gym.spaces.Box:
    return self._action_space

  @property
  def observation_space(self) -> gym.spaces.Box:
    if self._observation_space is None:
      if self._rgb_observation:
        self._observation_space = gym.spaces.Box(0., 1., (64, 64, 3),
                                                 np.float32)
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
      z = np.complex(*ego_xy(pos))  # X, Y as real, imaginary components
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
      sensors.append(utils.quat2mat(quat))
    return sensors
