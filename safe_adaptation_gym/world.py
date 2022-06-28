from copy import deepcopy
from types import SimpleNamespace
from typing import List, Mapping, Tuple, Union

import numpy as np

import safe_adaptation_gym.consts as c
import safe_adaptation_gym.primitive_objects as po
import safe_adaptation_gym.utils as utils
from safe_adaptation_gym.mujoco_bridge import MujocoBridge
from safe_adaptation_gym.robot import Robot
from safe_adaptation_gym.tasks.task import Task


class World:
  DEFAULT = {
      'placements_margin': 0.0,
      'robot_keepout': 0.4,
      'hazards_size': 0.2,
      'vases_size': 0.1,
      'pillars_size': 0.2,
      'gremlins_size': 0.1,
      'hazards_keepout': 0.18,
      'gremlins_keepout': 0.4,
      'vases_keepout': 0.15,
      'pillars_keepout': 0.3,
      'gremlins_travel': 0.35,
      'obstacles_size_noise_scale': 0.025,
      'robot_ctrl_range_scale': 0.05,
      'action_noise': 0.01,
      'max_bound': 25
  }

  def __init__(self,
               rs: np.random.RandomState,
               task: Task,
               robot: Robot,
               config=None):
    if config is None:
      config = {}
    tmp_config = deepcopy(self.DEFAULT)
    tmp_config.update(config)
    self.config = SimpleNamespace(**tmp_config)
    self.task = task
    self.rs = rs
    self.robot = robot
    obstacle_sizes_scale = self.task.obstacle_scales(
        self.rs) * self.config.obstacles_size_noise_scale + 1.0
    # Make sure that there are no negative size scales (otherwise objects
    # would have a negative size, causing mujoco not to compile)
    obstacle_sizes_scale = np.clip(obstacle_sizes_scale, 0.5, 1.25)
    self._obstacle_sizes = {
        k: scale * value
        for k, scale, value in zip(c.OBSTACLES, obstacle_sizes_scale, [
            self.config.hazards_size, self.config.vases_size,
            self.config.gremlins_size, self.config.pillars_size
        ])
    }
    keepouts = obstacle_sizes_scale * np.asarray([
        self.config.hazards_keepout, self.config.vases_keepout,
        self.config.gremlins_keepout, self.config.pillars_keepout
    ])
    obstacle_size_array = np.asarray(list(self._obstacle_sizes.values()))
    keepouts = np.where(keepouts < obstacle_size_array, obstacle_size_array,
                        keepouts)
    self._obstacle_keepouts = {
        k: value for k, value in zip(c.OBSTACLES, keepouts)
    }
    self._obstacle_keepouts['robot'] = self.config.robot_keepout
    self._placements = self._setup_placements()
    self._robot_ctrl_range_scale = self.task.ctrl_scale(
        self.rs, self.robot.nu) * self.config.robot_ctrl_range_scale + 1.0
    self._layout = None
    self.bound = self.task.constraint_bound(self.rs, self.config.max_bound)

  def _setup_placements(self):
    """ Build a dict of placements. """
    obstacle_samples = self.rs.multinomial(self.task.num_obstacles,
                                           self.task.obstacles_distribution)
    placements = {
        **self._placement_dict_from_object('robot', 1),
    }
    for obstacle, num in zip(c.OBSTACLES, obstacle_samples):
      if num > 0:
        placements.update(self._placement_dict_from_object(obstacle, num))
    utils.merge(placements, self.task.setup_placements())
    return placements

  def _placement_dict_from_object(self, name: str,
                                  num_placements) -> Mapping[str, tuple]:
    """ Get the placements' dict subset just for a given object name """
    placements_dict = {}
    object_fmt = name + '{i}' if name != 'robot' else name
    object_keepout = self._obstacle_keepouts[name]
    for i in range(num_placements):
      # Set placements to None so that obstacles positions are sampled randomly.
      placements = None
      placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
    return placements_dict

  def sample_layout(self):
    self._layout = self._generate_new_layout()
    return self._build_world_config()

  def _build_world_config(self):
    """ Create a world_config from our own config """
    world_config = {
        'robot_xy': self._layout['robot'],
        'robot_z_height': self.robot.z_height,
        # https://keisan.casio.com/exec/system/1180573169
        'robot_ctrl_range_scale': self._robot_ctrl_range_scale,
        'robot_rot': utils.random_rot(self.rs),
        'bodies': {}
    }
    for name, xy in self._layout.items():
      if 'vase' in name:
        world_config['bodies'][name] = po.get_vase(
            name, self._obstacle_sizes['vases'], xy, utils.random_rot(self.rs))
      elif 'gremlin' in name:
        world_config['bodies'][name] = po.get_gremlin(
            name, self._obstacle_sizes['gremlins'], xy,
            utils.random_rot(self.rs))
      elif 'hazard' in name:
        world_config['bodies'][name] = po.get_hazard(
            name, self._obstacle_sizes['hazards'], xy,
            utils.random_rot(self.rs))
      elif 'pillar' in name:
        world_config['bodies'][name] = po.get_pillar(
            name, self._obstacle_sizes['pillars'], xy,
            utils.random_rot(self.rs))
    utils.merge(world_config,
                self.task.build_world_config(self._layout, self.rs))
    return world_config

  def compute_reward(self,
                     mujoco_bridge: MujocoBridge) -> Tuple[float, bool, dict]:
    return self.task.compute_reward(self._layout, self._placements, self.rs,
                                    mujoco_bridge)

  def compute_cost(self, mujoco_bridge: MujocoBridge) -> float:
    mujoco_bridge.physics.forward()  # Ensure positions and contacts are correct
    cost = float(mujoco_bridge.robot_contacts(c.OBSTACLES))
    robot_pos = mujoco_bridge.robot_pos()[:2]
    for name in self._layout.keys():
      if not name.startswith('hazards'):
        continue
      dist = np.linalg.norm(robot_pos - mujoco_bridge.body_pos(name)[:2])
      if dist <= self._obstacle_sizes['hazards']:
        cost += 1.
    cost += self.task.compute_cost(mujoco_bridge)
    return float(cost > 0.)

  def set_mocaps(self, mujoco_bridge: MujocoBridge):
    phase = float(mujoco_bridge.time)
    for name, _ in self._layout.items():
      if name.startswith('gremlins'):
        target = np.array([np.sin(phase), np.cos(phase)])
        target *= self.config.gremlins_travel
        pos = np.r_[target, [0.]]
        mujoco_bridge.set_mocap_pos(name + 'mocap', pos)
    self.task.set_mocaps(mujoco_bridge)

  def reset(self, mujoco_bridge: MujocoBridge):
    """ Resets the task. Allows the concrete implementation to perform
    specialized reset """
    self.task.reset(self._layout, self._placements, self.rs, mujoco_bridge)

  def _generate_new_layout(self):
    """ Rejection sample a placement of objects to find a layout. """
    for _ in range(10000):
      new_layout = self._sample_layout()
      if new_layout is not None:
        return new_layout
    else:
      raise utils.ResamplingError('Failed to sample layout of objects')

  def _sample_layout(self) -> Union[dict, None]:
    """ Sample a single layout, returning True if successful, else False. """

    def placement_is_valid(xy: np.ndarray):
      for other_name, other_xy in layout.items():
        other_keepout = self._placements[other_name][1]
        dist = np.linalg.norm(xy - other_xy)
        if dist < other_keepout + self.config.placements_margin + keepout:
          return False
      return True

    layout = {}
    for name, (placements, keepout) in self._placements.items():
      conflicted = True
      for _ in range(100):
        xy = utils.draw_placement(self.rs, placements,
                                  self.task.placement_extents, keepout)
        if placement_is_valid(xy):
          conflicted = False
          break
      if conflicted:
        return None
      layout[name] = xy
    return layout

  def body_positions(
      self, mujoco_bridge: MujocoBridge
  ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    obstacles, objects, goal = [], [], []
    for name in self._layout.keys():
      group = mujoco_bridge.user_groups[name]
      if group == c.GROUP_OBSTACLES:
        obstacles.append(mujoco_bridge.body_pos(name))
      elif group == c.GROUP_GOAL:
        goal.append(mujoco_bridge.body_pos(name))
      elif group == c.GROUP_OBJECTS:
        objects.append(mujoco_bridge.body_pos(name))
    return obstacles, objects, goal
