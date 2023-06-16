# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The superclass for Isaac Sim based environments."""


import abc
import gym
import numpy as np
import torch
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Union

import math
import omni.isaac.orbit.utils.kit as kit_utils
#from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.markers import StaticMarker
from omni.isaac.orbit.objects import RigidObject
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulator
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import quat_inv, quat_mul, random_orientation, sample_uniform, scale_transform
from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from omni.isaac.orbit_envs.locomotion.pepper import PepperEnvCfg, RandomizationCfg


class PepperEnv(IsaacEnv):
  
    def __init__(self, cfg: PepperEnvCfg, headless: bool = False, viewport: bool = False, **kwargs):
        
         # copy configuration
        self.cfg = cfg
        # parse the configuration for controller configuration
        # note: controller decides the robot control mode
        self._pre_process_cfg()
        # create classes (these are called by the function :meth:`_design_scene`)
        self.robot = MobileManipulator(cfg=self.cfg.robot)
        self.object = RigidObject(cfg=self.cfg.object)

        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()

        # prepare the observation manager
        self._observation_manager = MoveObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        # prepare the reward manager
        self._reward_manager = MoveRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space: base joint state + base-position + goal-position + actions
        num_obs = self._observation_manager.group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")

        # Take an initial step to initialize the scene.
        # This is required to compute quantities like Jacobians used in step().
        self.sim.step()
        # -- fill up buffers
        self.robot.update_buffers(self.dt)
        self.object.update_buffers(self.dt)
        
    def _design_scene(self) -> List[str]:
        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane")
        # robot
        self.robot.spawn(self.template_env_ns + "/Pepper")
        # object
        self.object.spawn(self.template_env_ns + "/Object")
        # setup debug visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            # create point instancer to visualize the goal points
            self._goal_markers = StaticMarker(
                "/Visuals/object_goal",
                self.num_envs,
                usd_path=self.cfg.goal_marker.usd_path,
                scale=self.cfg.goal_marker.scale,
            )
            # create marker for viewing end-effector pose
            self._ee_markers = StaticMarker(
                "/Visuals/ee_current",
                self.num_envs,
                usd_path=self.cfg.frame_marker.usd_path,
                scale=self.cfg.frame_marker.scale,
            )
        # return list of global prims
        return ["/World/defaultGroundPlane"]
            
    def _reset_idx(self, env_ids: VecEnvIndices):
        # randomize the MDP
        # -- robot DOF state
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # -- pepper goal pose
        self._randomize_object_initial_pose(env_ids=env_ids, cfg=self.cfg.randomization.object_initial_pose)
        self._randomize_robot_initial_pose(env_ids=env_ids, cfg=self.cfg.randomization.robot_initial_pose)
        
        # -- Reward logging
        # fill extras with episode information
        self.extras["episode"] = dict()
        # reset
        # -- rewards manager: fills the sums for terminated episodes
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
        # -- obs manager
        self._observation_manager.reset_idx(env_ids)
        # -- reset history
        self.previous_actions[env_ids] = 0
        # -- MDP reset
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        self.actions = actions.clone().to(device=self.device)
        self.actions[:, :3] = 5*self.actions[:, :3]
            # offset actuator command with position offsets
        dof_pos_offset = self.robot.data.actuator_pos_offset
        self.actions[:, : self.robot.base_num_dof] -= dof_pos_offset[:, : self.robot.base_num_dof]
        # perform physics stepping
        for _ in range(self.cfg.control.decimation):
            # # set actions into buffer.
            self.robot.apply_action(self.actions)
            # # simulate
            self.sim.step(render=self.enable_render)
            # check that simulation is playing
            if self.sim.is_stopped():
                return
        # post-step:
        # -- compute common buffers
        self.robot.update_buffers(self.dt)
        self.object.update_buffers(self.dt)
        
        # -- compute MDP signals
        # reward
        self.reward_buf = self._reward_manager.compute()

        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        # -- add information to extra if task completed
        d_err=torch.norm(self.object_init_pose_w[:, :2] - self.robot.data.base_dof_pos[:, :2],dim=1)
        yaw =   torch_utils.get_euler_xyz(self.object_init_pose_w[:, 3:7])[-1]
        yaw_robot = self.robot.data.base_dof_pos[:,2]
        diff_angle = yaw_robot*180.0/np.pi - yaw*180.0/np.pi
        diff_angle_norm =  ((diff_angle+180)%360 - 180)/180
        cond_1_low = diff_angle_norm < 0.02
        cond_1_high = diff_angle_norm  > -0.02
        cond_1 = torch.where(d_err < 0.02 , 0.5, 0.0)
        cond_2 = torch.where((cond_1_low & cond_1_high) , 0.5, 0.0)
        is_success = cond_1 + cond_2 
        self.extras["is_success"] = torch.where(is_success ==1.0, 1, self.reset_buf)
        # -- update USD visualization
        if self.cfg.viewer.debug_vis and self.enable_render:
            self._debug_vis()
            
    def _get_observations(self) -> VecEnvObs:
        # compute observations
        return self._observation_manager.compute()

    """
    Helper functions - Scene handling.
    """

    def _pre_process_cfg(self) -> None:
        """Pre-processing of configuration parameters."""
        self.cfg.robot.rigid_props.disable_gravity = True
        self.cfg.object.collision_props.collision_enabled = False

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

        # convert configuration parameters to torchee
        # randomization
        # -- initial pose
        config = self.cfg.randomization.object_initial_pose
        for attr in ["position_uniform_min", "position_uniform_max"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()

        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Pepper")
        self.object.initialize(self.env_ns + "/.*/Object")
        self.num_actions = self.robot.num_actions

        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        # buffers
        self.object_root_pose = torch.zeros((self.num_envs, 7), device=self.device)
        # time-step = 0
        self.object_init_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_init_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        
    def _debug_vis(self):
        """Visualize the environment in debug mode."""
        # apply to instance manager
        # -- goal
        self._goal_markers.set_world_poses(self.object_init_pose_w[:, :3], self.object_init_pose_w[:, 3:7])
        # -- end effector 
        ee_positions = self.robot.data.base_dof_pos+ self.envs_positions
        ee_positions[:, 2]=0
        w  = torch.cos(self.robot.data.base_dof_pos[:,2]/2)
        x  = torch.zeros_like(w,device=self.device)
        y  = torch.zeros_like(w,device=self.device)
        z  = torch.sin(self.robot.data.base_dof_pos[:,2]/2)
        ee_orientation= torch.stack([w,x,y,z],dim=1)
        self._ee_markers.set_world_poses(ee_positions,ee_orientation)

    """
    Helper functions - MDP.
    """

    def _check_termination(self) -> None:
        # access buffers from simulator
        object_pos = self.object.data.root_pos_w - self.envs_positions
        # extract values from buffer
        self.reset_buf[:] = 0
        # compute resets
        # -- when task is successful
        if self.cfg.terminations.is_success:
             position_error = torch.norm(object_pos[:,:2] - self.robot.data.base_dof_pos[:,:2], dim=1)
             pos_reach = position_error < 0.02
             yaw =   torch_utils.get_euler_xyz(self.object.data.root_state_w[:, 3:7])[-1]
             yaw_robot = self.robot.data.base_dof_pos[:,2]
             torch_pi=torch.acos(torch.zeros(1)).item() * 2
             diff_angle = yaw_robot*180.0/torch_pi - yaw*180.0/torch_pi
             diff_angle_norm =  ((diff_angle+180)%360 - 180)/180
             ore_reach = diff_angle_norm < 0.02
             self.reset_buf = torch.where((pos_reach & ore_reach), 1, self.reset_buf)
            
        # -- episode length
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)
            return self.reset_buf

    def _randomize_object_initial_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.ObjectInitialPoseCfg):
        """Randomize the initial pose of the object."""
        # get the default root state
        root_state = self.object.get_default_root_state(env_ids)
        # -- object root position
        if cfg.position_cat == "default":
            pass
        elif cfg.position_cat == "uniform":
            # sample uniformly from box
            # note: this should be within in the workspace of the robot
            root_state[:, 0:3] = sample_uniform(
                cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
            )
        else:
            raise ValueError(f"Invalid category for randomizing the object positions '{cfg.position_cat}'.")
        # -- object root orientation
        if cfg.orientation_cat == "default":
            pass
        elif cfg.orientation_cat == "uniform":
            # sample uniformly in yaw
            ramdom_yaw = (torch.rand(len(env_ids))-0.5)*2 * np.pi
            random_eula = torch.zeros(len(env_ids), 3)
            random_eula[:, -1] = ramdom_yaw
            random_quat = torch_utils.euler_angles_to_quats(random_eula)
            root_state[:, 3:7] = random_quat
        else:
            raise ValueError(f"Invalid category for randomizing the object orientation '{cfg.orientation_cat}'.")
        # transform command from local env to world
        root_state[:, 0:3] += self.envs_positions[env_ids]
        # update object init pose
        self.object_init_pose_w[env_ids] = root_state[:, 0:7]
        # set the root state
        self.object.set_root_state(root_state, env_ids=env_ids)
        
    def _randomize_robot_initial_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.RobotInitialPoseCfg):
            # get the default root state       
            root_state = self.robot.get_default_root_state(env_ids)
            # -- object root position
            if cfg.position_cat == "default":
                pass
            elif cfg.position_cat == "uniform":
                # sample uniformly from box
                # note: this should be within in the workspace of the robot
                root_state[:, 0:3] = 5*(2*torch.rand(1)-1)
            else:
                raise ValueError(f"Invalid category for randomizing the object positions '{cfg.position_cat}'.")
            # -- object root orientation
            if cfg.orientation_cat == "default":
                pass
            elif cfg.orientation_cat == "uniform":
                # sample uniformly in yaw
                root_state[:, 3:7] = random_orientation(len(env_ids), self.device)
                R, P, Y =  torch_utils.get_euler_xyz(root_state[:, 3:7])
                R = R*0
                P = P*0
                random_euler = torch.stack([P,R,Y],dim=1)
                root_state[:, 3:7] = torch_utils.euler_angles_to_quats(random_euler)
            else:
                raise ValueError(f"Invalid category for randomizing the robot orientation '{cfg.orientation_cat}'.")
            # transform command from local env to world
            # set the root state
            random_pose, random_vel = self.robot.get_default_dof_state(env_ids=env_ids)
            random_pose = 5*(2*torch.rand(random_pose.shape,device=self.device)-1)
            random_pose[:,3:] = 0.0
            random_pose[:, 2] = torch.where(random_pose[:, 2] <= -6.28, -6.28, random_pose[:, 2] )
            random_pose[:, 2] = torch.where(random_pose[:, 2] >= 6.28, 6.28, random_pose[:, 2] )
            self.robot.set_dof_state(random_pose, random_vel, env_ids) 
            root_state[:, 0:3] += self.envs_positions[env_ids]
            # update object init pose
            self.robot_init_pose_w[env_ids] = root_state[:, 0:7]
            

class MoveObservationManager(ObservationManager):
    """Reward manager for single-arm reaching environment."""

    def base_dof_pos(self, env: PepperEnv):
        """DOF positions for the arm."""
        return env.robot.data.base_dof_pos
 
    def base_orientation(self, env: PepperEnv):
        """Robot Base Orientation."""
        yaw_robot = env.robot.data.base_dof_pos[:,2]
        roll_robot = torch.zeros_like(yaw_robot)
        pitch_robot = torch.zeros_like(yaw_robot)
        quat_w_robot = torch.stack([roll_robot, pitch_robot, yaw_robot], dim=1)
        return quat_w_robot
    
    def object_positions(self, env: PepperEnv):
        """Current object position."""
        return env.object_init_pose_w[:, :2] - env.envs_positions[:, :2]

    def object_orientations(self, env: PepperEnv):
        """Current object orientation."""
        # make the first element positive
        return env.object_init_pose_w[:,3:7]


class MoveRewardManager(RewardManager):
    """Reward manager for pepper moving environment."""
    
    def reaching_position(self, env: PepperEnv):
        #Penalize base tracking position error using L2-kernel.
        distance  =  torch.norm(env.robot.data.base_dof_pos[:, :2] - env.object_init_pose_w[:, :2] + env.envs_positions[:, :2], dim=1)
        return -distance
    
    # ### should be the angle difference between the current world_rotation to the target world_rotation

    def reaching_orientation(self, env: PepperEnv):
         #Penalize base oreintation
         yaw =   torch_utils.get_euler_xyz(env.object_init_pose_w[:, 3:7])[-1]
         yaw_robot = env.robot.data.base_dof_pos[:,2]
         yaw_robot_degree = yaw_robot*180.0/np.pi 
         yaw_target_degree = yaw*180.0/np.pi
         diff_angle = yaw_robot_degree- yaw_target_degree
         diff_angle_norm =  ((diff_angle+180)%360 - 180)
         rwd = torch.abs(diff_angle_norm/180)
         return -rwd
    
    def tracking_position_success(self, env: PepperEnv, threshold: float):
         #Sparse reward if object is tracking successfully.
         x_err = torch.sqrt(torch.square(env.object_init_pose_w[:, 0] - env.robot.data.base_dof_pos[:, 0] + env.envs_positions[:, 0]))
         y_err = torch.sqrt(torch.square(env.object_init_pose_w[:, 1] - env.robot.data.base_dof_pos[:, 1] + env.envs_positions[:, 1]))
         d_err  =  torch.sqrt(torch.square(x_err)+torch.square(y_err))
         yaw =   torch_utils.get_euler_xyz(env.object_init_pose_w[:, 3:7])[-1]
         yaw_robot = env.robot.data.base_dof_pos[:,2]
         diff_angle = yaw_robot*180.0/np.pi - yaw*180.0/np.pi
         diff_angle_norm =  ((diff_angle+180)%360 - 180)/180
         cond_1_low = diff_angle_norm < 0.02
         cond_1_high = diff_angle_norm  > -0.02
         cond_1 = torch.where(d_err < 0.02 , 0.5, 0.0)  #Success condition 1: distance difference <0.02
         cond_2 = torch.where((cond_1_low & cond_1_high) , 0.5, 0.0)    #Success condition 2: oreientation difference <0.02
         is_success = cond_1 + cond_2 
         rwd = torch.where(is_success == 1.0 , 5.0, 0.0)
         return rwd
     