# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.objects import RigidObjectCfg
from omni.isaac.orbit.robots.config.PEPPER_CFG import PEPPER_CFG
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, PhysxCfg, SimCfg, ViewerCfg

##
# Scene settings
##

@configclass
class ManipulationObjectCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(1.0, 1.0, 1.0),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
    physics_material = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/cubeMaterial"
    )
    collision_props = RigidObjectCfg.CollisionPropertiesCfg(
	collision_enabled=False,
    )

@configclass
class GoalMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.75, 0.75, 0.75]  # x,y,z


@configclass
class FrameMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.75, 0.75, 0.75]  # x,y,z


##
# MDP settings
##


@configclass
class RandomizationCfg:
    """Randomization of scene at reset."""

    @configclass
    class ObjectInitialPoseCfg:
        """Randomization of object initial pose."""

        # category
        position_cat: str = "uniform"  # randomize position: "default", "uniform"
        orientation_cat: str = "uniform"  # randomize position: "default", "uniform"
        # randomize position
        position_uniform_min = [-5, -5, 0.075]  # position (x,y,z)
        position_uniform_max = [5, 5, 0.08]  # position (x,y,z)
   
    @configclass
    class RobotInitialPoseCfg:
        """Randomization of object initial pose."""

        # category
        position_cat: str = "uniform"  # randomize position: "default", "uniform"
        orientation_cat: str = "uniform"  # randomize position: "default", "uniform"
        # randomize position
        position_uniform_min = [-5, -5, 0.5]  # position (x,y,z)
        position_uniform_max = [5, 5, 5]  # position (x,y,z)
        
    # initialize
    object_initial_pose: ObjectInitialPoseCfg = ObjectInitialPoseCfg()
    object_desired_pose: ObjectDesiredPoseCfg = ObjectDesiredPoseCfg()
    robot_initial_pose: RobotInitialPoseCfg = RobotInitialPoseCfg()

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg:
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = True
 
        # observation terms
        # -- base state
        base_dof_pos = {"scale": 1.0}
        base_orientation = {"scale":1.0}
        # -- object state
        object_positions = {"scale": 1.0}
        object_orientations = {"scale": 1.0}

    # global observation settings
    return_dict_obs_in_group = False
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- robot-centric
    reaching_position = {"weight": 1.5}
    reaching_orientation = {"weight": 3.0}
    tracking_position_success = {"weight": 5.0, "threshold": 0.03}


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    episode_timeout = True  # reset when episode length ended
    is_success = False  # reset when object is lifted


@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    control_type = "default"  # "default", "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 10



##
# Environment configuration
##


@configclass
class PepperEnvCfg(IsaacEnvCfg):
    """Configuration for the Lift environment."""

    # General Settings
    env: EnvCfg = EnvCfg(num_envs=4096, env_spacing=10.0, episode_length_s=20.0)
    viewer: ViewerCfg = ViewerCfg(debug_vis=True, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(
        dt=0.01,
        substeps=1,
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
            friction_offset_threshold=0.01,
            bounce_threshold_velocity=0.2,
        ),
    )

    # Scene Settings
    # -- robot
    robot: PEPPER_CFG = PEPPER_CFG
    # -- visualization marker
    goal_marker: GoalMarkerCfg = GoalMarkerCfg()
    frame_marker: FrameMarkerCfg = FrameMarkerCfg()
    # -- object
    object: ManipulationObjectCfg = ManipulationObjectCfg()
    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()
