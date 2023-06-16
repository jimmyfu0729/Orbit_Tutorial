# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim with lights and a ground plane."""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="pepper", help="Name of the robot.")
parser.add_argument("--num_robots", type=int, default=2, help="Number of robots to spawn.")

args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""

import torch
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view
import omni.isaac.orbit.utils.kit as kit_utils
import omni.kit.commands
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.carb import set_carb_setting
from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulator
from omni.isaac.orbit.robots.config.PEPPER_CFG import PEPPER_CFG
import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401

"""
Main
"""

def main():
    """Spawns lights in the stage and sets the camera view."""
    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Clone Scene
    # Enable flatcache which avoids passing data over to USD structure
    # this speeds up the read-write operation of GPU buffers
    if sim.get_physics_context().use_gpu_pipeline:
        sim.get_physics_context().enable_flatcache(True)
    # Enable hydra scene-graph instancing
    # this is needed to visualize the scene when flatcache is enabled
    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    # Create interface to clone the scene
    cloner = GridCloner(spacing=3.5)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/pepper_ctr_0" will be cloned
    prim_utils.define_prim("/World/envs/pepper_ctr_0")

    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane")

    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )
    # get stage handle
    stage = omni.usd.get_context().get_stage()

    # spawn the pepper robot
    if args_cli.robot == "pepper":
        robot_cfg = PEPPER_CFG
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: pepper, pepper_base, pepper_simple, pepper_3dof")
 
    robot = MobileManipulator(cfg=robot_cfg)
    robot.spawn("/World/envs/pepper_ctr_0/pepper", translation=(0.0, 0.0, 0.0))
    # Clone the scene
    num_envs = args_cli.num_robots
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/pepper_ctr", num_paths=num_envs)
    envs_positions = cloner.clone(
        source_prim_path="/World/envs/pepper_ctr_0", prim_paths=envs_prim_paths, replicate_physics=True
    )
    # convert environment positions to torch tensor
    envs_positions = torch.tensor(envs_positions, dtype=torch.float, device=sim.device)
    # filter collisions within each environment instance
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", envs_prim_paths, global_paths=["/World/defaultGroundPlane"]
    )

    # Play the simulator
    sim.reset()
    # Initialize handles
    robot.initialize("/World/envs/pepper_ctr_.*/pepper")
    # Reset states
    robot.reset_buffers()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Create buffers to store actions
    robot_actions = torch.ones(robot.count, robot.num_actions, device=robot.device)
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0

    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # reset
        print("robot base pos",robot.data.base_dof_pos)
        
        if ep_step_count % 1000 == 0:
            sim_time = 0.0
            ep_step_count = 0
            # reset dof state
            dof_pos, dof_vel = robot.get_default_dof_state()
            r_state=robot.get_default_root_state()
            print("The root state is:",r_state)
            robot.set_dof_state(dof_pos, dof_vel)
            robot.reset_buffers()
            # reset command
            actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
            actions[:, 0 : robot.num_actions] = 0.0
            print(">>>>>>>> Reset!")
        # change the base action
	# 0:XDisp 1:YDisp 2:ZRot
        if ep_step_count == 200:
            actions[:, : robot.base_num_dof] = 0.0
            actions[:, 0] = 1.0
        if ep_step_count == 300:
            actions[:, : robot.base_num_dof] = 0.0
            actions[:, 0] = -1.0
        if ep_step_count == 400:
            actions[:, : robot.base_num_dof] = 0.0
            actions[:, 1] = 1.0
        if ep_step_count == 500:
            actions[:, : robot.base_num_dof] = 0.0
            actions[:, 1] = -1.0
        if ep_step_count == 600:
            actions[:, : robot.base_num_dof] = 0.0
            actions[:, 2] = 1.0
        if ep_step_count == 700:
            actions[:, : robot.base_num_dof] = 0.0
            actions[:, 2] = -1.0
        if ep_step_count == 800:
            actions[:, : robot.base_num_dof] = 0.0
            actions[:, 2] = 0.0

        # apply action
        robot.apply_action(actions)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)


if __name__ == "__main__":
    # Run stage
    main()
    # Close the simulator
    simulation_app.close()
