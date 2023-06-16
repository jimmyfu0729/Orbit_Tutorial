from omni.isaac.orbit.actuators.group import ActuatorGroupCfg
from omni.isaac.orbit.actuators.group.actuator_group_cfg import ActuatorControlCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg
from omni.isaac.orbit.robots.single_arm import SingleArmManipulatorCfg
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulatorCfg

PEPPER_USD_PATH="/home/cjf/Desktop/Isaac/orbit/Orbit/Asset/pepper_description/urdf/pepper/pepper.usd"  # Here change the path to your  usd path.

PEPPER_CFG = MobileManipulatorCfg(
    meta_info=MobileManipulatorCfg.MetaInfoCfg(
        usd_path=PEPPER_USD_PATH,
	base_num_dof=3,
	arm_num_dof=0,
	tool_num_dof=0,
 
    ),
    init_state=MobileManipulatorCfg.InitialStateCfg(
        dof_pos={
        # pepper base
	   "XDisp": 0.0,
	   "YDisp": 0.0,
	   "ZRot": 0.0,
        },
        dof_vel={".*": 0.0},
    ),
    ee_info=MobileManipulatorCfg.EndEffectorFrameCfg(
        body_name="Head", rot_offset=(1.0, 0.0, 0.0, 0.0)
    ),
    rigid_props=SingleArmManipulatorCfg.RigidBodyPropertiesCfg(
        max_depenetration_velocity=5.0,
    ),
    collision_props=SingleArmManipulatorCfg.CollisionPropertiesCfg(
        contact_offset=0.005,
        rest_offset=0.0,
    ),
    articulation_props=SingleArmManipulatorCfg.ArticulationRootPropertiesCfg(
        enable_self_collisions=True,
    ),
    actuator_groups={
        "pepper_base": ActuatorGroupCfg(
            dof_names=["XDisp",
		       "YDisp",
                "ZRot",
		      ],
            model_cfg=ImplicitActuatorCfg(velocity_limit=1000.0, torque_limit=1000.0),
            control_cfg=ActuatorControlCfg(
                command_types=["v_abs"],
                stiffness={".*": 800.0},
                damping={".*": 400.0},
            ),
        ),
    },
)
