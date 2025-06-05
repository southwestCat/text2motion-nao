"""Configuration for a NAO robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from my_utils import DEG

from pathlib import Path
current_dir = Path(__file__).parent

# NAO usd file
nao_usd_file = f"../data/Robots/NAO/nao_d.usd"
nao_usd_full_path=str(current_dir/nao_usd_file)

NAO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=nao_usd_full_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32), rot=(1,0,0,0), joint_pos={
    "HeadYaw":0.0,"HeadPitch":0.0,
    "LShoulderPitch":DEG(90),"LShoulderRoll":DEG(10),"LElbowYaw":0.0,"LElbowRoll":-0.035,"LWristYaw":DEG(-90),
    "RShoulderPitch":DEG(90),"RShoulderRoll":DEG(-10),"RElbowYaw":0.0,"RElbowRoll":0.035,"RWristYaw":DEG(90),
    "LHipYawPitch":0.0,"LHipRoll":0.0,"LHipPitch":-0.433096,"LKneePitch":0.853201,"LAnklePitch":-0.420104,"LAnkleRoll":0.0,
    "RHipYawPitch":0.0,"RHipRoll":0.0,"RHipPitch":-0.433096,"RKneePitch":0.853201,"RAnklePitch":-0.420104,"RAnkleRoll":0.0},
    joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["HeadYaw","HeadPitch"],
            effort_limit=10.0, velocity_limit=7.0, 
            stiffness=150.0, damping=5.0
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*Shoulder.*",".*Elbow.*",".*Wrist.*"],
            effort_limit=10.0, velocity_limit=7.0, 
            stiffness=150.0, damping=5.0
        ),
        "legs_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*HipPitch",".*KneePitch",".*AnklePitch"],
            effort_limit=20.0, velocity_limit=6.4, 
            stiffness={
                ".*HipPitch": 200.0,
                ".*Knee.*": 200.0,
                ".*AnklePitch": 200.0,
                },
            damping={
                ".*HipPitch.*": 5.0,
                ".*Knee.*": 5.0,
                ".*AnklePitch": 5.0,
                },
        ),
        "legs_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*HipRoll",".*AnkleRoll"],
            effort_limit=20.0, velocity_limit=4.0, 
            stiffness={
                ".*HipRoll": 150.0,
                ".*AnkleRoll": 150.0
                },
            damping={
                ".*HipRoll.*": 5.0,
                ".*AnkleRoll": 5.0
                },
        ),
        "torso": ImplicitActuatorCfg(
            effort_limit=30.0,
            velocity_limit=4.0,
            joint_names_expr=[".*HipYawPitch"],
            stiffness=200.0,
            damping=5.0,
        ),
    },
)

"""Configuration for a NAO robot."""