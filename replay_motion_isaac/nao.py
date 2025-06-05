import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from pathlib import Path
current_dir = Path(__file__).parent

def DEG(x):
    return x * 3.1415926 / 180.0

# The absolute path of the nao_d.usd file
usd_file_path = f"nao/nao_d.usd"
usd_full_path = str(current_dir/usd_file_path)

NAO_V50_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_full_path,
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
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    # rot=( 1,0,0,0 )
    # rotx(90deg)=( 0.7071068,0.7071068,0,0 )
    # rotx(45deg)=( 0.9238795,0.3826834,0,0 )
    # roty(90deg)=( 0.7071068,0,0.7071068,0 ) 
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4), rot=( 1,0,0,0 ), 
        joint_pos={
    "HeadYaw":0.0,"HeadPitch":0.0,
    "LShoulderPitch":DEG(90),"LShoulderRoll":DEG(10),"LElbowYaw":0.0,"LElbowRoll":-0.035,"LWristYaw":DEG(-90),
    "RShoulderPitch":DEG(90),"RShoulderRoll":DEG(-10),"RElbowYaw":0.0,"RElbowRoll":0.035,"RWristYaw":DEG(90),
    "LHipYawPitch":0.0,"LHipRoll":0.0,"LHipPitch":-0.433096,"LKneePitch":0.853201,"LAnklePitch":-0.420104,"LAnkleRoll":0.0,
    "RHipYawPitch":0.0,"RHipRoll":0.0,"RHipPitch":-0.433096,"RKneePitch":0.853201,"RAnklePitch":-0.420104,"RAnkleRoll":0.0},
    joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "HeadYaw": ImplicitActuatorCfg(
            joint_names_expr=["HeadYaw"],
            effort_limit=1.547, velocity_limit=8.26797,
            stiffness=150.0, damping=5.0
        ),
        "HeadPitch": ImplicitActuatorCfg(
            joint_names_expr=["HeadPitch"],
            effort_limit=1.532, velocity_limit=7.19407,
            stiffness=150.0, damping=5.0
        ),
        "ShoulderPitch": ImplicitActuatorCfg(
            joint_names_expr=[".*ShoulderPitch"],
            effort_limit=1.329, velocity_limit=8.26797,
            stiffness=150.0, damping=5.0
        ),
        "ShoulderRoll": ImplicitActuatorCfg(
            joint_names_expr=[".*ShoulderRoll"],
            effort_limit=1.7835, velocity_limit=7.19407,
            stiffness=180.0, damping=5.0
        ),
        "ElbowYaw": ImplicitActuatorCfg(
            joint_names_expr=[".*ElbowYaw"],
            effort_limit=1.547, velocity_limit=8.26797,
            stiffness=150.0, damping=5.0
        ),
        "ElbowRoll": ImplicitActuatorCfg(
            joint_names_expr=[".*ElbowRoll"],
            effort_limit=1.532, velocity_limit=7.19407,
            stiffness=150.0, damping=5.0
        ),
        "WristYaw": ImplicitActuatorCfg(
            joint_names_expr=[".*WristYaw"],
            effort_limit=0.4075, velocity_limit=24.6229,
            stiffness=50.0, damping=2.0
        ),
        "HipYawPitch": ImplicitActuatorCfg(
            joint_names_expr=[".*HipYawPitch"],
            effort_limit=3.348, velocity_limit=4.16174,
            stiffness=350.0, damping=5.0
        ),
        "HipRoll": ImplicitActuatorCfg(
            joint_names_expr=[".*HipRoll"],
            effort_limit=3.348, velocity_limit=4.161747,
            stiffness=350.0, damping=5.0
        ),
        "HipPitch": ImplicitActuatorCfg(
            joint_names_expr=[".*HipPitch"],
            effort_limit=3.023, velocity_limit=6.40239,
            stiffness=300.0, damping=5.0
        ),
        "KneePitch": ImplicitActuatorCfg(
            joint_names_expr=[".*KneePitch"],
            effort_limit=3.023, velocity_limit=6.40239,
            stiffness=300.0, damping=5.0
        ),
        "AnklePitch": ImplicitActuatorCfg(
            joint_names_expr=[".*AnklePitch"],
            effort_limit=3.023, velocity_limit=6.40239,
            stiffness=300.0, damping=5.0
        ),
        "AnkleRoll": ImplicitActuatorCfg(
            joint_names_expr=[".*AnkleRoll"],
            effort_limit=3.348, velocity_limit=4.16174,
            stiffness=350.0, damping=5.0
        ),

        # "1A": ImplicitActuatorCfg(
        #     joint_names_expr=[".*HipYawPitch", ".*HipRoll",".*AnkleRoll"],
        #     effort_limit=13.6884, velocity_limit=4.3178,
        #     stiffness=280.0, damping=5.0
        # ),
        # "1B": ImplicitActuatorCfg(
        #     joint_names_expr=[".*HipPitch",".*KneePitch",".*AnklePitch"],
        #     effort_limit=8.8978, velocity_limit=6.6425,
        #     stiffness=200.0, damping=5.0
        # ),
        # "2A": ImplicitActuatorCfg(
        #     joint_names_expr=[".*WristYaw"],
        #     effort_limit=0.475734, velocity_limit=17.3809,
        #     stiffness=20.0, damping=0.5
        # ),
        # "3A": ImplicitActuatorCfg(
        #     joint_names_expr=["HeadYaw",".*ElbowYaw"],
        #     effort_limit=2.148861, velocity_limit=7.4566,
        #     stiffness=45.0, damping=2.0
        # ),
        # "3B": ImplicitActuatorCfg(
        #     joint_names_expr=["HeadPitch",".*ShoulderRoll",".*ElbowRoll"],
        #     effort_limit=2.477046, velocity_limit=6.4687,
        #     stiffness=50.0, damping=2.0
        # ),
        # "4A": ImplicitActuatorCfg(
        #     joint_names_expr=[".*ShoulderPitch"],
        #     effort_limit=3.366048, velocity_limit=8.8503,
        #     stiffness=70.0, damping=3.0
        # ),
        # "head": ImplicitActuatorCfg(
        #     joint_names_expr=["HeadYaw","HeadPitch"],
        #     effort_limit=10.0, velocity_limit=7.0, 
        #     stiffness=150.0, damping=5.0
        # ),
        # "arms": ImplicitActuatorCfg(
        #     joint_names_expr=[".*Shoulder.*",".*Elbow.*",".*Wrist.*"],
        #     effort_limit=10.0, velocity_limit=7.0, 
        #     stiffness=150.0, damping=5.0
        # ),
        # "legs_pitch": ImplicitActuatorCfg(
        #     joint_names_expr=[".*HipPitch",".*KneePitch",".*AnklePitch"],
        #     effort_limit=20.0, velocity_limit=6.4, 
        #     stiffness={
        #         ".*HipPitch": 200.0,
        #         ".*Knee.*": 200.0,
        #         ".*AnklePitch": 200.0,
        #         },
        #     damping={
        #         ".*HipPitch.*": 5.0,
        #         ".*Knee.*": 5.0,
        #         ".*AnklePitch": 5.0,
        #         },
        # ),
        # "legs_roll": ImplicitActuatorCfg(
        #     joint_names_expr=[".*HipRoll",".*AnkleRoll"],
        #     effort_limit=20.0, velocity_limit=4.0, 
        #     stiffness={
        #         ".*HipRoll": 150.0,
        #         ".*AnkleRoll": 150.0
        #         },
        #     damping={
        #         ".*HipRoll.*": 5.0,
        #         ".*AnkleRoll": 5.0
        #         },
        # ),
        # "torso": ImplicitActuatorCfg(
        #     effort_limit=30.0,
        #     velocity_limit=4.0,
        #     joint_names_expr=[".*HipYawPitch"],
        #     stiffness=200.0,
        #     damping=5.0,
        # ),
    },
)
