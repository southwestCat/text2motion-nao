import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import tasks.manager_based.pose_mapping.skeleton.mdp as mdp
from my_utils import DEG
import os

HOME_PATH=os.getenv('HOME')

##
# Pre-defined configs
##
from assets import NAO_CFG

##
# Scene definition
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        update_period=0.0,
        debug_vis=True,
        history_length=3, 
        track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    random_pos = mdp.MotionRandomCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        joint_names=[".*Shoulder.*", ".*Elbow.*", ".*WristYaw", ".*Hip.*", ".*Knee.*"],
        motion_path = HOME_PATH+"/PoseMapping/train_npy",
    )
    sequence_pose = mdp.MotionSequenceCommandCfg(
        asset_name="robot",
        resampling_time_range=(15.0, 15.0),
        joint_names=[".*Shoulder.*", ".*Elbow.*", ".*WristYaw", ".*Hip.*", ".*Knee.*"],
        motion_sequence_path = HOME_PATH+"/PoseMapping/play_npy",
        delayed_time=3.0
    )


@configclass
class ActionCfg:
    joint_pos = mdp.JointPositionHipYawActionCfg(
        asset_name="robot", 
        joint_names=["LHipYawPitch",".*HipRoll",".*HipPitch",".*Knee.*",".*Ankle.*",".*Shoulder.*",".*Elbow.*",".*WristYaw"],
        scale=0.2,
        use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "random_pos"})
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*Shoulder.*",".*Elbow.*",".*Wrist.*",".*Hip.*",".*Knee.*",".*Ankle.*"])
            },
            noise=Unoise(n_min=-0.05, n_max=0.05))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.1),
            "dynamic_friction_range": (0.2, 0.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.0, 0.2),
            "operation": "add",
        },
    )
    add_hand_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wrist.*"),
            "mass_distribution_params": (0.0, 0.1),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params={
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2)
            }
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_joint_position_exp = RewTerm(
        func=mdp.track_joint_command_targets_exp,
        weight=10.0,
        params={
            "command_name": "random_pos",
            "std": 20.0/4.0
        }
    )
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_walk,
    #     weight=1.5,
    #     params={
    #         "command_name": "random_pos",
    #         "threshold": 0.4,
    #         "walk_mode_threshold": 40.0,
    #         "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*ankle"]),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip.*",".*Knee.*"])
    #     }
    # )
    feet_flat_orientation_l2 = RewTerm(
        func=mdp.feet_flat_orientation_l2,
        weight=0.3,
        params={
            "std": math.sqrt(DEG(2)**2 / 4.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle")
        }
    )

    # feet_slide = RewTerm(
    #     func = mdp.feet_slide,
    #     weight=-1.0e-1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*ankle"]),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["l_ankle","r_ankle"]),
    #     }
    # )

    # penalize joint torque
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*Shoulder.*",".*Elbow.*",".*WristYaw",".*Hip.*",".*Knee.*"])
        },
        weight=-8.0e-4)
    ankle_torque_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*Ankle.*"])
        },
        weight=-2.0e-3)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*Shoulder.*",".*Elbow.*",".*WristYaw",".*Hip.*",".*Knee.*",".*Ankle.*"])
        },
        weight=-2.5e-7)
    dof_action_rate_l2= RewTerm(
        func=mdp.action_rate_l2,
        weight=-2.0e-2)
    dof_action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-6.5e-4)
    # torso flat
    torso_flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.2,
    )
    # undesired contact
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 10.0,
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*wrist", ".*Thigh", "base_link"])
        }
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": DEG(60)}
    )


@configclass
class NAOPoseRandEnvCfg(ManagerBasedRLEnvCfg):
    
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionCfg = ActionCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 3
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.004
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        self.scene.robot = NAO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        

class NAOPoseRandEnvCfg_PLAY(NAOPoseRandEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 10.0

        self.events.add_base_mass = None
        self.events.physics_material = None
        self.events.push_robot = None

        self.observations.policy.pose_commands.params = {"command_name": "sequence_pose"}