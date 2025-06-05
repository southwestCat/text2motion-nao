import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationData, ArticulationCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from assets import NAO_CFG # isort: skip
from my_utils import DEG # isort: skip

import re
import ast

policy_default_joint_pos = {
    "HeadYaw":0.0,"HeadPitch":0.0,
    "LShoulderPitch":DEG(90),"LShoulderRoll":DEG(10),"LElbowYaw":0.0,"LElbowRoll":-0.035,"LWristYaw":DEG(-90),
    "RShoulderPitch":DEG(90),"RShoulderRoll":DEG(-10),"RElbowYaw":0.0,"RElbowRoll":0.035,"RWristYaw":DEG(90),
    "LHipYawPitch":0.0,"LHipRoll":0.0,"LHipPitch":-0.433096,"LKneePitch":0.853201,"LAnklePitch":-0.420104,"LAnkleRoll":0.0,
    "RHipYawPitch":0.0,"RHipRoll":0.0,"RHipPitch":-0.433096,"RKneePitch":0.853201,"RAnklePitch":-0.420104,"RAnkleRoll":0.0}

DEG_str = \
'''
import math


def DEG(x):
    return x*math.pi/180.0
'''

@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = NAO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0, 
        history_length=6, debug_vis=True
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    robot: Articulation = scene["robot"]
    robot_joint_names = robot.joint_names

    # CommandsCfg
    CommandsCfg = {
        "joint_names":[".*Shoulder.*", ".*Elbow.*", ".*WristYaw", ".*Hip.*", ".*Knee.*"]
    }
    _, c_names = robot.find_joints(CommandsCfg["joint_names"])
    command_joint_names = c_names

    # ActionCfg
    ActionCfg = {
        "joint_names":["LHipYawPitch",".*HipRoll",".*HipPitch",".*Knee.*",".*Ankle.*",".*Shoulder.*",".*Elbow.*",".*WristYaw"],
        "scale":0.2
    }
    
    _, a_names = robot.find_joints(ActionCfg["joint_names"])
    action_joint_names = a_names

    action_joint_scale = ActionCfg["scale"]
    # action_joint_scale = dict()
    # for scale_name in action_scale.keys():
    #     _id, _names = robot.find_joints(scale_name)
    #     for _name in _names:
    #         action_joint_scale[_name] = action_scale[scale_name]

    # ObservationsCfg
    ObservationsCfg = {
        "joint_names": [".*Shoulder.*",".*Elbow.*",".*Wrist.*",".*Hip.*",".*Knee.*",".*Ankle.*"]
    }
    _, o_names = robot.find_joints(ObservationsCfg["joint_names"])
    observation_joint_names = o_names


    with open("policy.py", 'w', encoding='utf-8') as file:
        file.write(DEG_str)
        file.write("\n\n")
        
        file.write("POLICY_INPUT_DIM = \n")
        file.write("POLICY_OUTPUT_DIM = \n")
        file.write("\n\n")


        file.write("policy_default_joint_pos = \n")
        file.write("\n\n")

        file.write(f"policy_action_joint_names = {action_joint_names}\n")
        file.write("\n\n")
        
        file.write(f"policy_action_joint_scale = {action_joint_scale}\n")
        file.write("\n\n")

        file.write(f"policy_command_joint_names = {command_joint_names}\n")
        file.write("\n\n")

        file.write(f"policy_observation_joint_names = {observation_joint_names}\n")
        file.write("\n\n")

        file.write(f"robot_joint_names = {robot_joint_names}\n")



def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()