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
import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationData, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from nao import NAO_V50_CFG
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


HOME_PATH=os.getenv('HOME')


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
    robot: ArticulationCfg = NAO_V50_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
    count = 0

    npy_dir_path = HOME_PATH+"/PoseMapping/npy"
    file_name = "001_limb_spec.npy"
    npy_file_path = os.path.join(npy_dir_path, file_name)
    
    motion_frames = np.load(npy_file_path)
    motion_frames_tensor = torch.tensor(motion_frames, device=sim.cfg.device)
    print(motion_frames.shape)
    print(motion_frames_tensor.shape)

    # Simulate physics
    while simulation_app.is_running():
        # Apply default actions to the robot
        robot: Articulation = scene["robot"]
        data: ArticulationData = robot.data
        # print(torch.sum(data.default_mass, dim=1))
        # -- generate actions/commands
        # targets = scene["robot"].data.default_joint_pos
        targets = motion_frames_tensor[count]

        for _ in range(sim.cfg.render_interval):
            # -- apply action to the robot
            scene["robot"].set_joint_position_target(targets)
            # -- write data to sim
            scene.write_data_to_sim()
            # perform step
            sim.step()
            # update sim-time
            sim_time += sim_dt
            # update buffers
            scene.update(sim_dt)
            print(f"Sim time: {sim_time:.3f}", end="\r")
        count += 1
        if count >= motion_frames_tensor.shape[0]:
            print("Done...")
            break
        


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.004, render_interval=3, device=args_cli.device)
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
