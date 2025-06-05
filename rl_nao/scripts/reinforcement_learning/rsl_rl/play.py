# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


import math
def DEG(x):
    return x*math.pi/180.0
policy_default_joint_pos = {
    "HeadYaw":0.0,"HeadPitch":0.0,
    "LShoulderPitch":DEG(90),"LShoulderRoll":DEG(10),"LElbowYaw":0.0,"LElbowRoll":-0.035,"LWristYaw":DEG(-90),
    "RShoulderPitch":DEG(90),"RShoulderRoll":DEG(-10),"RElbowYaw":0.0,"RElbowRoll":0.035,"RWristYaw":DEG(90),
    "LHipYawPitch":0.0,"LHipRoll":0.0,"LHipPitch":-0.433096,"LKneePitch":0.853201,"LAnklePitch":-0.420104,"LAnkleRoll":0.0,
    "RHipYawPitch":0.0,"RHipRoll":0.0,"RHipPitch":-0.433096,"RKneePitch":0.853201,"RAnklePitch":-0.420104,"RAnkleRoll":0.0}


policy_action_joint_names = ['LHipYawPitch', 'LShoulderPitch', 'RShoulderPitch', 'LHipRoll', 'LShoulderRoll', 'RHipRoll', 'RShoulderRoll', 'LHipPitch', 'LElbowYaw', 'RHipPitch', 'RElbowYaw', 'LKneePitch', 'LElbowRoll', 'RKneePitch', 'RElbowRoll', 'LAnklePitch', 'LWristYaw', 'RAnklePitch', 'RWristYaw', 'LAnkleRoll', 'RAnkleRoll']


policy_action_joint_scale = 0.2
LOG_INFO = False
LOG_LENGTH = 730


def process_actions(raw_actions:torch.Tensor):
    actions = torch.zeros_like(raw_actions)
    for i,name in enumerate(policy_action_joint_names):
        actions[i] = raw_actions[i]*policy_action_joint_scale+policy_default_joint_pos[name]
    return actions
    ...


import csv
actions_list = list()
log_written = False
joint_pos_list = list()
def apply_actions(actions:torch.Tensor,joint_pos):
    global actions_list
    global log_written
    actions_list.append(actions.tolist())
    joint_pos_list.append(joint_pos.tolist())
    if len(actions_list)>=LOG_LENGTH and not log_written:
        with open("actions.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(actions_list)
        with open("jointpos.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(joint_pos_list)
        print("write log")
        log_written = True


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    sim_time = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # custom log info
            if LOG_INFO:
                processed_actions = process_actions(actions[0])
                joint_pos = env.unwrapped.scene['robot'].data.joint_pos[0]
                apply_actions(processed_actions,joint_pos)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        # show current time
        sim_time += dt
        print(f"Sim time: {sim_time:.3f}", end="\r")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
