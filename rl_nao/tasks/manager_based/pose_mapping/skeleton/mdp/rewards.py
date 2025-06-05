# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
import numpy as np
import csv
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, ArticulationData
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from tasks.manager_based.pose_mapping.skeleton.mdp import MotionSequenceCommand
from my_utils.rotation_convert import (
    homogeneous_multiply,
    homogeneous_inverse,
    quaternion_to_axis_angle
)
from my_utils.fk import walk_mode
from my_utils import DEG

def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.01
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_leg_move(env:ManagerBasedRLEnv, command_name: str, delayed_time:float, threshold: float, leg_move_threshold: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero. Commands small means that the change in the robot's leg joint positions relative to the initial position is small.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    command_term: MotionSequenceCommand = env.command_manager.get_term(command_name)
    cfg_joint_ids = asset_cfg.joint_ids
    joint_ids = command_term._joint_ids
    indices = [joint_ids.index(_id) for _id in cfg_joint_ids]
    command_b = command_term.pos_command_b[:,indices]
    command_init = command_term.pos_command_init[:,indices]
    reward *= torch.norm(command_b-command_init, dim=1) > leg_move_threshold
    reward *= (command_term._frame_idx*env.step_dt > delayed_time)
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_flat_orientation_l2(env: ManagerBasedRLEnv, std:float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    w_ankle_state = asset.data.body_state_w[:,asset_cfg.body_ids]
    w_ankle_quat = w_ankle_state[:,:,3:7]
    w_ankle_axis = quaternion_to_axis_angle(w_ankle_quat)
    w_ankle_axis_xy_l2 = torch.exp(-torch.sum(torch.square(w_ankle_axis[:,:,:2]),dim=-1) / std**2)
    return torch.sum(w_ankle_axis_xy_l2, dim=-1)


def track_joint_targets_exp(env:ManagerBasedRLEnv, std:float, command_name:str, asset_cfg: SceneEntityCfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: MotionSequenceCommand = env.command_manager.get_term(command_name)
    commands: torch.Tensor = env.command_manager.get_command(command_name)
    command_joint_ids = command_term._joint_ids
    cfg_joint_ids = asset_cfg.joint_ids
    cfg_in_command_ids = [command_joint_ids.index(_id) for _id in cfg_joint_ids]
    asset_data = asset.data
    joint_pos = asset_data.joint_pos

    pos_err = torch.sum(
        torch.abs(commands[:,cfg_in_command_ids]-joint_pos[:,cfg_joint_ids]),
        dim=1
    )
    return torch.exp(-pos_err/std)


def track_joint_command_targets_exp(env:ManagerBasedRLEnv, std:float, command_name:str, asset_cfg: SceneEntityCfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: MotionSequenceCommand = env.command_manager.get_term(command_name)
    joint_command: torch.Tensor = env.command_manager.get_command(command_name)
    command_joint_ids = command_term._joint_ids
    joint_pos = asset.data.joint_pos[:,command_joint_ids]
    joint_err = torch.sum(
        torch.abs(joint_command-joint_pos),
        dim=1
    )
    return torch.exp(-joint_err/std)
    


def track_joint_targets_max_exp(env:ManagerBasedRLEnv, std:float, command_name:str,  asset_cfg: SceneEntityCfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: MotionSequenceCommand = env.command_manager.get_term(command_name)
    commands: torch.Tensor = env.command_manager.get_command(command_name)
    command_joint_ids = command_term._joint_ids
    cfg_joint_ids = asset_cfg.joint_ids
    cfg_in_command_ids = [command_joint_ids.index(_id) for _id in cfg_joint_ids]
    asset_data = asset.data
    joint_pos = asset_data.joint_pos

    pos_err,indice = torch.max(
        torch.abs(commands[:,cfg_in_command_ids]-joint_pos[:,cfg_joint_ids]),
        dim=1
    )
    return torch.exp(-pos_err/std)


def feet_air_time_walk(env:ManagerBasedRLEnv, command_name:str, threshold:float, walk_mode_threshold:float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: MotionSequenceCommand = env.command_manager.get_term(command_name)
    commands: torch.Tensor = env.command_manager.get_command(command_name)
    command_joint_ids = command_term._joint_ids
    cfg_joint_ids = asset_cfg.joint_ids

    feet_air_reward = feet_air_time_positive_biped(env, command_name, threshold, sensor_cfg)

    lleg_joints_names = ["LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch"]
    rleg_joints_names = ["RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch"]
    joint_names = asset.joint_names
    lleg_joint_ids = [
        joint_names.index(_name) for _name in lleg_joints_names
    ]
    rleg_joint_ids = [
        joint_names.index(_name) for _name in rleg_joints_names
    ]
    leg_joint_ids = lleg_joint_ids + rleg_joint_ids
    for _id in cfg_joint_ids:
        if _id  not in leg_joint_ids:
            assert False, f"joint {_id} is not in leg_joint_ids"
        if _id not in command_joint_ids:
            assert False, f"joint {_id} is not in command_joint_ids"
    lleg_in_command_ids = [
        command_joint_ids.index(_id) for _id in lleg_joint_ids
    ]
    rleg_in_command_ids = [
        command_joint_ids.index(_id) for _id in rleg_joint_ids
    ]
    lleg_command = commands[:,lleg_in_command_ids]
    rleg_command = commands[:,rleg_in_command_ids]
    
    in_walk = walk_mode(lleg_command,rleg_command, walk_mode_threshold)
    feet_air_weight = torch.where(in_walk, 1.0, 0.0)

    reward = feet_air_weight*feet_air_reward    
    return reward


def hybrid_joint_feet_reward(env:ManagerBasedRLEnv, std:float, command_name:str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: MotionSequenceCommand = env.command_manager.get_term(command_name)
    commands: torch.Tensor = env.command_manager.get_command(command_name)
    command_joint_ids = command_term._joint_ids
    cfg_joint_ids = asset_cfg.joint_ids
    cfg_in_command_ids = [command_joint_ids.index(_id) for _id in cfg_joint_ids]

    pos_track_reward = track_joint_targets_exp(env, std, command_name, asset_cfg)
    feet_air_reward = feet_air_time_leg_move(env, command_name, 1.0, DEG(5), sensor_cfg, asset_cfg)

    lleg_joints_names = ["LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch"]
    rleg_joints_names = ["RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch"]
    leg_joint_names = lleg_joints_names + rleg_joints_names
    joint_names = asset.joint_names
    lleg_joint_ids = [
        joint_names.index(_name) for _name in lleg_joints_names
    ]
    rleg_joint_ids = [
        joint_names.index(_name) for _name in rleg_joints_names
    ]
    leg_joint_ids = lleg_joint_ids + rleg_joint_ids
    for _id in cfg_joint_ids:
        if _id  not in leg_joint_ids:
            assert False, f"joint {_id} is not in leg_joint_ids"
    lleg_in_command_ids = [
        command_joint_ids.index(_id) for _id in lleg_joint_ids
    ]
    rleg_in_command_ids = [
        command_joint_ids.index(_id) for _id in rleg_joint_ids
    ]
    lleg_command = commands[:,lleg_in_command_ids]
    rleg_command = commands[:,rleg_in_command_ids]
    
    in_walk = walk_mode(lleg_command,rleg_command, 40.0)
    feet_air_weight = torch.where(in_walk, 0.2, 0.0)
    pos_track_weight = torch.where(in_walk, 0.2, 1.0)

    reward = pos_track_weight*pos_track_reward + feet_air_weight*feet_air_reward    
    return reward
    # return torch.zeros_like(pos_track_reward)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward
    


def monitor_max_err_joint(env:ManagerBasedRLEnv, std:float, command_name:str, asset_cfg: SceneEntityCfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: MotionSequenceCommand = env.command_manager.get_term(command_name)
    commands: torch.Tensor = env.command_manager.get_command(command_name)
    command_joint_ids = command_term._joint_ids
    cfg_joint_ids = asset_cfg.joint_ids
    joint_names = asset.joint_names
    cfg_joint_names = [joint_names[_id] for _id in cfg_joint_ids]
    cfg_in_command_ids = [command_joint_ids.index(_id) for _id in cfg_joint_ids]
    asset_data = asset.data
    joint_pos = asset_data.joint_pos

    err_max,indice = torch.max(
        torch.abs(commands[:,cfg_in_command_ids]-joint_pos[:,cfg_joint_ids]),
        dim=1
    )

    unique_values, counts = torch.unique(indice, return_counts=True)

    max_count_index = torch.argmax(counts)
    most_frequent_value = unique_values[max_count_index]
    most_frequent_count = counts[max_count_index]
    max_indice = torch.where(indice==most_frequent_value)[0]
    print(f"id: {cfg_joint_names[most_frequent_value.item()]}, pos_err: {torch.mean(err_max[max_indice])}\n")
    return torch.zeros_like(err_max)