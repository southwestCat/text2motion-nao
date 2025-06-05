# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from .commands import MotionSequenceCommand

def motion_frame_finish(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Check if the current motion frame is the last frame."""
    command_term: MotionSequenceCommand = env.command_manager.get_term(command_name)
    return command_term._frame_idx >= command_term._frame_lens


def no_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor falls below the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # no_contact = torch.all(air_time > threshold, dim=1)
    # check if any contact force exceeds the threshold
    return torch.all(air_time > threshold, dim=1)