from __future__ import annotations

import torch
import os
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import MotionRandomCommandCfg

class MotionRandomCommand(CommandTerm):
    
    cfg: MotionRandomCommandCfg

    def __init__(self, cfg: MotionRandomCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._joint_ids,self._joint_names = self.robot.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        self.pos_command_b = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        # (b,n) b: concat all npy, n: number of joints
        self._motion_data = self.load_motion(self.cfg.motion_path)
        self._len = self._motion_data.shape[0]
        # (b,nc) nc: number of commanded joints
        self._motion_commands = self._motion_data[:,self._joint_ids]
        # random id for each env
        self._idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        print("\n\n\n\n======== MotionRandomCommand ========")
        print(self._motion_data.shape)
        print("\n\n\n\n")
        

    def load_motion(self, motion_dir: str) -> torch.Tensor:
        motion_sequence = []
        for file_name in sorted(os.listdir(motion_dir)):
            if file_name.endswith(".npy"):
                motion = np.load(os.path.join(motion_dir, file_name))
                motion_sequence.append(motion)
        motion_sequence = np.concatenate(motion_sequence, axis=0)
        motion_sequence = torch.tensor(motion_sequence, dtype=torch.float32, device=self.device)
        return motion_sequence
    
    def __str__(self) -> str:
        msg = "MotionRandomCommand:\n"
        msg += f"\tLoad motion from: {self.cfg.motion_path}\n"
        msg += f"\tJoint names: {self._joint_names}\n"
        msg += f"\tJoint ids: {self._joint_ids}\n"
        msg += f"\tCommand shape: {self.command.shape}\n"
        msg += f"\tMotion shape: {self._motion_commands.shape}\n"
        return msg
        
    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return self.pos_command_b
    
    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the joint position error
        self.metrics["pos_err"] = torch.mean(torch.abs(self.pos_command_b - self.robot.data.joint_pos[:,self._joint_ids]), dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        self._idx[env_ids] = torch.randint(0, self._len, size=(len(env_ids),), device=self.device, dtype=torch.int32)
        self.pos_command_b[env_ids] = self._motion_commands[self._idx[env_ids]]

    def _update_command(self):
        pass