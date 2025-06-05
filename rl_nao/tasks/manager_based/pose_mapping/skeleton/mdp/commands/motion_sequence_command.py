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
    from .commands_cfg import MotionSequenceCommandCfg

class MotionSequenceCommand(CommandTerm):
    
    cfg: MotionSequenceCommandCfg

    def __init__(self, cfg: MotionSequenceCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._joint_ids,self._joint_names = self.robot.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        self.pos_command_b = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self.pos_command_init = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._init_frame_id = int(cfg.delayed_time/env.step_dt)
        
        _max_frames_per_motion = self._env.max_episode_length

        assert self._init_frame_id < _max_frames_per_motion, "MotionCommand delayed time to large."
        # 加载动作序列，包含所有关节
        # (b,l,n) b: 有多少种动作, l: 动作序列长度固定值, n: 关节数
        # _lens: 每种动作原本的时间长度
        self._motion_sequence_full_joints,self._lens = self.load_motion(self.cfg.motion_sequence_path,_max_frames_per_motion)
        # 只保留需要的关节
        self._motion_sequence = self._motion_sequence_full_joints[:,:,self._joint_ids]
        # 动作种类数量
        self._num_motions = self._motion_sequence.shape[0]
        # 每个机器人的动作帧序号
        self._frame_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # 每个机器人的动作长度
        self._frame_lens = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # 每个机器人的动作种类
        self._motion_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def load_motion(self, motion_dir: str, frames_per_motion: int) -> tuple[torch.Tensor, torch.Tensor]:
        motion_sequence = []
        frame_lens = []

        for file_name in sorted(os.listdir(motion_dir)):  # 确保遍历顺序一致
            if file_name.endswith(".npy"):
                motion = np.load(os.path.join(motion_dir, file_name))  # 读取文件
                frame_len = min(motion.shape[0], frames_per_motion)  # 记录真实帧数
                
                # 处理超长或不足的情况
                if motion.shape[0] >= frames_per_motion:
                    motion = motion[:frames_per_motion]  # 截断
                else:
                    pad_size = frames_per_motion - motion.shape[0]
                    motion = np.pad(motion, ((0, pad_size), (0, 0)), mode="edge")

                motion_sequence.append(motion)
                frame_lens.append(frame_len)

        # 转换为 PyTorch Tensor，并移动到 self.device
        motion_sequence = np.array(motion_sequence, dtype=np.float32)
        motion_sequence = torch.tensor(motion_sequence, device=self.device)
        frame_lens = torch.tensor(frame_lens, dtype=torch.int32, device=self.device)

        return motion_sequence, frame_lens
    
    def __str__(self) -> str:
        msg = "MotionSequenceCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\t Num of motions: {self._num_motions}\n"
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
        self._motion_idx[env_ids] = torch.randint(0, self._num_motions, size=(len(env_ids),), device=self.device,dtype=torch.int32)
        self._frame_idx[env_ids] = 0
        self._frame_lens[env_ids] = self._lens[self._motion_idx[env_ids]]
        self.pos_command_b[env_ids] = self._motion_sequence[self._motion_idx[env_ids], self._frame_idx[env_ids]]
        self.pos_command_init[env_ids] = self._motion_sequence[self._motion_idx[env_ids], self._frame_idx[env_ids]]

    def _update_command(self):
        self.pos_command_b = self._motion_sequence[self._motion_idx, self._frame_idx]
        self.pos_command_init = self._motion_sequence[self._motion_idx, self._init_frame_id]
        self._frame_idx += 1
        self._frame_idx = torch.clip(self._frame_idx, max=self._lens[self._motion_idx])