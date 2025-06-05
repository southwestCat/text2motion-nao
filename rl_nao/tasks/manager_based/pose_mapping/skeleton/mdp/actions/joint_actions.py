from __future__ import annotations

from isaaclab.envs.mdp.actions import JointPositionAction
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import JointPositionHipYawActionCfg


class JointPositionHipYawAction(JointPositionAction):
    
    cfg: JointPositionHipYawActionCfg

    def __init__(self, cfg: JointPositionHipYawActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        assert "RHipYawPitch" not in self._joint_names, "Do not use RHipYawPitch explicitly!"
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        if "LHipYawPitch" in self._joint_names:
            self._asset._data.joint_pos_target[:,self._asset.joint_names.index("RHipYawPitch")] = self._asset._data.joint_pos_target[:,self._asset.joint_names.index("LHipYawPitch")]