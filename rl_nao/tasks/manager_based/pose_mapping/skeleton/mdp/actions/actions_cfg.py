from dataclasses import MISSING
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from . import joint_actions

@configclass
class JointPositionHipYawActionCfg(JointPositionActionCfg):
    """Configuration for the joint position action term."""
    class_type: type[ActionTerm] = joint_actions.JointPositionHipYawAction