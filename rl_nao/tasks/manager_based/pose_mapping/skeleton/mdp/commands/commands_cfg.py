import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .motion_sequence_command import MotionSequenceCommand
from .motion_random_command import MotionRandomCommand


@configclass
class MotionSequenceCommandCfg(CommandTermCfg):
    
    class_type: type = MotionSequenceCommand
    asset_name: str = MISSING
    joint_names: str | list[str] = MISSING
    motion_sequence_path: str = MISSING
    delayed_time: float = MISSING

@configclass
class MotionRandomCommandCfg(CommandTermCfg):

    class_type: type = MotionRandomCommand
    asset_name: str = MISSING
    joint_names: str | list[str] = MISSING
    motion_path: str = MISSING