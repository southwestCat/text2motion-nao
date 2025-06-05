from enum import IntEnum, auto

class Joints(IntEnum):
    def _generate_next_value_(name, start, count, last_values):
        return count

    LShoulderPitch = auto()
    LShoulderRoll = auto()
    LElbowYaw = auto()
    LElbowRoll = auto()
    LWristYaw = auto()
    RShoulderPitch = auto()
    RShoulderRoll = auto()
    RElbowYaw = auto()
    RElbowRoll = auto()
    RWristYaw = auto()
    LHipYawPitch = auto()
    LHipRoll = auto()
    LHipPitch = auto()
    LKneePitch = auto()
    LAnklePitch = auto()
    LAnkleRoll = auto()
    RHipRoll = auto()
    RHipPitch = auto()
    RKneePitch = auto()
    RAnklePitch = auto()
    RAnkleRoll = auto()
    NumOfJoints = auto()