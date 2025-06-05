import numpy as np
from nao_interfaces.msg import Joints
from .policy import policy_default_joint_pos
from .motion_utils import DEG


MOTION_LOOP_TIME = 0.012


JointNames = [
    "HeadYaw",
    "HeadPitch",
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "LHand",
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
    "RWristYaw",
    "RHand",
    "LHipYawPitch",
    "LHipRoll",
    "LHipPitch",
    "LKneePitch",
    "LAnklePitch",
    "LAnkleRoll",
    "RHipYawPitch",
    "RHipRoll",
    "RHipPitch",
    "RKneePitch",
    "RAnklePitch",
    "RAnkleRoll",
]

# Default joint offset in JointNames order
def create_default_joint_offset() -> np.ndarray:
    assert len(JointNames) == Joints.NUM_OF_JOINTS
    joint_offset = np.zeros(Joints.NUM_OF_JOINTS, dtype=np.float32)
    for name in policy_default_joint_pos.keys():
        assert name in JointNames, f"Joint name {name} not found in JointNames"
        joint_offset[JointNames.index(name)] = policy_default_joint_pos[name]
    return joint_offset