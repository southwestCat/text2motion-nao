from enum import IntEnum

joint_names = ['HeadYaw', 'LHipYawPitch', 'LShoulderPitch', 'RHipYawPitch', 'RShoulderPitch', 'HeadPitch', 'LHipRoll', 'LShoulderRoll', 'RHipRoll', 'RShoulderRoll', 'LHipPitch', 'LElbowYaw', 'RHipPitch', 'RElbowYaw', 'LKneePitch', 'LElbowRoll', 'RKneePitch', 'RElbowRoll', 'LAnklePitch', 'LWristYaw', 'RAnklePitch', 'RWristYaw', 'LAnkleRoll', 'RAnkleRoll']


def jointId(name):
    assert name in joint_names, "Unknown joint name: "+name
    return joint_names.index(name)


class Joints(IntEnum):
    HeadYaw = jointId("HeadYaw")
    HeadPitch = jointId("HeadPitch")
    LShoulderPitch = jointId("LShoulderPitch")
    LShoulderRoll = jointId("LShoulderRoll")
    LElbowYaw = jointId("LElbowYaw")
    LElbowRoll = jointId("LElbowRoll")
    LWristYaw = jointId("LWristYaw")
    RShoulderPitch = jointId("RShoulderPitch")
    RShoulderRoll = jointId("RShoulderRoll")
    RElbowYaw = jointId("RElbowYaw")
    RElbowRoll = jointId("RElbowRoll")
    RWristYaw = jointId("RWristYaw")
    LHipYawPitch = jointId("LHipYawPitch")
    LHipRoll = jointId("LHipRoll")
    LHipPitch = jointId("LHipPitch")
    LKneePitch = jointId("LKneePitch")
    LAnklePitch = jointId("LAnklePitch")
    LAnkleRoll = jointId("LAnkleRoll")
    RHipYawPitch = jointId("RHipYawPitch")
    RHipRoll = jointId("RHipRoll")
    RHipPitch = jointId("RHipPitch")
    RKneePitch = jointId("RKneePitch")
    RAnklePitch = jointId("RAnklePitch")
    RAnkleRoll = jointId("RAnkleRoll")
