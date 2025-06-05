import numpy as np

def DEG(deg):
    return deg*np.pi/180.0

DEFAULT_JOINT_NAMES = ['HeadYaw', 'LHipYawPitch', 'LShoulderPitch', 'RHipYawPitch', 'RShoulderPitch', 'HeadPitch', 'LHipRoll', 'LShoulderRoll', 'RHipRoll', 'RShoulderRoll', 'LHipPitch', 'LElbowYaw', 'RHipPitch', 'RElbowYaw', 'LKneePitch', 'LElbowRoll', 'RKneePitch', 'RElbowRoll', 'LAnklePitch', 'LWristYaw', 'RAnklePitch', 'RWristYaw', 'LAnkleRoll', 'LHand', 'RAnkleRoll', 'RHand']

DEFAULT_JOINT_OFFSET = {
    "HeadYaw":0.0,"HeadPitch":0.0,
    "LShoulderPitch":DEG(90),"LShoulderRoll":DEG(10),"LElbowYaw":0.0,"LElbowRoll":-0.035,"LWristYaw":DEG(-90),
    "RShoulderPitch":DEG(90),"RShoulderRoll":DEG(-10),"RElbowYaw":0.0,"RElbowRoll":0.035,"RWristYaw":DEG(90),
    "LHipYawPitch":0.0,"LHipRoll":0.0,"LHipPitch":-0.433096,"LKneePitch":0.853201,"LAnklePitch":-0.420104,"LAnkleRoll":0.0,
    "RHipYawPitch":0.0,"RHipRoll":0.0,"RHipPitch":-0.433096,"RKneePitch":0.853201,"RAnklePitch":-0.420104,"RAnkleRoll":0.0}

OUTPUTS_JOINT_NAMES = [
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
    "RWristYaw",
    "LHipYawPitch",
    "LHipRoll",
    "LHipPitch",
    "LKneePitch",
    "LAnklePitch",
    "LAnkleRoll",
    "RHipRoll",
    "RHipPitch",
    "RKneePitch",
    "RAnklePitch",
    "RAnkleRoll"
]