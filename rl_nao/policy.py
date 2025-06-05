
import math


def DEG(x):
    return x*math.pi/180.0


POLICY_INPUT_DIM = 64
POLICY_OUTPUT_DIM = 21


policy_default_joint_pos = {
	"HeadYaw":0.0,"HeadPitch":0.0,
    "LShoulderPitch":DEG(90),"LShoulderRoll":DEG(10),"LElbowYaw":0.0,"LElbowRoll":-0.035,"LWristYaw":DEG(-90),
    "RShoulderPitch":DEG(90),"RShoulderRoll":DEG(-10),"RElbowYaw":0.0,"RElbowRoll":0.035,"RWristYaw":DEG(90),
    "LHipYawPitch":0.0,"LHipRoll":0.0,"LHipPitch":-0.433096,"LKneePitch":0.853201,"LAnklePitch":-0.420104,"LAnkleRoll":0.0,
    "RHipYawPitch":0.0,"RHipRoll":0.0,"RHipPitch":-0.433096,"RKneePitch":0.853201,"RAnklePitch":-0.420104,"RAnkleRoll":0.0}


policy_action_joint_names = ['LHipYawPitch', 'LShoulderPitch', 'RShoulderPitch', 'LHipRoll', 'LShoulderRoll', 'RHipRoll', 'RShoulderRoll', 'LHipPitch', 'LElbowYaw', 'RHipPitch', 'RElbowYaw', 'LKneePitch', 'LElbowRoll', 'RKneePitch', 'RElbowRoll', 'LAnklePitch', 'LWristYaw', 'RAnklePitch', 'RWristYaw', 'LAnkleRoll', 'RAnkleRoll']


policy_action_joint_scale = 0.2


policy_command_joint_names = ['LHipYawPitch', 'LShoulderPitch', 'RHipYawPitch', 'RShoulderPitch', 'LHipRoll', 'LShoulderRoll', 'RHipRoll', 'RShoulderRoll', 'LHipPitch', 'LElbowYaw', 'RHipPitch', 'RElbowYaw', 'LKneePitch', 'LElbowRoll', 'RKneePitch', 'RElbowRoll', 'LWristYaw', 'RWristYaw']


policy_observation_joint_names = ['LHipYawPitch', 'LShoulderPitch', 'RHipYawPitch', 'RShoulderPitch', 'LHipRoll', 'LShoulderRoll', 'RHipRoll', 'RShoulderRoll', 'LHipPitch', 'LElbowYaw', 'RHipPitch', 'RElbowYaw', 'LKneePitch', 'LElbowRoll', 'RKneePitch', 'RElbowRoll', 'LAnklePitch', 'LWristYaw', 'RAnklePitch', 'RWristYaw', 'LAnkleRoll', 'RAnkleRoll']


robot_joint_names = ['HeadYaw', 'LHipYawPitch', 'LShoulderPitch', 'RHipYawPitch', 'RShoulderPitch', 'HeadPitch', 'LHipRoll', 'LShoulderRoll', 'RHipRoll', 'RShoulderRoll', 'LHipPitch', 'LElbowYaw', 'RHipPitch', 'RElbowYaw', 'LKneePitch', 'LElbowRoll', 'RKneePitch', 'RElbowRoll', 'LAnklePitch', 'LWristYaw', 'RAnklePitch', 'RWristYaw', 'LAnkleRoll', 'LHand', 'RAnkleRoll', 'RHand']
