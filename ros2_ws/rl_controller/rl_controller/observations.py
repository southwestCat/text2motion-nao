import numpy as np
from nao_interfaces.msg import NaoSensor, Joints

from .madgwickahrs import  MadgwickAHRS
from .motion_config import create_default_joint_offset, JointNames
from .quaternion import quaternion_to_rotation_matrix
from .policy import policy_action_joint_names,policy_observation_joint_names


class Observations:
    # joint_pos_init in Joints order
    joint_pos_init = create_default_joint_offset()
    acc = np.array([0.0]*3, dtype=np.float32)
    gyro = np.array([0.0]*3, dtype=np.float32)
    gravity_projection = np.array([0.0]*3, dtype=np.float32)
    joint_pos_rel = np.zeros_like(joint_pos_init)
    last_actions = np.zeros(len(policy_action_joint_names), dtype=np.float32)

    def __init__(self, dt):
        self.filter_ = MadgwickAHRS(dt)        
    
    def __str__(self):
        obstr = f"Observations:\n"
        obstr += f"\tacc: {self.acc}\n"
        obstr += f"\tgyro: {self.gyro}\n"
        obstr += f"\tgravity_projection: {self.gravity_projection}\n"
        return obstr
    

    def update(self, msg:NaoSensor):
        gyro = msg.gyro
        acc = msg.acc
        joint_pos = msg.position
        self.gravity_projection = self.update_gravity_projection(gyro, acc)
        self.joint_pos_rel = self.update_joint_pos_rel(joint_pos)


    def update_gravity_projection(self, gyro, acc):
        self.filter_.update_imu(gyro, acc)
        quaternion = self.filter_.quaternion.conj().q
        vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        rotation = quaternion_to_rotation_matrix(quaternion)
        return np.dot(rotation, vector).astype(np.float32)
    

    def update_joint_pos_rel(self, joint_pos):
        joint_pos_rel = joint_pos - self.joint_pos_init
        obs_joint_pos_rel = np.zeros(len(policy_observation_joint_names), dtype=np.float32)
        for i, joint_name in enumerate(policy_observation_joint_names):
            obs_joint_pos_rel[i] = joint_pos_rel[JointNames.index(joint_name)]
        return obs_joint_pos_rel
    

    def update_last_actions(self, last_actions:np.ndarray):
        if len(self.last_actions) != len(last_actions):
            raise ValueError(f"last_actions has length {len(last_actions)}, but should have length {len(self.last_actions)}")
        self.last_actions = last_actions.copy()