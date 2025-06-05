'''
Calculate robot forward kinematics.
'''

import torch
import numpy as np
from rotation import DEG
from joints import Joints
from rotation_convert import *


def robot_arm_length() -> float:
    '''
    Robot arm length. [mm]
    '''
    larm_base = np.array([0.0,98.0,185.0])
    lhand = np.array([218.7,113.0,197.31])
    larm = lhand-larm_base
    return np.linalg.norm(larm)


def robot_leg_length() -> float:
    '''
    Robot leg length. [mm]
    '''
    return 100.0+102.9+45.19


def fk_norm_tensor(theta:torch.tensor) -> torch.Tensor:
    if not (isinstance(theta, torch.Tensor) and theta.shape[-1] == Joints.NumOfJoints):
        raise ValueError(f'theta must be a torch tensor with shape (, {Joints.NumOfJoints})')
    
    LSD_ID = Joints.LShoulderPitch
    RSD_ID = Joints.RShoulderPitch

    dtype = theta.dtype
    device = theta.device
    
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
    n = theta.shape[0]

    larm_base = TTrans(torch.tensor([0.0, 98.0, 185.0],dtype=dtype, device=device).repeat(n, 1))
    lshoulder = TTrans(torch.tensor([0.0, 98.0, 185.0],dtype=dtype, device=device).repeat(n, 1))@TRotY(theta[:, LSD_ID])
    lbiceps = lshoulder@TRotZ(theta[:, LSD_ID+1])
    lelbow = lbiceps@TTrans(torch.tensor([105.0, 15.0, 0.0],dtype=dtype, device=device).repeat(n, 1))@TRotX(theta[:, LSD_ID+2])
    lforeArm = lelbow@TRotZ(theta[:, LSD_ID+3])
    lwrist = lforeArm@TTrans(torch.tensor([55.95, 0.0, 0.0],dtype=dtype, device=device).repeat(n, 1))@TRotX(theta[:, LSD_ID+4])
    lhand = lwrist@TTrans(torch.tensor([57.75, 0.0, 12.31],dtype=dtype, device=device).repeat(n, 1))
    larm = lhand[:,:3,3]-larm_base[:,:3,3]
    larm_norm = larm / robot_arm_length()

    rarm_base = TTrans(torch.tensor([0.0, -98.0, 185.0],dtype=dtype, device=device).repeat(n, 1))
    rshoulder = TTrans(torch.tensor([0.0, -98.0, 185.0],dtype=dtype, device=device).repeat(n, 1))@TRotY(theta[:, RSD_ID])
    rbiceps = rshoulder@TRotZ(theta[:, RSD_ID+1])
    relbow = rbiceps@TTrans(torch.tensor([105.0, -15.0, 0.0],dtype=dtype, device=device).repeat(n, 1))@TRotX(theta[:, RSD_ID+2])
    rforeArm = relbow@TRotZ(theta[:, RSD_ID+3])
    rwrist = rforeArm@TTrans(torch.tensor([55.95, 0.0, 0.0],dtype=dtype, device=device).repeat(n, 1))@TRotX(theta[:, RSD_ID+4])
    rhand = rwrist@TTrans(torch.tensor([57.75, 0.0, 12.31],dtype=dtype, device=device).repeat(n, 1))
    rarm = rhand[:,:3,3]-rarm_base[:,:3,3]
    rarm_norm = rarm / robot_arm_length()


    LHP_ID = Joints.LHipYawPitch
    lleg_base = TTrans(torch.tensor([0.0, 50.0, 0.0],dtype=dtype, device=device).repeat(n, 1))
    lpelvis = TTrans(torch.tensor([0.0, 50.0, 0.0],dtype=dtype, device=device).repeat(n, 1))@TRotX(torch.tensor(torch.pi/4.0,dtype=dtype,device=device).repeat(n))@TRotZ(-theta[:, LHP_ID])@TRotX(torch.tensor(-torch.pi/4.0,dtype=dtype,device=device).repeat(n))
    lhip = lpelvis@TRotX(theta[:, LHP_ID+1])
    lthigh = lhip@TRotY(theta[:, LHP_ID+2])
    ltibia = lthigh@TTrans(torch.tensor([0.0, 0.0, -100.0],dtype=dtype, device=device).repeat(n, 1))@TRotY(theta[:, LHP_ID+3])
    lankle = ltibia@TTrans(torch.tensor([0.0, 0.0, -102.9],dtype=dtype, device=device).repeat(n, 1))@TRotY(theta[:, LHP_ID+4])
    lfoot = lankle@TRotX(theta[:, LHP_ID+5])
    lsole = lfoot@TTrans(torch.tensor([0.0, 0.0, -45.19],dtype=dtype, device=device).repeat(n, 1))
    lleg = lsole[:,:3,3]-lleg_base[:,:3,3]
    lleg_norm = lleg / robot_leg_length()

    # no RHipYawPitch
    RHP_ID = Joints.LAnkleRoll
    rleg_base = TTrans(torch.tensor([0.0, -50.0, 0.0],dtype=dtype, device=device).repeat(n, 1))
    rpelvis = TTrans(torch.tensor([0.0, -50.0, 0.0],dtype=dtype, device=device).repeat(n, 1))@TRotX(torch.tensor(-torch.pi/4.0,dtype=dtype,device=device).repeat(n))@TRotZ(theta[:, LHP_ID])@TRotX(torch.tensor(torch.pi/4.0,dtype=dtype,device=device).repeat(n))
    rhip = rpelvis@TRotX(theta[:, RHP_ID+1])
    rthigh = rhip@TRotY(theta[:, RHP_ID+2])
    rtibia = rthigh@TTrans(torch.tensor([0.0, 0.0, -100.0],dtype=dtype, device=device).repeat(n, 1))@TRotY(theta[:, RHP_ID+3])
    rankle = rtibia@TTrans(torch.tensor([0.0, 0.0, -102.9],dtype=dtype, device=device).repeat(n, 1))@TRotY(theta[:, RHP_ID+4])
    rfoot = rankle@TRotX(theta[:, RHP_ID+5])
    rsole = rfoot@TTrans(torch.tensor([0.0, 0.0, -45.19],dtype=dtype, device=device).repeat(n, 1))
    rleg = rsole[:,:3,3]-rleg_base[:,:3,3]
    rleg_norm = rleg / robot_leg_length()
    
    result = torch.zeros((n, 4, 7), dtype=dtype, device=device)
    result[:,0,:3] = larm_norm
    result[:,0,3:] = matrix_to_quaternion(lhand[:,:3,:3])
    result[:,1,:3] = rarm_norm
    result[:,1,3:] = matrix_to_quaternion(rhand[:,:3,:3])
    result[:,2,:3] = lleg_norm
    result[:,2,3:] = matrix_to_quaternion(lsole[:,:3,:3])
    result[:,3,:3] = rleg_norm
    result[:,3,3:] = matrix_to_quaternion(rsole[:,:3,:3])
    
    return result.reshape(n,-1)
