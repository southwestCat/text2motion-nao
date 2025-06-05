import torch
from .rotation_convert import *


def walk_mode(lleg:torch.Tensor, rleg:torch.Tensor, threshold:float) -> torch.Tensor:
    dtype = lleg.dtype
    device = lleg.device
    n = lleg.shape[0]

    LHP_ID = 0
    lpelvis = TTrans(torch.tensor([0.0, 50.0, 0.0],dtype=dtype, device=device).repeat(n, 1))@TRotX(torch.tensor(torch.pi/4.0,dtype=dtype,device=device).repeat(n))@TRotZ(-lleg[:, LHP_ID])@TRotX(torch.tensor(-torch.pi/4.0,dtype=dtype,device=device).repeat(n))
    lhip = lpelvis@TRotX(lleg[:, LHP_ID+1])
    lthigh = lhip@TRotY(lleg[:, LHP_ID+2])
    ltibia = lthigh@TTrans(torch.tensor([0.0, 0.0, -100.0],dtype=dtype, device=device).repeat(n, 1))@TRotY(lleg[:, LHP_ID+3])
    lankle = ltibia@TTrans(torch.tensor([0.0, 0.0, -102.9],dtype=dtype, device=device).repeat(n, 1))

    RHP_ID = 0
    rpelvis = TTrans(torch.tensor([0.0, -50.0, 0.0],dtype=dtype, device=device).repeat(n, 1))@TRotX(torch.tensor(-torch.pi/4.0,dtype=dtype,device=device).repeat(n))@TRotZ(rleg[:, RHP_ID])@TRotX(torch.tensor(torch.pi/4.0,dtype=dtype,device=device).repeat(n))
    rhip = rpelvis@TRotX(rleg[:, RHP_ID+1])
    rthigh = rhip@TRotY(rleg[:, RHP_ID+2])
    rtibia = rthigh@TTrans(torch.tensor([0.0, 0.0, -100.0],dtype=dtype, device=device).repeat(n, 1))@TRotY(rleg[:, RHP_ID+3])
    rankle = rtibia@TTrans(torch.tensor([0.0, 0.0, -102.9],dtype=dtype, device=device).repeat(n, 1))

    LFootX = lankle[:,0,3]
    LFootY = lankle[:,1,3]
    LFootZ = lankle[:,2,3]
    RFootX = rankle[:,0,3]
    RFootY = rankle[:,1,3]
    RFootZ = rankle[:,2,3]

    dX = torch.abs(LFootX-RFootX) > threshold
    return dX
