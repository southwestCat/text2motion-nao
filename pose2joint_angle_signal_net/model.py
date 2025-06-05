import torch
import torch.nn as nn
from robot import NAO_JOINT_LIMITS
from myutils import *

INPUT_DIM = 4*7
OUTPUT_DIM = 5*2+1+5*2

# Angle Signal Net Base
class InverseKinematicsNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=[128,128]):
        super(InverseKinematicsNN, self).__init__()
        layers = []
        prev_units = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(prev_units, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            prev_units = units
        layers.append(nn.Linear(prev_units, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Joint Limits
        self.register_buffer("joint_limits", torch.tensor(NAO_JOINT_LIMITS, dtype=torch.float32))

    def forward(self, x):
        joint_limits = self.joint_limits.to(x.device)
        q_min, q_max = torch.unbind(joint_limits, dim=1)

        raw_output = self.network(x)
        raw_sigmoid = torch.sigmoid(raw_output) 
        output = raw_sigmoid * (q_max - q_min) + q_min
        return output