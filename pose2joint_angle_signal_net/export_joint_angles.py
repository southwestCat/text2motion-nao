import torch
import numpy as np
import os
from model import InverseKinematicsNN, INPUT_DIM, OUTPUT_DIM
from joints import Joints
from iknn import load_inputs

def check_joints_order(order:list) -> bool:
    if len(order) != Joints.NumOfJoints:
        raise ValueError(f"Length of order should be {len(Joints)}")
    for i,name in enumerate(order):
        if name != Joints(i).name:
            raise ValueError(f"{i}, {name} should be {Joints(i).name}")
    return True


original_order = [
    'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw',
    'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', "RWristYaw", 
    'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 
    'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'
]


target_order = ['HeadYaw', 'LHipYawPitch', 'LShoulderPitch', 'RHipYawPitch', 'RShoulderPitch', 'HeadPitch', 'LHipRoll', 'LShoulderRoll', 'RHipRoll', 'RShoulderRoll', 'LHipPitch', 'LElbowYaw', 'RHipPitch', 'RElbowYaw', 'LKneePitch', 'LElbowRoll', 'RKneePitch', 'RElbowRoll', 'LAnklePitch', 'LWristYaw', 'RAnklePitch', 'RWristYaw', 'LAnkleRoll', 'LHand', 'RAnkleRoll', 'RHand']

check_joints_order(original_order)

model = InverseKinematicsNN(INPUT_DIM, OUTPUT_DIM)
print(model)

best_model = "weights/best.pt"
print(f"Load best model: {best_model}")
model.load_state_dict(torch.load(best_model, weights_only=True))
model.to("cuda")
model.eval()

for root,dirs,files in os.walk("../pose2joint_datasets"):
    for file in files:
        if file.endswith(".npy"):
            file_path = os.path.join(root, file)
            inputs = np.load(file_path)
            output_file_path = os.path.join("outputs", file)
            n = inputs.shape[0]
            inputs = inputs.reshape(n, -1)
            with torch.no_grad():
                outputs = model(torch.tensor(inputs, dtype=torch.float32).to("cuda"))
                new_tensor_columns = []
                for joint in target_order:
                    if joint in original_order:
                        new_tensor_columns.append(outputs[:, original_order.index(joint)])
                    elif joint == 'RHipYawPitch':
                        new_tensor_columns.append(outputs[:, original_order.index('LHipYawPitch')])
                    else:
                        new_tensor_columns.append(torch.zeros_like(outputs[:, original_order.index('LHipYawPitch')]))
                new_tensor = torch.stack(new_tensor_columns, dim=1)
            print(f"Save npy in {output_file_path}")
            np.save(output_file_path, new_tensor.cpu().numpy())