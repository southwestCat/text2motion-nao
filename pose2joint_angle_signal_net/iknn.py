import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

from robot import NAO_JOINT_LIMITS
from fk import fk_norm_tensor, robot_arm_length, robot_leg_length
from rotation_convert import *
from myutils import *

from model import InverseKinematicsNN, INPUT_DIM, OUTPUT_DIM


# load data
def load_dataset(input_data, output_data, batch_size=64, train_split=0.8):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    output_tensor = torch.tensor(output_data, dtype=torch.float32)
    dataset = TensorDataset(input_tensor, output_tensor)
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_inputs(dir_path:str) -> np.ndarray:
    data = list()
    fps = 20.0
    for root,dirs,files in os.walk(dir_path):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                npy = np.load(file_path)
                print(f"Loading {file_path}, frames: {npy.shape[0]}, times: {float(npy.shape[0])/fps:.1f}s.")
                data.append(npy)
    if len(data) == 0:
        raise ValueError(f"No npy files found in {dir_path}.")
    total_data = np.concatenate(data, axis=0)
    print(f"Loaded {total_data.shape[0]} samples.")
    return total_data


def quaternion_loss(predict:torch.Tensor, target:torch.Tensor):
        err = quaternion_multiply(target, quaternion_invert(predict))
        loss = (2.0 * torch.acos(torch.clamp(torch.abs(err[..., 0]), min=-1.0 + 1e-6, max=1.0 - 1e-6))).mean()
        return loss


def NPRLoss(output:torch.Tensor, target:torch.Tensor, trans_weight=1.0, rot_weight=1.0) -> torch.Tensor:
    limb = fk_norm_tensor(output)
    n_limb = limb.shape[0]
    limb_pose = limb.reshape(n_limb, 4, -1)

    predicted_trans = limb_pose[:,:,:3]
    predicted_arm_trans = predicted_trans[:,:2]
    predicted_leg_trans = predicted_trans[:,2:]
    predicted_quat = limb_pose[:,:,3:]

    n_target = target.shape[0]
    target_pose = target.reshape(n_target, 4, -1)
    target_trans = target_pose[:,:,:3]
    target_arm_trans = target_trans[:,:2]
    target_leg_trans = target_trans[:,2:]
    target_quat = target_pose[:,:,3:]

    arm_trans_loss = nn.MSELoss()(predicted_arm_trans, target_arm_trans)
    leg_trans_loss = nn.MSELoss()(predicted_leg_trans, target_leg_trans)
    trans_loss = robot_arm_length()*arm_trans_loss + robot_leg_length()*leg_trans_loss
    rot_loss = quaternion_loss(predicted_quat, target_quat)

    total_loss = trans_weight*trans_loss + rot_weight*rot_loss*180.0/torch.pi
    return total_loss


def save_debug_info(model, inputs, save_dir="debug"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Debug information will be saved in: {save_dir}")
    model_path = os.path.join(save_dir, f"model_debug_{timestamp}.pt")
    input_path = os.path.join(save_dir, f"input_debug_{timestamp}.pt")
    
    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to: {model_path}")
    
    torch.save(inputs, input_path)
    print(f"Input tensor saved to: {input_path}")


# train
def train(model:nn.Module, train_loader:DataLoader, test_loader:DataLoader, epochs=3000, lr=1e-3, device='cuda', save_dir="checkpoints"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # current time, format: YYYYMMDD_HHMMSS
    save_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Models will be saved in: {save_dir}")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    best_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best.pt")
    
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-6:
            print(f"Learning rate ({current_lr}) is below threshold (1e-6). Stopping training.")
            break

        model.train()
        total_loss = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, targets in train_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = NPRLoss(outputs, targets, trans_weight=1.0, rot_weight=1.0)
            loss.backward()
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            total_loss += loss.item()

            train_loop.set_postfix(train_loss=total_loss / len(train_loader))
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0

            test_loop = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=False)
            with torch.no_grad():
                for inputs, targets in test_loop:
                    inputs, targets = inputs.to(device), targets.to(device)
                    loss = NPRLoss(model(inputs), targets)
                    test_loss += loss.item()
                    
                    test_loop.set_postfix(test_loss=test_loss / len(test_loader))
            test_loss = test_loss / len(test_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}, Test Loss: {test_loss:.4f}")

            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(test_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != prev_lr:
                print(f"Learning rate updated: {prev_lr} -> {new_lr}")

            if (epoch + 1) % 100 == 0:
                checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")
            
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with test loss: {best_loss:.4f}")
    print(f"Best model saved {best_model_path} with test loss: {best_loss:.4f}")
    cp_best_cmd = f"cp {best_model_path} weights/best.pt"
    os.system(cp_best_cmd)


def xavier_normal_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def main():
    torch.autograd.set_detect_anomaly(True)
    input_dim = INPUT_DIM
    output_dim = OUTPUT_DIM

    input_data = load_inputs("../pose2joint_datasets")
    input_data = input_data.reshape(-1, input_dim)
    output_data = input_data.copy()

    train_loader, test_loader = load_dataset(input_data, output_data)
    model = InverseKinematicsNN(input_dim, output_dim)
    model.apply(xavier_normal_init)
    train(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
    
