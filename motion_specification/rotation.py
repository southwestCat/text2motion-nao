import torch
import numpy as np
import rotation_convert as rc


def DEG(x):
    return x * np.pi / 180.0

def RAD(x):
    return x * 180.0 / np.pi

def rotation_matrix_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def rotation_matrix_from_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    k = np.cross(v1, v2)
    k_norm = np.linalg.norm(k)
    theta = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    if k_norm < 1e-2:
        return np.eye(3)
    k = k / k_norm
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    return R

def matrix_to_quaternion(matrix:np.ndarray):
    if not (isinstance(matrix, np.ndarray) and matrix.shape == (3, 3)):
        raise ValueError("input must be 3x3 rotation matrix")
    
    return rc.matrix_to_quaternion(torch.tensor(matrix,dtype=torch.float32,device="cpu")).numpy()


def quaternion_invert(q:np.ndarray) -> np.ndarray:
    if not (isinstance(q, np.ndarray) and q.shape == (4,)):
        raise ValueError("input must be (4,) quaternion")

    return rc.quaternion_invert(torch.tensor(q,dtype=torch.float32,device="cpu")).numpy()


def quaternion_multiply(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    if not (isinstance(a, np.ndarray) and a.shape == (4,) and isinstance(b, np.ndarray) and b.shape == (4,)):
        raise ValueError("input must be (4,) quaternion")
    return rc.quaternion_multiply(torch.tensor(a,dtype=torch.float32,device="cpu"),torch.tensor(b,dtype=torch.float32,device="cpu")).numpy()
    

if __name__ == '__main__':
    v1 = np.array([1,0,0])
    v2 = np.array([1,2,5])
    R = rotation_matrix_from_vectors(v1,v2)
    quat = matrix_to_quaternion(R)
    quat_invert = quaternion_invert(quat)
    print(quat)
    print(quat_invert)
    print(quaternion_multiply(quat,quat_invert))
    