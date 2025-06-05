import numpy as np

def DEG(x):
    return x*np.pi/180.0
def RAD(x):
    return x*180.0/np.pi

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


def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)

    if trace > 0:
        w = np.sqrt(1.0 + trace) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            x = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
            w = (R[2, 1] - R[1, 2]) / (4 * x)
            y = (R[0, 1] + R[1, 0]) / (4 * x)
            z = (R[0, 2] + R[2, 0]) / (4 * x)
        elif R[1, 1] > R[2, 2]:
            y = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) / 2
            w = (R[0, 2] - R[2, 0]) / (4 * y)
            x = (R[0, 1] + R[1, 0]) / (4 * y)
            z = (R[1, 2] + R[2, 1]) / (4 * y)
        else:
            z = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) / 2
            w = (R[1, 0] - R[0, 1]) / (4 * z)
            x = (R[0, 2] + R[2, 0]) / (4 * z)
            y = (R[1, 2] + R[2, 1]) / (4 * z)

    return np.array([w, x, y, z])