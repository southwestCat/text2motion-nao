import numpy as np

def DEG(x:float) -> float:
    return x * np.pi / 180.0


def interpolate(from_pos:np.ndarray, to_pos:np.ndarray, ratio:float) -> np.ndarray:
    assert len(from_pos) == len(to_pos), f"from_pos and to_pos must have the same length, but got {len(from_pos)} and {len(to_pos)}"

    target = np.zeros_like(from_pos)
    for i in range(len(from_pos)):
        f = from_pos[i]
        t = to_pos[i]
        target[i] = ratio*(t-f)+f
    return target


def check_policy_joint_names(robot_joint_names, policy_joint_names):
    for name in policy_joint_names:
        assert name in robot_joint_names, f"Joint name {name} not found in robot_joint_names"