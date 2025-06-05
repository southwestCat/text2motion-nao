import numpy as np
from rotation import *

humanml3d_joints_dict = {
    "Root": 0,
    "Neck": 9,

    "LShoulderInner": 13,
    "LShoulder": 16,
    "LElbow": 18,
    "LWrist": 20,

    "RShoulderInner": 14,
    "RShoulder": 17,
    "RElbow": 19,
    "RWrist": 21,

    "LHip": 1,
    "LKnee": 4,
    "LAnkle": 7,
    "LFoot": 10,

    "RHip": 2,
    "RKnee": 5,
    "RAnkle": 8,
    "RFoot": 11
}


def clamp_v(v:np.array):
    if (np.linalg.norm(v) > 1.0):
        v = v / np.linalg.norm(v)
    return v


def norm_v(v:np.array):
    if (np.linalg.norm(v) < 1e-6):
        raise ValueError("norm_v vector length < 1e-6")
    return v / np.linalg.norm(v)


def calc_arm_length_max(skeleton_data:np.ndarray) -> float:
    l_arm_length_max = 0.5
    r_arm_length_max = 0.5
    len = skeleton_data.shape[0]
    for i in range(len):
        lshoulder = skeleton_data[i, humanml3d_joints_dict["LShoulder"], :]
        lelbow = skeleton_data[i, humanml3d_joints_dict["LElbow"], :]
        lwrist = skeleton_data[i, humanml3d_joints_dict["LWrist"], :]
        larm_length = calc_arm_length(lshoulder, lelbow, lwrist)
        if (larm_length > l_arm_length_max):
            l_arm_length_max = larm_length
        rshoulder = skeleton_data[i, humanml3d_joints_dict["RShoulder"], :]
        relbow = skeleton_data[i, humanml3d_joints_dict["RElbow"], :]
        rwrist = skeleton_data[i, humanml3d_joints_dict["RWrist"], :]
        rarm_length = calc_arm_length(rshoulder, relbow, rwrist)
        if (rarm_length > r_arm_length_max):
            r_arm_length_max = rarm_length
    return l_arm_length_max, r_arm_length_max


def calc_leg_length_max(skeleton_data:np.ndarray) -> float:
    l_leg_length_max = 0.5
    r_leg_length_max = 0.5
    len = skeleton_data.shape[0]
    for i in range(len):
        lhip = skeleton_data[i, humanml3d_joints_dict["LHip"], :]
        lknee = skeleton_data[i, humanml3d_joints_dict["LKnee"], :]
        lankle = skeleton_data[i, humanml3d_joints_dict["LAnkle"], :]
        lleg_length = calc_leg_length(lhip, lknee, lankle)
        if (lleg_length > l_leg_length_max):
            l_leg_length_max = lleg_length
        rhip = skeleton_data[i, humanml3d_joints_dict["RHip"], :]
        rknee = skeleton_data[i, humanml3d_joints_dict["RKnee"], :]
        rankle = skeleton_data[i, humanml3d_joints_dict["RAnkle"], :]
        rleg_length = calc_leg_length(rhip, rknee, rankle)
        if (rleg_length > r_leg_length_max):
            r_leg_length_max = rleg_length
    return l_leg_length_max, r_leg_length_max


def calc_arm_length(shoulder:np.ndarray, elbow:np.ndarray, wrist:np.ndarray) -> float:
    if not (isinstance(shoulder, np.ndarray) and shoulder.shape == (3,)):
        raise ValueError("shoulder must be a numpy array of shape (3,)")
    if not (isinstance(elbow, np.ndarray) and elbow.shape == (3,)):
        raise ValueError("elbow must be a numpy array of shape (3,)")
    if not (isinstance(wrist, np.ndarray) and wrist.shape == (3,)):
        raise ValueError("wrist must be a numpy array of shape (3,)")
    
    arm_length = np.linalg.norm(shoulder - elbow) + np.linalg.norm(elbow - wrist)
    return arm_length


def calc_leg_length(hip:np.ndarray, knee:np.ndarray, ankle:np.ndarray) -> float:
    if not (isinstance(hip, np.ndarray) and hip.shape == (3,)):
        raise ValueError("hip must be a numpy array of shape (3,)")
    if not (isinstance(knee, np.ndarray) and knee.shape == (3,)):
        raise ValueError("knee must be a numpy array of shape (3,)")
    if not (isinstance(ankle, np.ndarray) and ankle.shape == (3,)):
        raise ValueError("ankle must be a numpy array of shape (3,)")
    leg_length = np.linalg.norm(hip - knee) + np.linalg.norm(knee - ankle)
    return leg_length


def calc_root_forward(root:np.ndarray, lhip:np.ndarray, rhip:np.ndarray) -> np.ndarray:
    if not (isinstance(root, np.ndarray) and root.shape == (3,)):
        raise ValueError("root must be a numpy array of shape (3,)")
    if not (isinstance(lhip, np.ndarray) and lhip.shape == (3,)):
        raise ValueError("lhip must be a numpy array of shape (3,)")
    if not (isinstance(rhip, np.ndarray) and rhip.shape == (3,)):
        raise ValueError("rhip must be a numpy array of shape (3,)")
    
    root_rhip = rhip - root
    root_lhip = lhip - root
    root_forward = np.cross(root_rhip, root_lhip)
    root_forward = root_forward / np.linalg.norm(root_forward)
    return root_forward


def calc_neck_forward(neck:np.ndarray, lshoulder:np.ndarray, rshoulder:np.ndarray) -> np.ndarray:
    if not (isinstance(neck, np.ndarray) and neck.shape == (3,)):
        raise ValueError("neck must be a numpy array of shape (3,)")
    if not (isinstance(lshoulder, np.ndarray) and lshoulder.shape == (3,)):
        raise ValueError("lshoulder must be a numpy array of shape (3,)")
    if not (isinstance(rshoulder, np.ndarray) and rshoulder.shape == (3,)):
        raise ValueError("rshoulder must be a numpy array of shape (3,)")

    neck_lshoulder = lshoulder - neck
    neck_rshoulder = rshoulder - neck
    neck_forward = np.cross(neck_lshoulder, neck_rshoulder)
    neck_forward = neck_forward / np.linalg.norm(neck_forward)
    return neck_forward


def get_root_forward_0(skeleton_data:np.ndarray) -> np.ndarray:
    root0 = skeleton_data[0, humanml3d_joints_dict["Root"], :]
    lhip0 = skeleton_data[0, humanml3d_joints_dict["LHip"], :]
    rhip0 = skeleton_data[0, humanml3d_joints_dict["RHip"], :]
    return calc_root_forward(root0, lhip0, rhip0)



def main():
    nao_v_max = 0.25
    nao_leg_length = (100+102.9+45.19) / 1000.0
    nao_arm_length = (105+55.95+57.75) / 1000.0
    human_leg_length = 1.0
    human_nao_ratio = human_leg_length / nao_leg_length
    human_arm_length = 0.5

    motion_id = "001"
    file_path = "../skeleton_npy/"+motion_id+".npy"
    fps = 20.0
    dt = 1.0 / fps
    data_src = np.load(file_path)
    data = data_src.copy()
    data[:,:,0] = data_src[:,:,2]
    data[:,:,1] = data_src[:,:,0]
    data[:,:,2] = data_src[:,:,1]

    frame_len = data.shape[0]
    save_npy = np.zeros((frame_len, 4, 7), dtype=np.float32)

    larm_length_max, rarm_length_max = calc_arm_length_max(data)
    lleg_length_max, rleg_length_max = calc_leg_length_max(data)

    # frame_id = 0
    for frame_id in range(frame_len):
        root = data[frame_id, humanml3d_joints_dict["Root"], :]
        neck = data[frame_id, humanml3d_joints_dict["Neck"], :]

        lshoulder = data[frame_id, humanml3d_joints_dict["LShoulder"], :]
        lelbow = data[frame_id, humanml3d_joints_dict["LElbow"], :]
        lwrist = data[frame_id, humanml3d_joints_dict["LWrist"], :]

        rshoulder = data[frame_id, humanml3d_joints_dict["RShoulder"], :]
        relbow = data[frame_id, humanml3d_joints_dict["RElbow"], :]
        rwrist = data[frame_id, humanml3d_joints_dict["RWrist"], :]

        lhip = data[frame_id, humanml3d_joints_dict["LHip"], :]
        lknee = data[frame_id, humanml3d_joints_dict["LKnee"], :]
        lankle = data[frame_id, humanml3d_joints_dict["LAnkle"], :]
        lfoot = data[frame_id, humanml3d_joints_dict["LFoot"], :]

        rhip = data[frame_id, humanml3d_joints_dict["RHip"], :]
        rknee = data[frame_id, humanml3d_joints_dict["RKnee"], :]
        rankle = data[frame_id, humanml3d_joints_dict["RAnkle"], :]
        rfoot = data[frame_id, humanml3d_joints_dict["RFoot"], :]

        root_forward = calc_root_forward(root, lhip, rhip)
        root_forward_0 = get_root_forward_0(data)
        root_forward_rot_w = rotation_matrix_from_vectors(root_forward, root_forward_0)

        larm_trans_w = lwrist - lshoulder
        larm_trans_root = root_forward_rot_w@larm_trans_w
        larm_trans_spec = larm_trans_root / larm_length_max
        lhand_trans_w = lwrist - lelbow
        lhand_trans_root = root_forward_rot_w@lhand_trans_w
        lhand_trans_root = norm_v(lhand_trans_root)
        lhand_trans_0 = np.array([1.0,0.0,0.0],dtype=np.float32)
        lhand_rot = rotation_matrix_from_vectors(lhand_trans_0, lhand_trans_root)
        larm_quat_spec = matrix_to_quaternion(lhand_rot)
 
        rarm_trans_w = rwrist - rshoulder
        rarm_trans_root = root_forward_rot_w@rarm_trans_w
        rarm_trans_spec = rarm_trans_root / rarm_length_max
        rhand_trans_w = rwrist - relbow
        rhand_trans_root = root_forward_rot_w@rhand_trans_w
        rhand_trans_root = norm_v(rhand_trans_root)
        rhand_trans_0 = np.array([1.0,0.0,0.0],dtype=np.float32)
        rhand_rot = rotation_matrix_from_vectors(rhand_trans_0, rhand_trans_root)
        rarm_quat_spec = matrix_to_quaternion(rhand_rot)

        lleg_trans_w = lankle - lhip
        lleg_trans_root = root_forward_rot_w@lleg_trans_w
        lleg_trans_spec = lleg_trans_root / lleg_length_max
        lfoot_trans_w = lfoot - lankle
        lfoot_trans_root = root_forward_rot_w@lfoot_trans_w
        lfoot_trans_root[2] = 0.0
        lfoot_forward = norm_v(lfoot_trans_root)
        lfoot_forward_0 = np.array([1.0,0.0,0.0],dtype=np.float32)
        lfoot_rot = rotation_matrix_from_vectors(lfoot_forward_0, lfoot_forward)
        lleg_quat_spec = matrix_to_quaternion(lfoot_rot)

        rleg_trans_w = rankle - rhip
        rleg_trans_root = root_forward_rot_w@rleg_trans_w
        rleg_trans_spec = rleg_trans_root / rleg_length_max
        rfoot_trans_w = rfoot - rankle
        rfoot_trans_root = root_forward_rot_w@rfoot_trans_w
        rfoot_trans_root[2] = 0.0
        rfoot_forward = norm_v(rfoot_trans_root)
        rfoot_forward_0 = np.array([1.0,0.0,0.0],dtype=np.float32)
        rfoot_rot = rotation_matrix_from_vectors(rfoot_forward_0, rfoot_forward)
        rleg_quat_spec = matrix_to_quaternion(rfoot_rot)

        save_npy[frame_id, 0, :3] = larm_trans_spec
        save_npy[frame_id, 0, 3:] = larm_quat_spec
        save_npy[frame_id, 1, :3] = rarm_trans_spec
        save_npy[frame_id, 1, 3:] = rarm_quat_spec
        save_npy[frame_id, 2, :3] = lleg_trans_spec
        save_npy[frame_id, 2, 3:] = lleg_quat_spec
        save_npy[frame_id, 3, :3] = rleg_trans_spec
        save_npy[frame_id, 3, 3:] = rleg_quat_spec
    save_path = "../pose2joint_datasets/"+motion_id+"_limb_spec.npy"
    np.save(save_path, save_npy)
    print("save limb spec to "+save_path)



if __name__ == '__main__':
    main()
