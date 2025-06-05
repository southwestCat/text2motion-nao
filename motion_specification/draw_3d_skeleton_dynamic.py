import matplotlib.pyplot as plt
import numpy as np

humanml3d_joints = [
    "root",   # 0: root
    "RH",     # 1: right hip
    "LH",     # 2: left hip
    "BP",     # 3: back of pelvis
    "RK",     # 4: right knee
    "LK",     # 5: left knee
    "BT",     # 6: back of torso     
    "RMrot",  # 7: right malleolus rotation 
    "LMrot",  # 8: left malleolus rotation
    "BLN",    # 9: back low neck
    "RF",     # 10: right foot
    "LF",     # 11: left foot
    "BMN",    # 12: back mid neck
    "RSI",    # 13: right shoulder inner
    "LSI",    # 14: left shoulder inner
    "BUN",    # 15: back upper neck
    "RS",     # 16: right shoulder
    "LS",     # 17: left shoulder
    "RE",     # 18: right elbow
    "LE",     # 19: left elbow
    "RW",     # 20: right wrist
    "LW",     # 21: left wrist
]


kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]


file_path = "../skeleton_npy/001.npy"
fps = 20.0
dt = 1.0 / fps
data_src = np.load(file_path)
data = data_src.copy()
data[:,:,0] = data_src[:,:,2]
data[:,:,1] = data_src[:,:,0]
data[:,:,2] = data_src[:,:,1]

frame_len = data.shape[0]
frame_id = 0

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.ion()

print(f"frame_len: {frame_len}")
for frame_id in range(frame_len):
    print(f"frame_id: {frame_id}.", end="\r")
    ax.clear()
    color_list = ["red", "green", "b", "k", "cyan"]
    for node in kinematic_tree:
        x = data[frame_id, node, 0]
        y = data[frame_id, node, 1]
        z = data[frame_id, node, 2]
        ax.scatter(x, y, z, color=color_list[0], s=50)
        ax.plot(x, y, z, color=color_list[0])
        color_list.pop(0)

    Root_node = 0
    x_root = data[frame_id, Root_node, 0]
    y_root = data[frame_id, Root_node, 1]
    z_root = data[frame_id, Root_node, 2]
    ax.text(x_root, y_root, z_root, 'Root', color='black', fontsize=10, zorder=10)

    LHip_node = 2
    x_LHip = data[frame_id, LHip_node, 0]
    y_LHip = data[frame_id, LHip_node, 1]
    z_LHip = data[frame_id, LHip_node, 2]
    ax.text(x_LHip, y_LHip-0.3, z_LHip, 'RHip', color='black', fontsize=10, zorder=10)

    RKnee_node = 5
    x_RAnkle = data[frame_id, RKnee_node, 0]
    y_RAnkle = data[frame_id, RKnee_node, 1]
    z_RAnkle = data[frame_id, RKnee_node, 2]
    ax.text(x_RAnkle, y_RAnkle-0.3, z_RAnkle, 'RKnee', color='black', fontsize=10, zorder=10)

    LAnkle_node = 8
    x_LAnkle = data[frame_id, LAnkle_node, 0]
    y_LAnkle = data[frame_id, LAnkle_node, 1]
    z_LAnkle = data[frame_id, LAnkle_node, 2]
    ax.text(x_LAnkle, y_LAnkle-0.5, z_LAnkle, 'RAnkle', color='black', fontsize=10, zorder=10)

    RFoot_node = 11
    x_LFoot = data[frame_id, RFoot_node, 0]
    y_LFoot = data[frame_id, RFoot_node, 1]
    z_LFoot = data[frame_id, RFoot_node, 2]
    ax.text(x_LFoot, y_LFoot-0.3, z_LFoot-0.1, 'RFoot', color='black', fontsize=10, zorder=10)

    RHip_node = 1
    x_RHip = data[frame_id, RHip_node, 0]
    y_RHip = data[frame_id, RHip_node, 1]
    z_RHip = data[frame_id, RHip_node, 2]
    ax.text(x_RHip, y_RHip, z_RHip, 'LHip', color='black', fontsize=10, zorder=10)

    LAnkle_node = 4
    x_LAnkle = data[frame_id, LAnkle_node, 0]
    y_LAnkle = data[frame_id, LAnkle_node, 1]
    z_LAnkle = data[frame_id, LAnkle_node, 2]
    ax.text(x_LAnkle, y_LAnkle, z_LAnkle, 'LKnee', color='black', fontsize=10, zorder=10)

    RAnkle_node = 7
    x_RAnkle = data[frame_id, RAnkle_node, 0]
    y_RAnkle = data[frame_id, RAnkle_node, 1]
    z_RAnkle = data[frame_id, RAnkle_node, 2]
    ax.text(x_RAnkle, y_RAnkle, z_RAnkle, 'LAnkle', color='black', fontsize=10, zorder=10)

    LFoot_node = 10
    x_LFoot = data[frame_id, LFoot_node, 0]
    y_LFoot = data[frame_id, LFoot_node, 1]
    z_LFoot = data[frame_id, LFoot_node, 2]
    ax.text(x_LFoot, y_LFoot, z_LFoot-0.1, 'LFoot', color='black', fontsize=10, zorder=10)

    Neck_node = 9
    x_Neck = data[frame_id, Neck_node, 0]
    y_Neck = data[frame_id, Neck_node, 1]
    z_Neck = data[frame_id, Neck_node, 2]
    ax.text(x_Neck, y_Neck, z_Neck, 'Neck', color='black', fontsize=10, zorder=10)

    RShoulder_node = 17
    x_RShoulder = data[frame_id, RShoulder_node, 0]
    y_RShoulder = data[frame_id, RShoulder_node, 1]
    z_RShoulder = data[frame_id, RShoulder_node, 2]
    ax.text(x_RShoulder, y_RShoulder-0.6, z_RShoulder, 'RShoulder', color='black', fontsize=10, zorder=10)

    RElbow_node = 19
    x_RElbow = data[frame_id, RElbow_node, 0]
    y_RElbow = data[frame_id, RElbow_node, 1]
    z_RElbow = data[frame_id, RElbow_node, 2]
    ax.text(x_RElbow, y_RElbow-0.5, z_RElbow, 'RElbow', color='black', fontsize=10, zorder=10)

    RWrist_node = 21
    x_RWrist = data[frame_id, RWrist_node, 0]
    y_RWrist = data[frame_id, RWrist_node, 1]
    z_RWrist = data[frame_id, RWrist_node, 2]
    ax.text(x_RWrist, y_RWrist-0.5, z_RWrist, 'RWrist', color='black', fontsize=10, zorder=10)

    LShoulder_node = 16
    x_LShoulder = data[frame_id, LShoulder_node, 0]
    y_LShoulder = data[frame_id, LShoulder_node, 1]
    z_LShoulder = data[frame_id, LShoulder_node, 2]
    ax.text(x_LShoulder, y_LShoulder, z_LShoulder, 'LShoulder', color='black', fontsize=10, zorder=10)

    LElbow_node = 18
    x_LElbow = data[frame_id, LElbow_node, 0]
    y_LElbow = data[frame_id, LElbow_node, 1]
    z_LElbow = data[frame_id, LElbow_node, 2]
    ax.text(x_LElbow, y_LElbow, z_LElbow, 'LElbow', color='black', fontsize=10, zorder=10)

    LWrist_node = 20
    x_LWrist = data[frame_id, LWrist_node, 0]
    y_LWrist = data[frame_id, LWrist_node, 1]
    z_LWrist = data[frame_id, LWrist_node, 2]
    ax.text(x_LWrist, y_LWrist, z_LWrist, 'LWrist', color='black', fontsize=10, zorder=10)

    ax.view_init(elev=0, azim=0)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax.set_xticks([])

    plt.axis('equal') 
    plt.tight_layout()
    plt.pause(dt)

print("Draw finished...")
plt.ioff()
plt.show()

