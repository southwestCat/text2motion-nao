import numpy as np

kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
kinematic_tree_sub = [[2, 5, 8, 11], [1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [14, 17, 19, 21], [13, 16, 18, 20]]

file_path = "../skeleton_npy/001.npy"
data_src = np.load(file_path)
data = data_src.copy()

data[:,:,0] = data_src[:,:,2]
data[:,:,1] = data_src[:,:,0]
data[:,:,2] = data_src[:,:,1]

data_mirror_src = data.copy()
data_mirror = data_mirror_src.copy()

data_mirror[:,:,1] = -data_mirror_src[:,:,1]

data_mirror_copy = data_mirror.copy()

data_mirror[:,kinematic_tree_sub[0]] = data_mirror_copy[:,kinematic_tree_sub[1]]
data_mirror[:,kinematic_tree_sub[1]] = data_mirror_copy[:,kinematic_tree_sub[0]]
data_mirror[:,kinematic_tree_sub[2]] = data_mirror_copy[:,kinematic_tree_sub[2]]
data_mirror[:,kinematic_tree_sub[3]] = data_mirror_copy[:,kinematic_tree_sub[4]]
data_mirror[:,kinematic_tree_sub[4]] = data_mirror_copy[:,kinematic_tree_sub[3]]

data_save = data_mirror.copy()
data_save[:,:,2] = data_mirror[:,:,0]
data_save[:,:,0] = data_mirror[:,:,1]
data_save[:,:,1] = data_mirror[:,:,2]

save_path = file_path[:-4] + "_mirror.npy"
np.save(save_path,data_save)
print("save to ",save_path)
