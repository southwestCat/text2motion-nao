import os
import numpy as np
from scipy.interpolate import interp1d
from motion_config import DEFAULT_JOINT_NAMES, DEFAULT_JOINT_OFFSET


def interpolate_default_offset_to_frame(default_offset:np.array,frame_offset:np.array,dt:float,T:float=1.0):
    target_offset = default_offset.copy()
    for joint_name in DEFAULT_JOINT_NAMES:
        joint_id = DEFAULT_JOINT_NAMES.index(joint_name)
        target_offset[joint_id] = frame_offset[joint_id]
    steps = int(T / dt)
    arr0 = default_offset.copy()
    arr1 = target_offset.copy()
    interpolated_results = [
        arr0 * (1 - alpha) + arr1 * alpha
        for alpha in np.linspace(0, 1, steps)
    ]
    return np.array(interpolated_results)


def joint_dict_to_np(joint_dict) -> np.array:
    joint_arr = np.array([0.0]*len(DEFAULT_JOINT_NAMES), dtype=np.float32)
    for joint_name, joint_angle in joint_dict.items():
        joint_angle = float(joint_angle)
        joint_id = DEFAULT_JOINT_NAMES.index(joint_name)
        joint_arr[joint_id] = joint_angle
    rhipyawpitch_id = DEFAULT_JOINT_NAMES.index("RHipYawPitch")
    lhipyawpitch_id = DEFAULT_JOINT_NAMES.index("LHipYawPitch")
    joint_arr[rhipyawpitch_id] = joint_arr[lhipyawpitch_id]
    return joint_arr


def interpolate_motion_upsampling():
    output_path = "outputs/"
    for file_name in os.listdir(output_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(output_path, file_name)
            data = np.load(file_path)
            original_fps = 20
            target_fps_list = [40, 80, 160]
            print(data.shape)

            current_data = data
            current_fps = original_fps

            for target_fps in target_fps_list:
                original_time = np.linspace(0,len(current_data)/current_fps,len(current_data))
                target_time = np.linspace(0, len(current_data) / current_fps, int(len(current_data) * (target_fps / current_fps)))
                upsampled_data = np.zeros((len(target_time), current_data.shape[1]))
                for i in range(current_data.shape[1]):
                    interp_func = interp1d(original_time, current_data[:, i], kind='linear', fill_value="extrapolate")
                    upsampled_data[:, i] = interp_func(target_time)
                current_data = upsampled_data
                current_fps = target_fps
            
            t = 0.0
            dt = 0.012
            duration = 1.0/current_fps
            # interpolate default pose to frame0 pose
            interpolated = interpolate_default_offset_to_frame(
                joint_dict_to_np(DEFAULT_JOINT_OFFSET),
                data[0],
                dt=dt,
                T=3.0
            )
            frame_id = 0
            while frame_id < len(current_data):
                frame_id = int(t/duration)
                if frame_id >= len(current_data): break
                curr_frame_offset = current_data[frame_id]
                curr_frame_offset = np.expand_dims(curr_frame_offset, axis=0)
                interpolated = np.concatenate((interpolated, curr_frame_offset), axis=0)
                t += dt
            npy_save_path = "interpolates/"+file_name
            np.save(npy_save_path, interpolated)
            print(f"save {npy_save_path} success!")
        


def main():
    interpolate_motion_upsampling()
    

if __name__ == '__main__':
    main()