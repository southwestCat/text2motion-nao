# Realizing Text-Driven Motion Generation on NAO Robot: A Reinforcement Learning-Optimized Control Pipeline

> This repository contains the source code and supplementary materials for the paper.



## Dependencies

+ Ubuntu 22.04,

+ [Isaaclab](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.0.0)=2.0.0,
+ IsaacSim=4.5,
+ rsl_rl=2.0.2,
+ ONNX Runtime=1.19.2
+ [ROS2 humble](https://docs.ros.org/en/humble/Installation.html)

Please refer to the Isaaclab installation documentation to install the necessary environment components.

We developed our code using Isaaclab 2.0.0. When using Isaaclab version 2.0.0, the version of `rsl_rl` must not exceed 2.0.2. 



## Directory Structure

```bash
.
├── motion_specification
├── pose2joint_angle_signal_net
├── pose2joint_datasets
├── process_diffusion_outputs
├── README.md
├── replay_motion_isaac
├── rl_nao
├── ros2_ws
└── skeleton_npy
```



## Usage

1. **Process diffusion outputs**.  The diffusion output is saved in `process_diffusion_outputs`, run the `export.py` script to get the skeleton data. Then copy `001.npy` to `skeleton_npy`.

   ```bash
   cd process_diffusion_outputs
   python export.py
   ```

2. **Generate Motion Specification**. In `motion_specification`, using `draw_3d_skeleton_dynamic.py` to visualize the `001.npy` motion. Using `mirros_skeleton.py` to generate mirrored data from the original motion data, thereby augmenting the dataset. Run `generate_limb_motion_specifications.py` to generate the normalized the motion specification, the name of `motion_id` needs to be modified manually.

   ```bash
   python draw_3d_skeleton_dynamic.py
   python mirror_skeleton.py 
   python generate_limb_motion_specifications.py
   ```

3. **Train Angle Signal Net**. Run `iknn.py` to train the angle signal network, and the best weight will saved in weights after the training. Using `export_joint_angles.py` to export the joint angle commands for NAO. Then run `interpolate_default_offset.py` to upsample the output joint commands to match the motion control cycle of the NAO robot. Create directories named `PoseMapping/npy`, `PoseMapping/train_npy` and `PoseMapping/play_npy`. Copy the interpolated data to the `PoseMapping/npy` , `PoseMapping/train_npy` and `PoseMapping/npy`.

   ```bash
   python iknn.py
   python export_joint_angles.py
   python interpolate_default_offset.py
   mkdir -p ~/PoseMapping/npy/ ~/PoseMapping/train_npy/ ~/PoseMapping/play_npy/
   cp interpolates/* ~/PoseMapping/npy/
   cp interpolates/* ~/PoseMapping/train_npy/
   cp interpolates/* ~/PoseMapping/play_npy/
   ```

4. **Replay the Interpolated Motion**.  Using `replay_motion_set.py` in `replay_motion_isaac` to check if the generated interpolated data is correct. This file requires the support of IsaacSim. 

   ```bash
   python replay_motion_set.py
   ```

5.  **Train RL Policy**. Using `train.sh` to start the rl training and `play.sh` for evaluating the results. After run `play`, the rl policy file `policy.onnx` can be find in `logs/rsl_rl/nao_pose_random/${Year-Month-Day_HH_MM_SS}/exported/`

   ```bash
   ./train.sh --task PoseRandom-NAO --seed 42 --device cuda --headless --num_envs 4096
   ./play.sh --task PoseRandom-NAO-Play --device cuda --num_envs 1
   ```

6. **Implement RL Policy On Real NAO**. Modified the `scripts/export.py` according to the RL training environment configurations, including CommandsCfg, ActionCfg and ObservationCfg. Then run the `export.sh` to generate `policy.py` which is used for real NAO implementation. Some variable values need to be edited manually. The `policy.py` defines the joint order for the RL policy outputs, as well as the action scale, default joint positions, etc., which are crucial for controlling the real robot.

   ```bash
   ./export.sh
   ```

7. Copy `~/PoseMapping/npy/*`to `ros2_ws/rl_controller/rl_controller/data`,and copy exported  `policy.onnx` to `ros2_ws/rl_controller/rl_controller/policy`, edit the selected motion name in `ros2_ws/rl_controller/rl_controller/config/config.yaml`. Build the `ros2_ws/nao_driver`,`ros2_ws/nao_interfaces` and `ros2_ws/rl_controller` in ROS2 workspace. We recommend `colcon build` for the build command. After the compilation is complete, copy the `install` folder to the `${HOME}` of the NAO robot.

   One important thing to note is that the NAO robot cannot use the original system; it need to be flashed to the [bhuman](https://docs.b-human.de/coderelease2024/) system. We have verified this code on `bhuman2023release`. Both `bhuman2023release` and `bhuman2024release` are built on Ubuntu Jammy as the base system, and both should be feasible.

 
