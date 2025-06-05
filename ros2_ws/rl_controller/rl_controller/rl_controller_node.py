import os
import rclpy
from rclpy.node import Node
import numpy as np
import yaml
import time
import onnxruntime as ort

from ament_index_python.packages import get_package_share_directory
from nao_interfaces.msg import NaoActuator, NaoSensor, Joints

from .observations import Observations
from .motion_utils import (
    interpolate,
    check_policy_joint_names
)
from .motion_config import (
    MOTION_LOOP_TIME,
    JointNames,
    create_default_joint_offset,
)
from .policy import (
    POLICY_INPUT_DIM,
    POLICY_OUTPUT_DIM,
    robot_joint_names,
    policy_action_joint_names,
    policy_action_joint_scale,
    policy_command_joint_names,
    policy_observation_joint_names,
    policy_default_joint_pos
)


class RLControllerNode(Node):
    '''
    RLControllerNode
    
    This class represents a ROS 2 node that implements a Reinforcement Learning (RL) controller.
    It subscribes to sensor data, processes it, and publishes actuator commands based on the RL policy.
    '''

    def __init__(self):
        super().__init__('rl_controller_node')
        self.subscription_ = self.create_subscription(
            NaoSensor,
            'nao_sensor',
            self.sensor_callback,
            1)
        self.publisher_ = self.create_publisher(
            NaoActuator,
            'nao_actuator',
            10)
        
        # Debug Mode, no publish data
        self.DEBUG_ = False
        
        # motion time
        self.t_ = 0.0

        # requests to nao
        self.joint_stiffness_requests_ = np.array([0.0]*Joints.NUM_OF_JOINTS, dtype=np.float32)
        self.joint_position_requests_ = np.zeros_like(self.joint_stiffness_requests_)
        self.last_joint_position_requests_ = self.joint_position_requests_.copy()

        # sensor readings
        self.sensor_position_ = np.zeros_like(self.joint_stiffness_requests_)
        self.sensor_acc_ = np.array([0.0,0.0,0.0], dtype=np.float32)
        self.sensor_gyro_ = np.array([0.0,0.0,0.0], dtype=np.float32)

        # init stand joint requests
        self.INIT_STAND_DURATION = 2.0
        self.init_stand_default_joint_position_ = create_default_joint_offset()
        self.init_standing_ = True
        self.joint_position_stand_interp_from_ = None

        # load npy file
        package_name = 'rl_controller'
        package_share_dir = get_package_share_directory(package_name)
        yaml_file_path = os.path.join(package_share_dir, 'config', 'config.yaml')
        _config = self.read_config(yaml_file_path)
        npy_file_name = _config["motion"]["name"]
        npy_file_path = os.path.join(package_share_dir, 'data', npy_file_name)

        npy_data:np.ndarray = np.load(npy_file_path)
        self.get_logger().info(f"Loaded data from {npy_file_path}, shape: {npy_data.shape}, dtype: {npy_data.dtype}")

        # load onnx model
        onnx_file_path = os.path.join(package_share_dir, 'policy', 'policy.onnx')
        if not os.path.isfile(onnx_file_path):
            self.get_logger().error(f"Onnx model not found: {onnx_file_path}")
            raise FileNotFoundError(f"Onnx model not found: {onnx_file_path}")
        self.ort_session_ = ort.InferenceSession(onnx_file_path)
        self.ort_input_name_ = self.ort_session_.get_inputs()[0].name
        self.ort_output_name_ = self.ort_session_.get_outputs()[0].name
        self.get_logger().info(f"Loaded onnx model from {onnx_file_path}, input: {self.ort_input_name_}, output: {self.ort_output_name_}")

        # motion frame (commands)
        self.frame_id_ = 0
        self.motion_data_ = npy_data
        self.motion_finished_ = False

        # check policy joint names
        check_policy_joint_names(robot_joint_names, policy_action_joint_names)
        check_policy_joint_names(robot_joint_names, policy_command_joint_names)
        check_policy_joint_names(robot_joint_names, policy_observation_joint_names)
        assert npy_data.shape[-1] == len(robot_joint_names), f"npy_data shape {npy_data.shape[-1]} != len(robot_joint_names) {len(robot_joint_names)}"

        # observations
        self.observations_ = Observations(MOTION_LOOP_TIME)

        # log joints
        self.logged_joints = list()
        self.log_written = False


    def read_config(self, file):
        try:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"Config not found: {file}")
            with open(file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.get_logger().error(f"Error reading config: {e}")
            return None
        
    
    def sensor_callback(self, msg:NaoSensor):
        self.read_nao_sensor(msg)
        self.observations_.update(msg)
        if self.joint_position_stand_interp_from_ is None:
            self.joint_position_stand_interp_from_ = self.sensor_position_.copy()
        self.update_joint_requests(self.t_)
        self.publish_nao_actuator()
        self.t_ += MOTION_LOOP_TIME


    def read_nao_sensor(self, msg:NaoSensor):
        self.sensor_position_ = msg.position.copy()
        self.sensor_acc_ = msg.acc.copy()
        self.sensor_gyro_ = msg.gyro.copy()


    def publish_nao_actuator(self):
        self.joint_position_requests_[Joints.R_HIP_YAW_PITCH] = self.joint_position_requests_[Joints.L_HIP_YAW_PITCH]
        msg = NaoActuator()
        msg.stiffness = self.joint_stiffness_requests_
        msg.position = self.joint_position_requests_
        if not self.DEBUG_:
            self.publisher_.publish(msg)
            self.last_joint_position_requests_ = self.joint_position_requests_.copy()


    def update_joint_requests(self,t):
        self.update_joint_stiffness_requests()
        self.update_joint_position_requests(t)


    def update_joint_stiffness_requests(self):
        self.joint_stiffness_requests_.fill(0.7)
        self.joint_stiffness_requests_[Joints.L_SHOULDER_ROLL] = 0.1
        self.joint_stiffness_requests_[Joints.R_SHOULDER_ROLL] = 0.1


    def update_joint_position_requests(self,t):
        if self.init_standing_:
            self.process_init_stand(t)
        elif not self.motion_finished_:
            self.onnx_motion_engine()
        else:
            if not self.log_written:
                _d = np.array(self.logged_joints)
                np.save("logged_joints.npy", _d)
                self.get_logger().info(f"Logged joints to logged_joints.npy, shape: {_d.shape}, dtype: {_d.dtype}")
                self.log_written = True

    
    def convert_frame_to_requests(self,frame_data):
        self.joint_position_requests_ = self.last_joint_position_requests_.copy()
        assert len(JointNames) ==  Joints.NUM_OF_JOINTS, "JointNames and Joints.NUM_OF_JOINTS mismatch"
        for i, name in enumerate(robot_joint_names):
            self.joint_position_requests_[JointNames.index(name)] = frame_data[i]


    def convert_actions_to_requests(self,actions):
        self.joint_position_requests_ = self.last_joint_position_requests_.copy()
        assert len(JointNames) ==  Joints.NUM_OF_JOINTS, "JointNames and Joints.NUM_OF_JOINTS mismatch"
        for i, name in enumerate(policy_action_joint_names):
            self.joint_position_requests_[JointNames.index(name)] = actions[i]


    def process_init_stand(self,t):
        if t > self.INIT_STAND_DURATION:
            self.init_standing_ = False
            return
        ratio = t / self.INIT_STAND_DURATION
        from_pos = self.joint_position_stand_interp_from_
        to_pos = self.init_stand_default_joint_position_
        self.joint_position_requests_ = interpolate(from_pos, to_pos, ratio)


    def create_observation_projected_gravity(self,projected_gravity:np.ndarray) -> np.ndarray:
        return projected_gravity.copy()
    

    def create_obervation_pose_commands(self, frame:np.ndarray) -> np.ndarray:
        commands = np.zeros(len(policy_command_joint_names), dtype=np.float32)
        for i, name in enumerate(policy_command_joint_names):
            commands[i] = frame[robot_joint_names.index(name)]
        return commands
    

    def create_observation_joint_pos_rel(self, joint_pos_rel:np.ndarray) -> np.ndarray:
        return joint_pos_rel.copy()
    

    def create_observation_actions(self, last_actions:np.ndarray) -> np.ndarray:
        return last_actions.copy()


    def create_observations(self) -> np.ndarray:
        projected_gravity = self.create_observation_projected_gravity(self.observations_.gravity_projection)
        pose_commands = self.create_obervation_pose_commands(self.motion_data_[self.frame_id_])
        joint_pos_rel = self.create_observation_joint_pos_rel(self.observations_.joint_pos_rel)
        actions = self.create_observation_actions(self.observations_.last_actions)
        obs = np.concatenate((projected_gravity, pose_commands, joint_pos_rel, actions))
        return np.reshape(obs, (1,POLICY_INPUT_DIM))
    

    def process_actions(self,raw_actions:np.ndarray) -> np.ndarray:
        assert len(raw_actions) == len(policy_action_joint_names), f"actions len {len(raw_actions)}!= len(policy_action_joint_names) {len(policy_action_joint_names)}"
        
        actions = np.zeros(len(policy_action_joint_names), dtype=np.float32)
        if isinstance(policy_action_joint_scale, float):
            for i, name in enumerate(policy_action_joint_names):
                actions[i] = raw_actions[i] * policy_action_joint_scale + policy_default_joint_pos[name]
        elif isinstance(policy_action_joint_scale, dict):
            for i, name in enumerate(policy_action_joint_names):
                actions[i] = raw_actions[i] * policy_action_joint_scale[name] + policy_default_joint_pos[name]
        else:
            raise ValueError(f"policy_action_joint_scale type {type(policy_action_joint_scale)} not supported")
        return actions
    

    def apply_actions(self, actions:np.ndarray) -> np.ndarray:
        joint_requests = create_default_joint_offset()
        for i, name in enumerate(policy_action_joint_names):
            joint_requests[JointNames.index(name)] = actions[i]
        joint_requests[Joints.R_HIP_YAW_PITCH] = joint_requests[Joints.L_HIP_YAW_PITCH]
        return joint_requests
    

    def onnx_motion_engine(self):
        if self.frame_id_ >= len(self.motion_data_):
            if not self.motion_finished_:
                self.get_logger().info("Motion finished.")
            self.motion_finished_ = True
            return
        start_time = time.time()
        # onnx start
        observations = self.create_observations()
        assert observations.shape[-1] == POLICY_INPUT_DIM, f"observations shape {observations.shape}!= (1,POLICY_INPUT_DIM)"
        outputs = self.ort_session_.run([self.ort_output_name_], {self.ort_input_name_: observations.reshape(1,-1)})
        assert outputs[0].shape[-1] == POLICY_OUTPUT_DIM, f"outputs shape {outputs[0].shape[-1]}!= (POLICY_OUTPUT_DIM,)"
        outputs = np.reshape(outputs, POLICY_OUTPUT_DIM)
        self.observations_.update_last_actions(outputs)
        processed_actions = self.process_actions(outputs)
        self.joint_position_requests_ = self.apply_actions(processed_actions)
        # onnx end
        end_time = time.time()
        print(f"id: {self.frame_id_}, cost: {end_time-start_time:.6f}s", end="\r")

        self.frame_id_ += 1

        self.logged_joints.append(self.sensor_position_.copy())


def main(args=None):
    try:
        rclpy.init(args=args)
        rl_controller = RLControllerNode()
        rclpy.spin(rl_controller)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
    finally:
        if rl_controller:
            rl_controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Node shutdown complete.")


if __name__ == '__main__':
    main()
