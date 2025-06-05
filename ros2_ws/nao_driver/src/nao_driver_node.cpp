#include <rclcpp/rclcpp.hpp>
#include <nao_interfaces/msg/nao_actuator.hpp>
#include <nao_interfaces/msg/nao_sensor.hpp>
#include <nao_interfaces/msg/joints.hpp>
#include <thread>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <sstream>
#include <cmath>
#include <csignal>

#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "json.hpp"
#include "joint_limits.h"
#include "lola_message.h"

using json = nlohmann::json;
using nao_interfaces::msg::Joints;
using std::placeholders::_1;

// Joint order in LoLA data
enum JointOrdLoLA
{
  HeadYaw,
  HeadPitch,
  LShoulderPitch,
  LShoulderRoll,
  LElbowYaw,
  LElbowRoll,
  LWristYaw,
  LHipYawPitch,
  RHipYawPitch = LHipYawPitch,
  LHipRoll,
  LHipPitch,
  LKneePitch,
  LAnklePitch,
  LAnkleRoll,
  RHipRoll,
  RHipPitch,
  RKneePitch,
  RAnklePitch,
  RAnkleRoll,
  RShoulderPitch,
  RShoulderRoll,
  RElbowYaw,
  RElbowRoll,
  RWristYaw,
  LHand,
  RHand
};
const int jointMappings[Joints::NUM_OF_JOINTS] = {
    HeadYaw, HeadPitch,
    LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw, LHand,
    RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw, RHand,
    LHipYawPitch, LHipRoll, LHipPitch, LKneePitch, LAnklePitch, LAnkleRoll,
    RHipYawPitch, RHipRoll, RHipPitch, RKneePitch, RAnklePitch, RAnkleRoll};

class NaoDriverNode : public rclcpp::Node
{
public:
  NaoDriverNode()
      : Node("nao_driver_node")
  {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGSEGV, signal_handler);
    signal(SIGKILL, signal_handler);
    
    instance_ = this;

    publisher_ = this->create_publisher<nao_interfaces::msg::NaoSensor>("nao_sensor", 10);

    // Create Subscription
    subscription_ = this->create_subscription<nao_interfaces::msg::NaoActuator>(
        "nao_actuator",
        10,
        std::bind(&NaoDriverNode::topic_callback, this, _1));

    socket_thread_ = std::thread([this]()
                                 { this->socket_loop(); });

    // Start Information
    RCLCPP_INFO(this->get_logger(), "NaoDriverNode started.");
  }

  ~NaoDriverNode()
  {
    exit_flag_ = true;
    if (socket_thread_.joinable())
    {
      socket_thread_.join();
    }
    close(socket_fd_);
    RCLCPP_INFO(this->get_logger(), "NaoDriverNode destroyed.");
  }
  static void signal_handler(int signum)
  {
    if (instance_)
    {
      RCLCPP_WARN(instance_->get_logger(), "Received %s. Exiting...", strsignal(signum));
    }
    rclcpp::shutdown();
  }

private:
  enum
  {
    NJoint = 25
  };
  enum
  {
    NEar = 10
  };
  enum
  {
    NChest = 3
  };
  enum
  {
    NEye = 24
  };
  enum
  {
    NFoot = 3
  };
  enum
  {
    NSkull = 12
  };
  enum
  {
    NSonar = 2
  };
  enum
  {
    MAX_LOSS = 10
  };
  // Data received by LoLA client (This program sent this data to LoLA).
  struct LoLAReceivedData
  {
    std::array<float, NJoint> Position;
    std::array<float, NJoint> Stiffness; // [0, 1]
    std::array<float, NEar> REar;
    std::array<float, NEar> LEar;
    std::array<float, NChest> Chest;
    std::array<float, NEye> LEye;
    std::array<float, NEye> REye;
    std::array<float, NFoot> LFoot;
    std::array<float, NFoot> RFoot;
    std::array<float, NSkull> Skull;
    std::array<bool, NSonar> Sonar;
  } pack;

private:
  void disable_stiffness_exit()
  {
    pack.Stiffness.fill(0.f);
    json send_frame;
    send_frame["Stiffness"] = pack.Stiffness;
    std::vector<uint8_t> sbuffer = json::to_msgpack(send_frame);
    send(socket_fd_, reinterpret_cast<char *>(sbuffer.data()), sbuffer.size(), 0);
  }
  void disable_stiffness()
  {
    pack.Stiffness.fill(0.f);

    json send_frame;
    send_frame["Stiffness"] = pack.Stiffness;
    std::vector<uint8_t> sbuffer = json::to_msgpack(send_frame);

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      while (!message_queue_.empty())
      {
        message_queue_.pop();
      }
      message_queue_.push(sbuffer);
    }
  }
  void socket_loop()
  {
    socket_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd_ < 0)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to create socket");
      return;
    }

    sockaddr_un address;
    address.sun_family = AF_UNIX;
    strcpy(address.sun_path, "/tmp/robocup");

    connect(socket_fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address));

    constexpr int LOLAMSGLEN = 896;
    uint8_t receivedPacket[LOLAMSGLEN];
    nlohmann::json j;
    LoLAMessage lola_message;
    auto message = nao_interfaces::msg::NaoSensor();
    while (!exit_flag_ && rclcpp::ok())
    {
      const size_t bytes_received = recv(socket_fd_, receivedPacket, LOLAMSGLEN, 0);
      if (bytes_received == LOLAMSGLEN)
      {
        j.clear();
        j = nlohmann::json::from_msgpack(receivedPacket);
        lola_message = j.get<LoLAMessage>();
        for (size_t i = 0; i < message.position.size(); i++)
        {
          message.position[i] = lola_message.Position[jointMappings[i]];
          message.stiffness[i] = lola_message.Stiffness[jointMappings[i]];
        }
        message.acc = lola_message.Accelerometer;
        message.gyro = lola_message.Gyroscope;
        publisher_->publish(message);
      }
      else if (bytes_received == 0)
      {
        RCLCPP_WARN(this->get_logger(), "Connection closed by peer");
        break;
      }
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Failed to receive data");
        break;
      }

      std::vector<uint8_t> sbuffer;
      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!message_queue_.empty())
        {
          lost_frames_ = 0;
          sbuffer = message_queue_.front();
          message_queue_.pop();
        }
      }

      if (!sbuffer.empty())
      {
        ssize_t bytes_sent = send(socket_fd_, reinterpret_cast<char *>(sbuffer.data()), sbuffer.size(), 0);
        if (bytes_sent < 0)
        {
          RCLCPP_ERROR(this->get_logger(), "Failed to send data");
        }
        else if (static_cast<size_t>(bytes_sent) != sbuffer.size())
        {
          RCLCPP_WARN(this->get_logger(), "Partial data sent");
        }
      }
      if (lost_frames_++ > MAX_LOSS)
      {
        lost_frames_ = MAX_LOSS;
        disable_stiffness();
      }
    }

    if (exit_flag_)
    {
      disable_stiffness_exit();
    }

    close(socket_fd_);
  }
  void topic_callback(const nao_interfaces::msg::NaoActuator &msg)
  {
    send_frame_.clear();
    std::array<float, NJoint> position;
    std::array<float, NJoint> stiffness;
    for (int i = 0; i < Joints::NUM_OF_JOINTS; i++)
    {
      position[jointMappings[i]] = joint_limits_.limits[i].clamped(msg.position[i]);
      stiffness[jointMappings[i]] = msg.stiffness[i];
    }
    send_frame_["Position"] = position;
    send_frame_["Stiffness"] = stiffness;
    std::vector<uint8_t> sbuffer = json::to_msgpack(send_frame_);

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      while (!message_queue_.empty())
      {
        message_queue_.pop();
      }
      message_queue_.push(sbuffer);
    }
  }

  rclcpp::Publisher<nao_interfaces::msg::NaoSensor>::SharedPtr publisher_;
  rclcpp::Subscription<nao_interfaces::msg::NaoActuator>::SharedPtr subscription_;
  std::thread socket_thread_;
  int socket_fd_;
  std::atomic<bool> exit_flag_{false};
  std::queue<std::vector<uint8_t>> message_queue_;
  std::mutex queue_mutex_;
  json send_frame_;
  int lost_frames_ = 0;
  JointLimits joint_limits_;

public:
  static NaoDriverNode *instance_;
};

NaoDriverNode *NaoDriverNode::instance_ = nullptr;

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<NaoDriverNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
