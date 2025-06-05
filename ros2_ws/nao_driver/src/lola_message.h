#pragma once

#include <array>
#include "json.hpp"

// Data sent by LoLA client
struct LoLAMessage {
  std::array<float, 25> Stiffness;
  std::array<float, 25> Position;
  std::array<float, 25> Temperature;
  std::array<float, 25> Current;
  std::array<float, 4> Battery;
  std::array<float, 3> Accelerometer;
  std::array<float, 3> Gyroscope;
  std::array<float, 2> Angles;
  std::array<float, 2> Sonar;
  std::array<float, 8> FSR;
  std::array<float, 14> Touch;
  std::array<int, 25> Status;  // Temperature status
  std::array<std::string, 4> RobotConfig;
};

void to_json(nlohmann::json& j, const LoLAMessage& l);
void from_json(const nlohmann::json& j, LoLAMessage& l);