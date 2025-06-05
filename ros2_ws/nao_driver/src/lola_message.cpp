#include "lola_message.h"

void to_json(nlohmann::json &j, const LoLAMessage &l) {
  j = nlohmann::json{
    {"Accelerometer", l.Accelerometer},
    {"Angles", l.Angles},
    {"Battery", l.Battery},
    {"Current", l.Current},
    {"FSR", l.FSR},
    {"Gyroscope", l.Gyroscope},
    {"Position", l.Position},
    {"RobotConfig", l.RobotConfig},
    {"Sonar", l.Sonar},
    {"Status", l.Status},
    {"Stiffness", l.Stiffness},
    {"Temperature", l.Temperature},
    {"Touch", l.Touch}
  };
}

void from_json(const nlohmann::json& j, LoLAMessage& l) {
  j.at("Accelerometer").get_to(l.Accelerometer);
  j.at("Angles").get_to(l.Angles);
  j.at("Battery").get_to(l.Battery);
  j.at("Current").get_to(l.Current);
  j.at("FSR").get_to(l.FSR);
  j.at("Gyroscope").get_to(l.Gyroscope);
  j.at("Position").get_to(l.Position);
  j.at("RobotConfig").get_to(l.RobotConfig);
  j.at("Sonar").get_to(l.Sonar);
  j.at("Status").get_to(l.Status);
  j.at("Stiffness").get_to(l.Stiffness);
  j.at("Temperature").get_to(l.Temperature);
  j.at("Touch").get_to(l.Touch);
}
