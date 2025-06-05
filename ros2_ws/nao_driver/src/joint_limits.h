#pragma once

#include <array>
#include "joint_angles.h"
#include <nao_interfaces/msg/joints.hpp>

class JointLimits
{
public:
    JointLimits()
    {
        limits.fill(Rangef(0_deg));
        limits[nao_interfaces::msg::Joints::HEAD_YAW] = Rangef(-119.5_deg, 119.5_deg);
        limits[nao_interfaces::msg::Joints::HEAD_PITCH] = Rangef(-38.5_deg, 29.5_deg);
        limits[nao_interfaces::msg::Joints::L_SHOULDER_PITCH] = Rangef(-119.5_deg, 119.5_deg);
        limits[nao_interfaces::msg::Joints::L_SHOULDER_ROLL] = Rangef(-18_deg, 76_deg);
        limits[nao_interfaces::msg::Joints::L_ELBOW_YAW] = Rangef(-119.5_deg, 119.5_deg);
        limits[nao_interfaces::msg::Joints::L_ELBOW_ROLL] = Rangef(-88.5_deg, -2_deg);
        limits[nao_interfaces::msg::Joints::L_WRIST_YAW] = Rangef(-104.5_deg, 104.5_deg);
        // limits[Joints::lHand] = Rangef(0_deg, 57.2958_deg);
        limits[nao_interfaces::msg::Joints::R_SHOULDER_PITCH] = Rangef(-119.5_deg, 119.5_deg);
        limits[nao_interfaces::msg::Joints::R_SHOULDER_ROLL] = Rangef(-76_deg, 18_deg);
        limits[nao_interfaces::msg::Joints::R_ELBOW_YAW] = Rangef(-119.5_deg, 119.5_deg);
        limits[nao_interfaces::msg::Joints::R_ELBOW_ROLL] = Rangef(2_deg, 88.5_deg);
        limits[nao_interfaces::msg::Joints::R_WRIST_YAW] = Rangef(-104.5_deg, 104.5_deg);
        // limits[Joints::rHand] = Rangef(0_deg, 57.2958_deg);
        limits[nao_interfaces::msg::Joints::L_HIP_YAW_PITCH] = Rangef(-65.62_deg, 42.44_deg);
        limits[nao_interfaces::msg::Joints::L_HIP_ROLL] = Rangef(-21.74_deg, 45.29_deg);
        limits[nao_interfaces::msg::Joints::L_HIP_PITCH] = Rangef(-88_deg, 27.73_deg);
        limits[nao_interfaces::msg::Joints::L_KNEE_PITCH] = Rangef(-5.29_deg, 121.04_deg);
        limits[nao_interfaces::msg::Joints::L_ANKLE_PITCH] = Rangef(-68.15_deg, 52.86_deg);
        limits[nao_interfaces::msg::Joints::L_ANKLE_ROLL] = Rangef(-22.79_deg, 44.06_deg);
        limits[nao_interfaces::msg::Joints::R_HIP_YAW_PITCH] = Rangef(-65.62_deg, 42.44_deg);
        limits[nao_interfaces::msg::Joints::R_HIP_ROLL] = Rangef(-45.29_deg, 21.74_deg);
        limits[nao_interfaces::msg::Joints::R_HIP_PITCH] = Rangef(-88_deg, 27.73_deg);
        limits[nao_interfaces::msg::Joints::R_KNEE_PITCH] = Rangef(-5.9_deg, 121.47_deg);
        limits[nao_interfaces::msg::Joints::R_ANKLE_PITCH] = Rangef(-67.97_deg, 53.4_deg);
        limits[nao_interfaces::msg::Joints::R_ANKLE_ROLL] = Rangef(-44.06_deg, 22.8_deg);
    }

    std::array<Rangef, nao_interfaces::msg::Joints::NUM_OF_JOINTS> limits;
};