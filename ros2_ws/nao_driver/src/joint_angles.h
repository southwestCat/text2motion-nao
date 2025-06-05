#pragma once

#include <array>
#include <cmath>
#include "sensor_data.h"
#include <nao_interfaces/msg/joints.hpp>

#define DEG(x) (x / 180.0 * M_PI)
#define TODEG(x) (x / M_PI * 180.0)

struct JointAngles
{
  JointAngles() { angles.fill(SensorData::off); }
  static constexpr float off = SensorData::off;
  static constexpr float ignore = SensorData::ignore;
  std::array<float, nao_interfaces::msg::Joints::NUM_OF_JOINTS> angles;
};

template <typename T>
class Range
{
public:
  constexpr Range() : min(T()), max(T()) {};
  constexpr Range(T minmax): min(minmax), max(minmax) {}
  constexpr Range(T min, T max) : min(min), max(max) {}
  constexpr T clamped(T t) const
  {
    return t < min ? min : t > max ? max
                                   : t;
  }
  T min;
  T max;
};

using Rangef = Range<float>;

class Angle
{
public:
  constexpr Angle() = default;
  constexpr Angle(float angle) : value(angle) {}

  operator float &() { return value; }
  constexpr operator const float &() const { return value; }

  constexpr Angle operator-() const { return Angle(-value); }
  Angle &operator+=(float angle)
  {
    value += angle;
    return *this;
  }
  Angle &operator-=(float angle)
  {
    value -= angle;
    return *this;
  }
  Angle &operator*=(float angle)
  {
    value *= angle;
    return *this;
  }
  Angle &operator/=(float angle)
  {
    value /= angle;
    return *this;
  }

  static constexpr Angle fromDegrees(float degrees) { return Angle((degrees / 180.f) * M_PI); }
  static constexpr Angle fromDegrees(int degrees) { return fromDegrees(static_cast<float>(degrees)); }

private:
  float value = 0.f;
};


inline constexpr Angle operator"" _deg(unsigned long long int angle)
{
    return Angle::fromDegrees(static_cast<float>(angle));
}

inline constexpr Angle operator"" _deg(long double angle)
{
    return Angle::fromDegrees(static_cast<float>(angle));
}
