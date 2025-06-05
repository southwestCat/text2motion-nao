#pragma once

namespace SensorData {
constexpr float off = 10000.f;  /**< Special value that indicates the sensor is tuned off. */
constexpr float ignore = 20000.f;  /**< Special value that indicates the sensor uses the previous value. */
}  // namespace SensorData