#ifndef OPENKF_MEAS_MODEL_H
#define OPENKF_MEAS_MODEL_H

#include "types.h"

namespace kf
{

namespace measmodel
{
/// @brief Base class for measurement models used by kalman filters
/// @tparam DIM_X State space vector dimension
/// @tparam DIM_Z Measurement space vector dimension
template <class Derived, int32_t DIM_X, int32_t DIM_Z>
class MeasModel
{
 public:
  /// @brief Get the measurement space vector dimension
  /// @return The measurement space vector dimension
  int32_t getDimZ() const { return DIM_Z; }

  /// @brief Measurement model function that maps the state space vector to
  /// the measurement space.
  /// @param vecX State space vector \vec{x}
  /// @return Measurement space vector \vec{z}
  Vector<DIM_Z> h(Vector<DIM_X> const& vecX) const
  {
    return static_cast<Derived const*>(this)->h(vecX);
  }

  /// @brief Method that calculates the jacobians of the measurement model.
  /// @param vecX State Space vector \vec{x}
  /// @return The jacobians of the measurement model.
  Matrix<DIM_Z, DIM_X> getJacobianHk(Vector<DIM_X> const& vecX) const
  {
    return static_cast<Derived const*>(this)->getJacobianHk(vecX);
  }

  /// @brief Get the measurement noise covariance R
  /// @return The measurement noise covariance R
  Matrix<DIM_Z, DIM_Z> getMeasurementNoiseCov() const
  {
    return static_cast<Derived const*>(this)->getMeasurementNoiseCov();
  }
};
}  // namespace measmodel

}  // namespace kf

#endif  // OPENKF_MEAS_MODEL_H
