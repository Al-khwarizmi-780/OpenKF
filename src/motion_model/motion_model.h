#ifndef OPENKF_MOTION_MODEL_H
#define OPENKF_MOTION_MODEL_H

#include "types.h"

namespace kf
{
namespace motionmodel
{
/// @brief Base class for motion models used by kalman filters
/// @tparam Derived Derived class which implement the interfaces
/// @tparam DIM_X State space vector dimension
template <class Derived, int32_t DIM_X>
class MotionModel
{
 public:
  /// @brief Prediction motion model function that propagate the previous state
  /// to next state in time.
  /// @param vecX State space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return Predicted/ propagated state space vector
  Vector<DIM_X> f(Vector<DIM_X> const& vecX, float32_t dt = 1.0F) const
  {
    return static_cast<Derived const*>(this)->f(vecX, dt);
  }

  /// @brief Get the process noise covariance Q
  /// @param vecX State space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The process noise covariance Q
  Matrix<DIM_X, DIM_X> getProcessNoiseCov(Vector<DIM_X> const& vecX,
                                          float32_t dt = 1.0F) const
  {
    return static_cast<Derived const*>(this)->getProcessNoiseCov(vecX, dt);
  }

  /// @brief Method that calculates the jacobians of the state transition model.
  /// @param vecX State Space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The jacobians of the state transition model.
  Matrix<DIM_X, DIM_X> getJacobianFk(Vector<DIM_X> const& vecX,
                                     float32_t dt = 1.0F) const
  {
    return static_cast<Derived const*>(this)->getJacobianFk(vecX, dt);
  }
};

/// @brief Base class for motion models with external inputs used by kalman
/// filters
/// @tparam DIM_X State space vector dimension
/// @tparam DIM_U Input space vector dimension
template <class Derived, int32_t DIM_X, int32_t DIM_U>
class MotionModelExtInput
{
 public:
  /// @brief Prediction motion model function that propagate the previous state
  /// to next state in time.
  /// @param vecX State space vector \vec{x}
  /// @param vecU Input space vector \vec{u}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return Predicted/ propagated state space vector
  Vector<DIM_X> f(Vector<DIM_X> const& vecX, Vector<DIM_U> const& vecU,
                  float32_t dt = 1.0F) const
  {
    return static_cast<Derived const*>(this)->f(vecX, vecU, dt);
  }

  /// @brief Get the process noise covariance Q
  /// @param vecX State space vector \vec{x}
  /// @param vecU Input space vector \vec{u}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The process noise covariance Q
  Matrix<DIM_X, DIM_X> getProcessNoiseCov(Vector<DIM_X> const& vecX,
                                          Vector<DIM_U> const& vecU,
                                          float32_t dt = 1.0F) const
  {
    return static_cast<Derived const*>(this)->getProcessNoiseCov(vecX, vecU,
                                                                 dt);
  }

  /// @brief Get the input noise covariance U
  /// @param vecX State space vector \vec{x}
  /// @param vecU Input space vector \vec{u}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The input noise covariance U
  Matrix<DIM_X, DIM_X> getInputNoiseCov(Vector<DIM_X> const& vecX,
                                        Vector<DIM_U> const& vecU,
                                        float32_t dt = 1.0F) const
  {
    return static_cast<Derived const*>(this)->getInputNoiseCov(vecX, vecU, dt);
  }

  /// @brief Method that calculates the jacobians of the state transition model.
  /// @param vecX State Space vector \vec{x}
  /// @param vecU Input Space vector \vec{u}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The jacobians of the state transition model.
  Matrix<DIM_X, DIM_X> getJacobianFk(Vector<DIM_X> const& vecX,
                                     Vector<DIM_U> const& vecU,
                                     float32_t dt = 1.0F) const
  {
    return static_cast<Derived const*>(this)->getJacobianFk(vecX, vecU, dt);
  }

  /// @brief Method that calculates the jacobians of the input transition model.
  /// @param vecX State Space vector \vec{x}
  /// @param vecU Input Space vector \vec{u}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The jacobians of the input transition model.
  Matrix<DIM_X, DIM_U> getJacobianBk(Vector<DIM_X> const& vecX,
                                     Vector<DIM_U> const& vecU,
                                     float32_t dt = 1.0F) const
  {
    return static_cast<Derived const*>(this)->getJacobianBk(vecX, vecU, dt);
  }
};
}  // namespace motionmodel
}  // namespace kf

#endif  // OPENKF_MOTION_MODEL_H
