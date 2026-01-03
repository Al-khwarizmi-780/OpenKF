#include "ca_motion_model.h"

namespace kf
{
namespace motionmodel
{
/// @brief Prediction motion model function that propagate the previous state
/// to next state in time.
/// @param vecX State space vector \vec{x}
Vector<DIM_X_CA> CaMotionModel::f(Vector<DIM_X_CA> const& vecX,
                                  Vector<DIM_X_CA> const& vecQ,
                                  float32_t dt) const
{
  Vector<DIM_X_CA> vecXPred;

  float32_t const halfDeltaT2 = dt * dt / 2.0F;

  // State transition model for constant acceleration (CA) motion model
  // [ pos_x ]   [ 1 0 dt  0 dt^2/2      0 ] [ pos_x ]   [ q1 ]
  // [ pos_y ] = [ 0 1  0 dt      0 dt^2/2 ] [ pos_y ] + [ q2 ]
  // [ vel_x ]   [ 0 0  1  0     dt      0 ] [ vel_x ]   [ q3 ]
  // [ vel_y ]   [ 0 0  0  1     0      dt ] [ vel_y ]   [ q4 ]
  // [ acc_x ]   [ 0 0  0  0     1       0 ] [ acc_x ]   [ q5 ]
  // [ acc_y ]   [ 0 0  0  0     0       1 ] [ acc_y ]   [ q6 ]

  vecXPred[0] = vecX[0] + vecX[2] * dt + vecX[4] * halfDeltaT2 + vecQ[0];
  vecXPred[1] = vecX[1] + vecX[3] * dt + vecX[5] * halfDeltaT2 + vecQ[1];
  vecXPred[2] = vecX[2] + vecX[4] * dt + vecQ[2];
  vecXPred[3] = vecX[3] + vecX[5] * dt + vecQ[3];
  vecXPred[4] = vecX[4] + vecQ[4];
  vecXPred[5] = vecX[5] + vecQ[5];

  return vecXPred;
}
/// @brief Get the process noise covariance Q
/// @param sigma Standard deviation of the process noise
/// @param dt Time step between state updates (unit: seconds)
Matrix<DIM_X_CA, DIM_X_CA> CaMotionModel::getProcessNoiseCov(float32_t sigma,
                                                             float32_t dt) const
{
  // Q = sigma^2*[T^5/20          0     T^4/8       0   T^3/6       0;
  //                   0     T^5/20         0   T^4/8       0   T^3/6;
  //               T^4/8          0     T^3/3       0   T^2/2       0;
  //                   0      T^4/8         0   T^2/2       0   T^2/2;
  //               T^3/6          0     T^2/2       0     T         0;
  //                   0      T^3/6         0   T^2/2       0       T;
  //             ];

  Matrix<DIM_X_CA, DIM_X_CA> matQ;

  const float32_t sigma2 = sigma * sigma;
  const float32_t dt2 = dt * dt;
  const float32_t dt3 = dt2 * dt;
  const float32_t dt4 = dt2 * dt2;
  const float32_t dt5 = dt4 * dt;

  matQ(0, 0) = sigma2 * (dt5) / 20.0F;
  matQ(0, 1) = 0.0F;
  matQ(0, 2) = sigma2 * (dt4) / 8.0F;
  matQ(0, 3) = 0.0F;
  matQ(0, 4) = sigma2 * (dt3) / 6.0F;
  matQ(0, 5) = 0.0F;

  matQ(1, 0) = 0.0F;
  matQ(1, 1) = sigma2 * (dt5) / 20.0F;
  matQ(1, 2) = 0.0F;
  matQ(1, 3) = sigma2 * (dt4) / 8.0F;
  matQ(1, 4) = 0.0F;
  matQ(1, 5) = sigma2 * (dt3) / 6.0F;

  matQ(2, 0) = sigma2 * (dt4) / 8.0F;
  matQ(2, 1) = 0.0F;
  matQ(2, 2) = sigma2 * (dt3) / 3.0F;
  matQ(2, 3) = 0.0F;
  matQ(2, 4) = sigma2 * (dt2) / 2.0F;
  matQ(2, 5) = 0.0F;

  matQ(3, 0) = 0.0F;
  matQ(3, 1) = sigma2 * (dt4) / 8.0F;
  matQ(3, 2) = 0.0F;
  matQ(3, 3) = sigma2 * (dt2) / 2.0F;
  matQ(3, 4) = 0.0F;
  matQ(3, 5) = sigma2 * (dt2) / 2.0F;

  matQ(4, 0) = sigma2 * (dt3) / 6.0F;
  matQ(4, 1) = 0.0F;
  matQ(4, 2) = sigma2 * (dt2) / 2.0F;
  matQ(4, 3) = 0.0F;
  matQ(4, 4) = sigma2 * (dt);
  matQ(4, 5) = 0.0F;

  matQ(5, 0) = 0.0F;
  matQ(5, 1) = sigma2 * (dt3) / 6.0F;
  matQ(5, 2) = 0.0F;
  matQ(5, 3) = sigma2 * (dt2) / 2.0F;
  matQ(5, 4) = 0.0F;
  matQ(5, 5) = sigma2 * (dt);

  return matQ;
}

/// @brief Method that calculates the jacobians of the state transition model.
/// @param vecX State Space vector \vec{x}
/// @param dt Time step between state updates (unit: seconds)
Matrix<DIM_X_CA, DIM_X_CA> CaMotionModel::getJacobianFk(
    Vector<DIM_X_CA> const& vecX, float32_t dt) const
{
  // State transition model for constant acceleration (CA) motion model
  // [ pos_x ]   [ 1 0 dt   0 dt^2/2       0 ] [ pos_x ]
  // [ pos_y ] = [ 0 1  0  dt      0  dt^2/2 ] [ pos_y ]
  // [ vel_x ]   [ 0 0  1   0     dt       0 ] [ vel_x ]
  // [ vel_y ]   [ 0 0  0   1      0      dt ] [ vel_y ]
  // [ acc_x ]   [ 0 0  0   0      1       0 ] [ acc_x ]
  // [ acc_y ]   [ 0 0  0   0      0       1 ] [ acc_y ]

  float32_t const halfdt2 = 0.5F * dt * dt;

  Matrix<DIM_X_CA, DIM_X_CA> matFk;
  matFk(0, 0) = 1.0F;
  matFk(0, 1) = 0.0F;
  matFk(0, 2) = dt;
  matFk(0, 3) = 0.0F;
  matFk(0, 4) = halfdt2;
  matFk(0, 5) = 0.0F;

  matFk(1, 0) = 0.0F;
  matFk(1, 1) = 1.0F;
  matFk(1, 2) = 0.0F;
  matFk(1, 3) = dt;
  matFk(1, 4) = 0.0F;
  matFk(1, 5) = halfdt2;

  matFk(2, 0) = 0.0F;
  matFk(2, 1) = 0.0F;
  matFk(2, 2) = 1.0F;
  matFk(2, 3) = 0.0F;
  matFk(2, 4) = dt;
  matFk(2, 5) = 0.0F;

  matFk(3, 0) = 0.0F;
  matFk(3, 1) = 0.0F;
  matFk(3, 2) = 0.0F;
  matFk(3, 3) = 1.0F;
  matFk(3, 4) = 0.0F;
  matFk(3, 5) = dt;

  matFk(4, 0) = 0.0F;
  matFk(4, 1) = 0.0F;
  matFk(4, 2) = 0.0F;
  matFk(4, 3) = 0.0F;
  matFk(4, 4) = 1.0F;
  matFk(4, 5) = 0.0F;

  matFk(5, 0) = 0.0F;
  matFk(5, 1) = 0.0F;
  matFk(5, 2) = 0.0F;
  matFk(5, 3) = 0.0F;
  matFk(5, 4) = 0.0F;
  matFk(5, 5) = 1.0F;

  return matFk;
}

}  // namespace motionmodel
}  // namespace kf