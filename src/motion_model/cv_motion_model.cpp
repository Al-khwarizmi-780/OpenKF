#include "cv_motion_model.h"

namespace kf
{
namespace motionmodel
{
/// @brief Prediction motion model function that propagate the previous state
/// to next state in time.
/// @param vecX State space vector \vec{x}
Vector<DIM_X_CV> CvMotionModel::f(Vector<DIM_X_CV> const& vecX,
                                  Vector<DIM_X_CV> const& vecQ,
                                  float32_t dt) const
{
  Vector<DIM_X_CV> vecXPred;

  // State transition model for constant velocity (CV) motion model
  // [ pos_x ]   [ 1 0 dt 0 ] [ pos_x ]   [ q1 ]
  // [ pos_y ] = [ 0 1 0 dt ] [ pos_y ] + [ q2 ]
  // [ vel_x ]   [ 0 0 1 0  ] [ vel_x ]   [ q3 ]
  // [ vel_y ]   [ 0 0 0 1  ] [ vel_y ]   [ q4 ]

  vecXPred[0] = vecX[0] + vecX[2] * dt + vecQ[0];
  vecXPred[1] = vecX[1] + vecX[3] * dt + vecQ[1];
  vecXPred[2] = vecX[2] + vecQ[2];
  vecXPred[3] = vecX[3] + vecQ[3];

  return vecXPred;
}
/// @brief Get the process noise covariance Q
/// @param sigma Standard deviation of the process noise
/// @param dt Time step between state updates (unit: seconds)
Matrix<DIM_X_CV, DIM_X_CV> CvMotionModel::getProcessNoiseCov(float32_t sigma,
                                                             float32_t dt) const
{
  // Q = sigma^2*[T^4/4   0       T^3/2   0;
  //              0       T^4/4   0       T^3/2;
  //              T^3/2   0       T^2     0;
  //              0       T^3/2   0       T^2;
  //             ];

  Matrix<DIM_X_CV, DIM_X_CV> matQ;

  const float32_t sigma2 = sigma * sigma;
  const float32_t dt2 = dt * dt;
  const float32_t dt3 = dt2 * dt;
  const float32_t dt4 = dt2 * dt2;

  matQ(0, 0) = sigma2 * (dt4) / 4.0F;
  matQ(0, 1) = 0.0F;
  matQ(0, 2) = sigma2 * (dt3) / 2.0F;
  matQ(0, 3) = 0.0F;

  matQ(1, 0) = 0.0F;
  matQ(1, 1) = sigma2 * (dt4) / 4.0F;
  matQ(1, 2) = 0.0F;
  matQ(1, 3) = sigma2 * (dt3) / 2.0F;

  matQ(2, 0) = sigma2 * (dt3) / 2.0F;
  matQ(2, 1) = 0.0F;
  matQ(2, 2) = sigma2 * (dt2);
  matQ(2, 3) = 0.0F;

  matQ(3, 0) = 0.0F;
  matQ(3, 1) = sigma2 * (dt3) / 2.0F;
  matQ(3, 2) = 0.0F;
  matQ(3, 3) = sigma2 * (dt2);

  return matQ;
}

/// @brief Method that calculates the jacobians of the state transition model.
/// @param vecX State Space vector \vec{x}
/// @param dt Time step between state updates (unit: seconds)
Matrix<DIM_X_CV, DIM_X_CV> CvMotionModel::getJacobianFk(
    Vector<DIM_X_CV> const& vecX, float32_t dt) const
{
  // State transition model for constant velocity (CV) motion model
  // [ pos_x ]   [ 1 0 dt 0 ] [ pos_x ]
  // [ pos_y ] = [ 0 1 0 dt ] [ pos_y ]
  // [ vel_x ]   [ 0 0 1 0  ] [ vel_x ]
  // [ vel_y ]   [ 0 0 0 1  ] [ vel_y ]

  Matrix<DIM_X_CV, DIM_X_CV> matFk;

  matFk(0, 0) = 1.0F;
  matFk(0, 1) = 0.0F;
  matFk(0, 2) = dt;
  matFk(0, 3) = 0.0F;

  matFk(1, 0) = 0.0F;
  matFk(1, 1) = 1.0F;
  matFk(1, 2) = 0.0F;
  matFk(1, 3) = dt;

  matFk(2, 0) = 0.0F;
  matFk(2, 1) = 0.0F;
  matFk(2, 2) = 1.0F;
  matFk(2, 3) = 0.0F;

  matFk(3, 0) = 0.0F;
  matFk(3, 1) = 0.0F;
  matFk(3, 2) = 0.0F;
  matFk(3, 3) = 1.0F;

  return matFk;
}

}  // namespace motionmodel
}  // namespace kf