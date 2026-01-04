#include "cv_motion_model.h"

namespace kf
{
namespace motionmodel
{

Vector<DIM_X_CV> CvMotionModel::f(Vector<DIM_X_CV> const& vecX,
                                  float32_t dt) const
{
  Vector<DIM_X_CV> vecXPred;

  // State transition model for constant velocity (CV) motion model
  // [ pos_x ]   [ 1 0 dt 0 ] [ pos_x ]   [ q1 ]
  // [ pos_y ] = [ 0 1 0 dt ] [ pos_y ] + [ q2 ]
  // [ vel_x ]   [ 0 0 1 0  ] [ vel_x ]   [ q3 ]
  // [ vel_y ]   [ 0 0 0 1  ] [ vel_y ]   [ q4 ]

  vecXPred[IDX_PX] = vecX[IDX_PX] + vecX[IDX_VX] * dt;
  vecXPred[IDX_PY] = vecX[IDX_PY] + vecX[IDX_VY] * dt;
  vecXPred[IDX_VX] = vecX[IDX_VX];
  vecXPred[IDX_VY] = vecX[IDX_VY];

  return vecXPred;
}

Matrix<DIM_X_CV, DIM_X_CV> CvMotionModel::getProcessNoiseCov(
    Vector<DIM_X_CV> const& /*vecX*/, float32_t dt) const
{
  // Q = sigma^2*[T^4/4   0       T^3/2   0;
  //              0       T^4/4   0       T^3/2;
  //              T^3/2   0       T^2     0;
  //              0       T^3/2   0       T^2;
  //             ];

  Matrix<DIM_X_CV, DIM_X_CV> matQ;

  const float32_t sigma2 = m_processNoiseVec[0] * m_processNoiseVec[0];
  const float32_t dt2 = dt * dt;
  const float32_t dt3 = dt2 * dt;
  const float32_t dt4 = dt2 * dt2;

  matQ(IDX_PX, IDX_PX) = sigma2 * (dt4) / 4.0F;
  matQ(IDX_PX, IDX_PY) = 0.0F;
  matQ(IDX_PX, IDX_VX) = sigma2 * (dt3) / 2.0F;
  matQ(IDX_PX, IDX_VY) = 0.0F;

  matQ(IDX_PY, IDX_PX) = 0.0F;
  matQ(IDX_PY, IDX_PY) = sigma2 * (dt4) / 4.0F;
  matQ(IDX_PY, IDX_VX) = 0.0F;
  matQ(IDX_PY, IDX_VY) = sigma2 * (dt3) / 2.0F;

  matQ(IDX_VX, IDX_PX) = sigma2 * (dt3) / 2.0F;
  matQ(IDX_VX, IDX_PY) = 0.0F;
  matQ(IDX_VX, IDX_VX) = sigma2 * (dt2);
  matQ(IDX_VX, IDX_VY) = 0.0F;

  matQ(IDX_VY, IDX_PX) = 0.0F;
  matQ(IDX_VY, IDX_PY) = sigma2 * (dt3) / 2.0F;
  matQ(IDX_VY, IDX_VX) = 0.0F;
  matQ(IDX_VY, IDX_VY) = sigma2 * (dt2);

  return matQ;
}

Matrix<DIM_X_CV, DIM_X_CV> CvMotionModel::getJacobianFk(
    Vector<DIM_X_CV> const& vecX, float32_t dt) const
{
  // State transition model for constant velocity (CV) motion model
  // [ pos_x ]   [ 1 0 dt 0 ] [ pos_x ]
  // [ pos_y ] = [ 0 1 0 dt ] [ pos_y ]
  // [ vel_x ]   [ 0 0 1 0  ] [ vel_x ]
  // [ vel_y ]   [ 0 0 0 1  ] [ vel_y ]

  Matrix<DIM_X_CV, DIM_X_CV> matFk;

  matFk(IDX_PX, IDX_PX) = 1.0F;
  matFk(IDX_PX, IDX_PY) = 0.0F;
  matFk(IDX_PX, IDX_VX) = dt;
  matFk(IDX_PX, IDX_VY) = 0.0F;

  matFk(IDX_PY, IDX_PX) = 0.0F;
  matFk(IDX_PY, IDX_PY) = 1.0F;
  matFk(IDX_PY, IDX_VX) = 0.0F;
  matFk(IDX_PY, IDX_VY) = dt;

  matFk(IDX_VX, IDX_PX) = 0.0F;
  matFk(IDX_VX, IDX_PY) = 0.0F;
  matFk(IDX_VX, IDX_VX) = 1.0F;
  matFk(IDX_VX, IDX_VY) = 0.0F;

  matFk(IDX_VY, IDX_PX) = 0.0F;
  matFk(IDX_VY, IDX_PY) = 0.0F;
  matFk(IDX_VY, IDX_VX) = 0.0F;
  matFk(IDX_VY, IDX_VY) = 1.0F;

  return matFk;
}

}  // namespace motionmodel
}  // namespace kf