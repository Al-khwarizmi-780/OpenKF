#include "ca_motion_model.h"

namespace kf
{
namespace motionmodel
{

Vector<DIM_X_CA> CaMotionModel::f(Vector<DIM_X_CA> const& vecX,
                                  float32_t dt) const
{
  // State transition model for constant acceleration (CA) motion model
  // [ pos_x ]   [ 1 0 dt  0 dt^2/2      0 ] [ pos_x ]   [ q1 ]
  // [ pos_y ] = [ 0 1  0 dt      0 dt^2/2 ] [ pos_y ] + [ q2 ]
  // [ vel_x ]   [ 0 0  1  0     dt      0 ] [ vel_x ]   [ q3 ]
  // [ vel_y ]   [ 0 0  0  1     0      dt ] [ vel_y ]   [ q4 ]
  // [ acc_x ]   [ 0 0  0  0     1       0 ] [ acc_x ]   [ q5 ]
  // [ acc_y ]   [ 0 0  0  0     0       1 ] [ acc_y ]   [ q6 ]

  float32_t const halfDeltaT2{dt * dt / 2.0F};

  Vector<DIM_X_CA> vecXPred;
  vecXPred[IDX_PX] =
      vecX[IDX_PX] + vecX[IDX_VX] * dt + vecX[IDX_AX] * halfDeltaT2;
  vecXPred[IDX_PY] =
      vecX[IDX_PY] + vecX[IDX_VY] * dt + vecX[IDX_AY] * halfDeltaT2;
  vecXPred[IDX_VX] = vecX[IDX_VX] + vecX[IDX_AX] * dt;
  vecXPred[IDX_VY] = vecX[IDX_VY] + vecX[IDX_AY] * dt;
  vecXPred[IDX_AX] = vecX[IDX_AX];
  vecXPred[IDX_AY] = vecX[IDX_AY];

  return vecXPred;
}

Matrix<DIM_X_CA, DIM_X_CA> CaMotionModel::getProcessNoiseCov(
    Vector<DIM_X_CA> const& /*vecX*/, float32_t dt) const
{
  // Q = sigma^2*[T^5/20          0     T^4/8       0   T^3/6       0;
  //                   0     T^5/20         0   T^4/8       0   T^3/6;
  //               T^4/8          0     T^3/3       0   T^2/2       0;
  //                   0      T^4/8         0   T^2/2       0   T^2/2;
  //               T^3/6          0     T^2/2       0     T         0;
  //                   0      T^3/6         0   T^2/2       0       T;
  //             ];

  Matrix<DIM_X_CA, DIM_X_CA> matQ;

  const float32_t sigma2{m_processNoiseVec[0] * m_processNoiseVec[0]};
  const float32_t dt2{dt * dt};
  const float32_t dt3{dt2 * dt};
  const float32_t dt4{dt2 * dt2};
  const float32_t dt5{dt4 * dt};

  matQ(IDX_PX, IDX_PX) = sigma2 * dt5 / 20.0F;
  matQ(IDX_PX, IDX_PY) = 0.0F;
  matQ(IDX_PX, IDX_VX) = sigma2 * dt4 / 8.0F;
  matQ(IDX_PX, IDX_VY) = 0.0F;
  matQ(IDX_PX, IDX_AX) = sigma2 * dt3 / 6.0F;
  matQ(IDX_PX, IDX_AY) = 0.0F;

  matQ(IDX_PY, IDX_PX) = 0.0F;
  matQ(IDX_PY, IDX_PY) = sigma2 * dt5 / 20.0F;
  matQ(IDX_PY, IDX_VX) = 0.0F;
  matQ(IDX_PY, IDX_VY) = sigma2 * dt4 / 8.0F;
  matQ(IDX_PY, IDX_AX) = 0.0F;
  matQ(IDX_PY, IDX_AY) = sigma2 * dt3 / 6.0F;

  matQ(IDX_VX, IDX_PX) = sigma2 * dt4 / 8.0F;
  matQ(IDX_VX, IDX_PY) = 0.0F;
  matQ(IDX_VX, IDX_VX) = sigma2 * dt3 / 3.0F;
  matQ(IDX_VX, IDX_VY) = 0.0F;
  matQ(IDX_VX, IDX_AX) = sigma2 * dt2 / 2.0F;
  matQ(IDX_VX, IDX_AY) = 0.0F;

  matQ(IDX_VY, IDX_PX) = 0.0F;
  matQ(IDX_VY, IDX_PY) = sigma2 * dt4 / 8.0F;
  matQ(IDX_VY, IDX_VX) = 0.0F;
  matQ(IDX_VY, IDX_VY) = sigma2 * dt2 / 2.0F;
  matQ(IDX_VY, IDX_AX) = 0.0F;
  matQ(IDX_VY, IDX_AY) = sigma2 * dt2 / 2.0F;

  matQ(IDX_AX, IDX_PX) = sigma2 * dt3 / 6.0F;
  matQ(IDX_AX, IDX_PY) = 0.0F;
  matQ(IDX_AX, IDX_VX) = sigma2 * dt2 / 2.0F;
  matQ(IDX_AX, IDX_VY) = 0.0F;
  matQ(IDX_AX, IDX_AX) = sigma2 * dt;
  matQ(IDX_AX, IDX_AY) = 0.0F;

  matQ(IDX_AY, IDX_PX) = 0.0F;
  matQ(IDX_AY, IDX_PY) = sigma2 * dt3 / 6.0F;
  matQ(IDX_AY, IDX_VX) = 0.0F;
  matQ(IDX_AY, IDX_VY) = sigma2 * dt2 / 2.0F;
  matQ(IDX_AY, IDX_AX) = 0.0F;
  matQ(IDX_AY, IDX_AY) = sigma2 * dt;

  return matQ;
}

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

  float32_t const halfdt2{0.5F * dt * dt};

  Matrix<DIM_X_CA, DIM_X_CA> matFk;
  matFk(IDX_PX, IDX_PX) = 1.0F;
  matFk(IDX_PX, IDX_PY) = 0.0F;
  matFk(IDX_PX, IDX_VX) = dt;
  matFk(IDX_PX, IDX_VY) = 0.0F;
  matFk(IDX_PX, IDX_AX) = halfdt2;
  matFk(IDX_PX, IDX_AY) = 0.0F;

  matFk(IDX_PY, IDX_PX) = 0.0F;
  matFk(IDX_PY, IDX_PY) = 1.0F;
  matFk(IDX_PY, IDX_VX) = 0.0F;
  matFk(IDX_PY, IDX_VY) = dt;
  matFk(IDX_PY, IDX_AX) = 0.0F;
  matFk(IDX_PY, IDX_AY) = halfdt2;

  matFk(IDX_VX, IDX_PX) = 0.0F;
  matFk(IDX_VX, IDX_PY) = 0.0F;
  matFk(IDX_VX, IDX_VX) = 1.0F;
  matFk(IDX_VX, IDX_VY) = 0.0F;
  matFk(IDX_VX, IDX_AX) = dt;
  matFk(IDX_VX, IDX_AY) = 0.0F;

  matFk(IDX_VY, IDX_PX) = 0.0F;
  matFk(IDX_VY, IDX_PY) = 0.0F;
  matFk(IDX_VY, IDX_VX) = 0.0F;
  matFk(IDX_VY, IDX_VY) = 1.0F;
  matFk(IDX_VY, IDX_AX) = 0.0F;
  matFk(IDX_VY, IDX_AY) = dt;

  matFk(IDX_AX, IDX_PX) = 0.0F;
  matFk(IDX_AX, IDX_PY) = 0.0F;
  matFk(IDX_AX, IDX_VX) = 0.0F;
  matFk(IDX_AX, IDX_VY) = 0.0F;
  matFk(IDX_AX, IDX_AX) = 1.0F;
  matFk(IDX_AX, IDX_AY) = 0.0F;

  matFk(IDX_AY, IDX_PX) = 0.0F;
  matFk(IDX_AY, IDX_PY) = 0.0F;
  matFk(IDX_AY, IDX_VX) = 0.0F;
  matFk(IDX_AY, IDX_VY) = 0.0F;
  matFk(IDX_AY, IDX_AX) = 0.0F;
  matFk(IDX_AY, IDX_AY) = 1.0F;

  return matFk;
}

}  // namespace motionmodel
}  // namespace kf