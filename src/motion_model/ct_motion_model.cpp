#include "ct_motion_model.h"

namespace kf
{
namespace motionmodel
{

Vector<DIM_X_CT> CtMotionModel::f(Vector<DIM_X_CT> const& vecX,
                                  float32_t dt) const
{
  // State transition model for constant turn rate (CT) motion model
  // [ pos_x ]   [ pos_x + vel_x * T * cos(theta) ]
  // [ pos_y ] = [ pos_y + vel_x * T * sin(theta) ]
  // [   vel ]   [ vel_x ]
  // [ theta ]   [ theta + omega * T ]
  // [ omega ]   [ omega ]

  float32_t const displacement{vecX[IDX_V] * dt};

  Vector<DIM_X_CT> vecXPred;
  vecXPred[IDX_PX] = vecX[IDX_PX] + (displacement * cosf(vecX[IDX_THETA]));
  vecXPred[IDX_PY] = vecX[IDX_PY] + (displacement * sinf(vecX[IDX_THETA]));
  vecXPred[IDX_V] = vecX[IDX_V];
  vecXPred[IDX_THETA] = vecX[IDX_THETA] + (vecX[IDX_OMEGA] * dt);
  vecXPred[IDX_OMEGA] = vecX[IDX_OMEGA];

  return vecXPred;
}

Matrix<DIM_X_CT, DIM_X_CT> CtMotionModel::getProcessNoiseCov(float32_t sigma,
                                                             float32_t dt) const
{
  Matrix<DIM_X_CT, DIM_X_CT> matQ{Matrix<DIM_X_CT, DIM_X_CT>::Zero()};

  float32_t const sigma2{sigma * sigma};

  matQ(IDX_V, IDX_V) = sigma2;
  matQ(IDX_OMEGA, IDX_OMEGA) = sigma2;

  return matQ;
}

Matrix<DIM_X_CT, DIM_X_CT> CtMotionModel::getJacobianFk(
    Vector<DIM_X_CT> const& vecX, float32_t dt) const
{
  // State transition model for constant turn rate (CT) motion model
  // [ pos_x ]   [ 1 0  T*cos(theta)  -T*sin(theta)      0] [ pos_x ]
  // [ pos_y ] = [ 0 1  T*sin(theta)   T*cos(theta)      0] [ pos_y ]
  // [   vel ]   [ 0 0             1              0      0] [ vel_x ]
  // [ theta ]   [ 0 0             0              1      T] [ theta ]
  // [ omega ]   [ 0 0             0              0      1] [ omega ]

  float32_t const halfdt2{0.5F * dt * dt};

  Matrix<DIM_X_CT, DIM_X_CT> matFk;
  matFk(IDX_PX, IDX_PX) = 1.0F;
  matFk(IDX_PX, IDX_PY) = 0.0F;
  matFk(IDX_PX, IDX_V) = dt * cosf(vecX[IDX_THETA]);
  matFk(IDX_PX, IDX_THETA) = -dt * sinf(vecX[IDX_THETA]);
  matFk(IDX_PX, IDX_OMEGA) = 0.0F;

  matFk(IDX_PY, IDX_PX) = 0.0F;
  matFk(IDX_PY, IDX_PY) = 1.0F;
  matFk(IDX_PY, IDX_V) = dt * sinf(vecX[IDX_THETA]);
  matFk(IDX_PY, IDX_THETA) = dt * cosf(vecX[IDX_THETA]);
  matFk(IDX_PY, IDX_OMEGA) = 0.0F;

  matFk(IDX_V, IDX_PX) = 0.0F;
  matFk(IDX_V, IDX_PY) = 0.0F;
  matFk(IDX_V, IDX_V) = 1.0F;
  matFk(IDX_V, IDX_THETA) = 0.0F;
  matFk(IDX_V, IDX_OMEGA) = 0.0F;

  matFk(IDX_THETA, IDX_PX) = 0.0F;
  matFk(IDX_THETA, IDX_PY) = 0.0F;
  matFk(IDX_THETA, IDX_V) = 0.0F;
  matFk(IDX_THETA, IDX_THETA) = 1.0F;
  matFk(IDX_THETA, IDX_OMEGA) = dt;

  matFk(IDX_OMEGA, IDX_PX) = 0.0F;
  matFk(IDX_OMEGA, IDX_PY) = 0.0F;
  matFk(IDX_OMEGA, IDX_V) = 0.0F;
  matFk(IDX_OMEGA, IDX_THETA) = 0.0F;
  matFk(IDX_OMEGA, IDX_OMEGA) = 1.0F;

  return matFk;
}

}  // namespace motionmodel
}  // namespace kf