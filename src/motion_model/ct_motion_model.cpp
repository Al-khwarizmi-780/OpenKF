#include "ct_motion_model.h"

namespace kf
{
namespace motionmodel
{

Vector<DIM_X_CT> CtMotionModel::f(Vector<DIM_X_CT> const& vecX,
                                  float32_t dt) const
{
  // State transition model for constant turn rate (CT) motion model
  // [ pos_x ]   [ pos_x + vel * T * cos(theta) ]
  // [ pos_y ] = [ pos_y + vel * T * sin(theta) ]
  // [   vel ]   [ vel ]
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

Matrix<DIM_X_CT, DIM_X_CT> CtMotionModel::getProcessNoiseCov(
    Vector<DIM_X_CT> const& vecX, float32_t dt) const
{
  Matrix<DIM_X_CT, DIM_X_CT> matQ{Matrix<DIM_X_CT, DIM_X_CT>::Zero()};

  float32_t const sigma_accel2{m_processNoiseVec[0] * m_processNoiseVec[0]};
  float32_t const sigma_alpha2{m_processNoiseVec[1] * m_processNoiseVec[1]};

  float32_t const dt2{dt * dt};
  float32_t const dt3{dt2 * dt};

  float32_t const dt2_div2{dt2 / 2.0F};
  float32_t const dt3_div3{dt3 / 3.0F};

  matQ(IDX_PX, IDX_PX) = sigma_accel2 * dt3_div3;
  matQ(IDX_PX, IDX_PY) = 0.0F;
  matQ(IDX_PX, IDX_V) = sigma_accel2 * dt2_div2 * cosf(vecX[IDX_THETA]);
  matQ(IDX_PX, IDX_THETA) = 0.0F;
  matQ(IDX_PX, IDX_OMEGA) = 0.0F;

  matQ(IDX_PY, IDX_PX) = 0.0F;
  matQ(IDX_PY, IDX_PY) = sigma_accel2 * dt3_div3;
  matQ(IDX_PY, IDX_V) = sigma_accel2 * dt2_div2 * sinf(vecX[IDX_THETA]);
  matQ(IDX_PY, IDX_THETA) = 0.0F;
  matQ(IDX_PY, IDX_OMEGA) = 0.0F;

  matQ(IDX_V, IDX_PX) = sigma_accel2 * dt2_div2 * cosf(vecX[IDX_THETA]);
  matQ(IDX_V, IDX_PY) = sigma_accel2 * dt2_div2 * sinf(vecX[IDX_THETA]);
  matQ(IDX_V, IDX_V) = sigma_accel2 * dt;
  matQ(IDX_V, IDX_THETA) = 0.0F;
  matQ(IDX_V, IDX_OMEGA) = 0.0F;

  matQ(IDX_THETA, IDX_PX) = 0.0F;
  matQ(IDX_THETA, IDX_PY) = 0.0F;
  matQ(IDX_THETA, IDX_V) = 0.0F;
  matQ(IDX_THETA, IDX_THETA) = sigma_alpha2 * dt3_div3;
  matQ(IDX_THETA, IDX_OMEGA) = sigma_alpha2 * dt2_div2;

  matQ(IDX_OMEGA, IDX_PX) = 0.0F;
  matQ(IDX_OMEGA, IDX_PY) = 0.0F;
  matQ(IDX_OMEGA, IDX_V) = 0.0F;
  matQ(IDX_OMEGA, IDX_THETA) = sigma_alpha2 * dt2_div2;
  matQ(IDX_OMEGA, IDX_OMEGA) = sigma_alpha2 * dt;

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