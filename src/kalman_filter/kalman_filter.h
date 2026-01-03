///
/// Copyright 2022 Mohanad Youssef (Al-khwarizmi)
///
/// Use of this source code is governed by an GPL-3.0 - style
/// license that can be found in the LICENSE file or at
/// https://opensource.org/licenses/GPL-3.0
///
/// @author Mohanad Youssef <mohanad.magdy.hammad@gmail.com>
/// @file kalman_filter.h
///

#ifndef KALMAN_FILTER_LIB_H
#define KALMAN_FILTER_LIB_H

#include "motion_model/motion_model.h"
#include "types.h"
#include <Eigen/Dense>

namespace kf
{
template <int32_t DIM_X, int32_t DIM_Z>
class KalmanFilter
{
 public:
  KalmanFilter() {}

  ~KalmanFilter() {}

  virtual Vector<DIM_X>& vecX() { return m_vecX; }
  virtual const Vector<DIM_X>& vecX() const { return m_vecX; }

  virtual Matrix<DIM_X, DIM_X>& matP() { return m_matP; }
  virtual const Matrix<DIM_X, DIM_X>& matP() const { return m_matP; }

  ///
  /// @brief predict state with a linear process model.
  /// @param matF state transition matrix
  /// @param matQ process noise covariance matrix
  ///
  void predictLKF(const Matrix<DIM_X, DIM_X>& matF,
                  const Matrix<DIM_X, DIM_X>& matQ)
  {
    m_vecX = matF * m_vecX;
    m_matP = matF * m_matP * matF.transpose() + matQ;

    // Ensure symmetry of covariance matrix
    m_matP = (m_matP + m_matP.transpose()) / 2.0F;
  }

  ///
  /// @brief correct state of with a linear measurement model.
  /// @param matZ measurement vector
  /// @param matR measurement noise covariance matrix
  /// @param matH measurement transition matrix (measurement model)
  ///
  void correctLKF(const Vector<DIM_Z>& vecZ, const Matrix<DIM_Z, DIM_Z>& matR,
                  const Matrix<DIM_Z, DIM_X>& matH)
  {
    const Matrix<DIM_X, DIM_X> matI{
        Matrix<DIM_X, DIM_X>::Identity()};  // Identity matrix

    // Innovation covariance
    Matrix<DIM_Z, DIM_Z> matSk{matH * m_matP * matH.transpose() + matR};

    // Ensure innovation covariance is symmetric
    matSk = (matSk + matSk.transpose()) / 2.0F;

    const Matrix<DIM_Z, DIM_Z> matSkInv{getInvertedMatS(matSk)};
    const Matrix<DIM_X, DIM_Z> matKk{m_matP * matH.transpose() *
                                     matSkInv};  // Kalman Gain

    m_vecX = m_vecX + matKk * (vecZ - (matH * m_vecX));
    m_matP = (matI - matKk * matH) * m_matP;

    // Ensure symmetry of covariance matrix
    m_matP = (m_matP + m_matP.transpose()) / 2.0F;
  }

  ///
  /// @brief predict state with a linear process model.
  /// @param predictionModel prediction model function callback
  /// @param matJacobF state jacobian matrix
  /// @param matQ process noise covariance matrix
  ///
  template <typename PredictionModelCallback>
  void predictEkf(PredictionModelCallback predictionModelFunc,
                  const Matrix<DIM_X, DIM_X>& matJacobF,
                  const Matrix<DIM_X, DIM_X>& matQ)
  {
    m_vecX = predictionModelFunc(m_vecX);
    m_matP = matJacobF * m_matP * matJacobF.transpose() + matQ;

    // Ensure symmetry of covariance matrix
    m_matP = (m_matP + m_matP.transpose()) / 2.0F;
  }

  ///
  /// @brief predict state with a linear process model.
  /// @param motionModel prediction motion model function
  ///
  template <class Derived>
  void predictEkf(motionmodel::MotionModel<Derived, DIM_X> const& motionModel)
  {
    Matrix<DIM_X, DIM_X> const matFk{motionModel.getJacobianFk(m_vecX)};
    Matrix<DIM_X, DIM_X> const matQk{motionModel.getProcessNoiseCov(m_vecX)};
    m_vecX = motionModel.f(m_vecX);
    m_matP = matFk * m_matP * matFk.transpose() + matQk;

    // Ensure symmetry of covariance matrix
    m_matP = (m_matP + m_matP.transpose()) / 2.0F;
  }

  ///
  /// @brief predict state with a linear process model with external input.
  /// @param motionModel prediction motion model function
  /// @param vecU input vector
  ///
  template <class Derived, int32_t DIM_U>
  void predictEkf(motionmodel::MotionModelExtInput<Derived, DIM_X, DIM_U> const&
                      motionModel,
                  Vector<DIM_U> const& vecU)
  {
    Matrix<DIM_X, DIM_X> const matFk{motionModel.getJacobianFk(m_vecX, vecU)};
    Matrix<DIM_X, DIM_X> const matQk{
        motionModel.getProcessNoiseCov(m_vecX, vecU)};
    m_vecX = motionModel.f(m_vecX, vecU);
    m_matP = matFk * m_matP * matFk.transpose() + matQk;

    // Ensure symmetry of covariance matrix
    m_matP = (m_matP + m_matP.transpose()) / 2.0F;
  }

  ///
  /// @brief correct state of with a linear measurement model.
  /// @param measurementModel measurement model function callback
  /// @param matZ measurement vector
  /// @param matR measurement noise covariance matrix
  /// @param matJcobH measurement jacobian matrix
  ///
  template <typename MeasurementModelCallback>
  void correctEkf(MeasurementModelCallback measurementModelFunc,
                  const Vector<DIM_Z>& vecZ, const Matrix<DIM_Z, DIM_Z>& matR,
                  const Matrix<DIM_Z, DIM_X>& matJcobH)
  {
    const Matrix<DIM_X, DIM_X> matI{
        Matrix<DIM_X, DIM_X>::Identity()};  // Identity matrix

    // Innovation covariance
    Matrix<DIM_Z, DIM_Z> matSk{matJcobH * m_matP * matJcobH.transpose() + matR};

    // Ensure matSk is symmetric
    matSk = (matSk + matSk.transpose()) / 2.0F;

    const Matrix<DIM_Z, DIM_Z> matSkInv{getInvertedMatS(matSk)};
    const Matrix<DIM_X, DIM_Z> matKk{m_matP * matJcobH.transpose() *
                                     matSkInv};  // Kalman Gain

    m_vecX = m_vecX + matKk * (vecZ - measurementModelFunc(m_vecX));
    m_matP = (matI - matKk * matJcobH) * m_matP;

    // Ensure symmetry of covariance matrix
    m_matP = (m_matP + m_matP.transpose()) / 2.0F;
  }

 protected:
  Vector<DIM_X> m_vecX{
      Vector<DIM_X>::Zero()};  /// @brief estimated state vector
  Matrix<DIM_X, DIM_X> m_matP{
      Matrix<DIM_X, DIM_X>::Zero()};  /// @brief state covariance matrix

  ///
  /// @brief utility function to handle matrix inversion with regularization if
  /// needed.
  /// @param matA input matrix to be inverted
  /// @return inverted matrix
  ///
  Matrix<DIM_Z, DIM_Z> getInvertedMatS(const Matrix<DIM_Z, DIM_Z>& matS)
  {
    // Check if matS is invertible.
    Eigen::FullPivLU<Matrix<DIM_Z, DIM_Z>> luMatS(matS);

    if (!luMatS.isInvertible())
    {
      // Regularization (Tikhonov/Ridge): add a small value to the
      // diagonal to make it invertible.
      float lambda = 1e-6F;  // Regularization parameter
      Matrix<DIM_Z, DIM_Z> matSReg{
          matS +
          lambda * Matrix<DIM_Z, DIM_Z>::Identity(matS.rows(), matS.cols())};

      return matSReg.inverse();
    }

    return luMatS.inverse();
  }
};
}  // namespace kf

#endif  // KALMAN_FILTER_LIB_H
