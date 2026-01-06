#ifndef OPENKF_POS2D_MEAS_MODEL_H
#define OPENKF_POS2D_MEAS_MODEL_H

#include "meas_model.h"
#include "types.h"

namespace kf
{
namespace measmodel
{
/// @brief Measurement space dimension for 2D position measurement model
/// \vec{z}=[pos_x, pos_y]^T
static constexpr int32_t DIM_Z_POS2D{2};

/// @brief  measurement model
/// @tparam DIM_X State space vector dimension \vec{x}=[pos_x, pos_y, ...]^T
template <int32_t DIM_X>
class Pos2dMeasModel
    : public MeasModel<Pos2dMeasModel<DIM_X>, DIM_X, DIM_Z_POS2D>
{
 public:
  Pos2dMeasModel(Vector<2> const& sensPos, float32_t const posSigma = 1.0F)
      : m_sensPos{sensPos}, m_posSigma{posSigma}
  {
  }
  ~Pos2dMeasModel() {}

  static constexpr int32_t IDX_X_PX{0};  //< Index for position x in state
  static constexpr int32_t IDX_X_PY{1};  //< Index for position y in state

  static constexpr int32_t IDX_Z_PX{0};  //< Index for position x in measurement
  static constexpr int32_t IDX_Z_PY{1};  //< Index for position y in measurement

  /// @brief Measurement model function that maps the state space vector to
  /// the measurement space.
  /// @param vecX State space vector \vec{x}
  /// @return Measurement space vector \vec{z}
  Vector<DIM_Z_POS2D> h(Vector<DIM_X> const& vecX) const
  {
    Vector<DIM_Z_POS2D> vecZ;
    vecZ(IDX_Z_PX) = vecX(IDX_X_PX) - m_sensPos(0);
    vecZ(IDX_Z_PY) = vecX(IDX_X_PY) - m_sensPos(1);
    return vecZ;
  }

  /// @brief Method that calculates the jacobians of the measurement model.
  /// @param vecX State Space vector \vec{x}
  /// @return The jacobians of the measurement model.
  Matrix<DIM_Z_POS2D, DIM_X> getJacobianHk(Vector<DIM_X> const& vecX) const
  {
    // H = [1 0 0 0 0;
    //      0 1 0 0 0];
    Matrix<DIM_Z_POS2D, DIM_X> matH;
    matH.setZero();
    matH(IDX_Z_PX, IDX_X_PX) = 1.0f;
    matH(IDX_Z_PY, IDX_X_PY) = 1.0f;
    return matH;
  }

  /// @brief Get the measurement noise covariance R
  /// @return The measurement noise covariance R
  Matrix<DIM_Z_POS2D, DIM_Z_POS2D> getMeasurementNoiseCov() const
  {
    float32_t const posNoiseSigma2{m_posSigma * m_posSigma};
    Matrix<DIM_Z_POS2D, DIM_Z_POS2D> matR;
    matR.setZero();
    matR(IDX_Z_PX, IDX_Z_PX) = posNoiseSigma2;
    matR(IDX_Z_PY, IDX_Z_PY) = posNoiseSigma2;
    return matR;
  }

  /// @brief Set position noise standard deviation
  /// @param sigma Position noise standard deviation
  void setPosSigma(float32_t const sigma) { m_posSigma = sigma; }

 private:
  Vector<2> m_sensPos;   //< Sensor position in 2D space
  float32_t m_posSigma;  //< Measurement noise vector
};
}  // namespace measmodel

}  // namespace kf
#endif  // OPENKF_POS2D_MEAS_MODEL_H
