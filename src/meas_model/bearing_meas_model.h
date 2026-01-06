#ifndef OPENKF_BEARING_MEAS_MODEL_H
#define OPENKF_BEARING_MEAS_MODEL_H

#include "meas_model.h"
#include "types.h"

namespace kf
{
namespace measmodel
{
/// @brief Measurement space dimension for bearing measurement
/// model
/// \vec{z}=[pos_x, pos_y]^T
static constexpr int32_t DIM_Z_BEARING{2};

template <int32_t DIM_X>
class BearingMeasModel
    : public MeasModel<BearingMeasModel<DIM_X>, DIM_X, DIM_Z_BEARING>
{
 public:
  BearingMeasModel(Vector<2> const& sensPos2D,
                   float32_t const bearingSigma = 1.0F)
      : m_sensPos2D{sensPos2D}, m_bearingNoiseSigma{bearingSigma}
  {
  }
  ~BearingMeasModel() {}

  static constexpr int32_t IDX_X_PX{0};  //< Index for position x in state
  static constexpr int32_t IDX_X_PY{1};  //< Index for position y in state

  static constexpr int32_t IDX_Z_BEARING{
      0};  //< Index for bearing in measurement

  /// @brief Measurement model function that maps the state space vector to
  /// the measurement space.
  /// @param vecX State space vector \vec{x}
  /// @return Measurement space vector \vec{z}
  Vector<DIM_Z_BEARING> h(Vector<DIM_X> const& vecX) const
  {
    Vector<DIM_Z_BEARING> vecZ;
    float32_t px = vecX(IDX_X_PX) - m_sensPos2D(IDX_X_PX);
    float32_t py = vecX(IDX_X_PY) - m_sensPos2D(IDX_X_PY);
    vecZ(IDX_Z_BEARING) = std::atan2(py, px);
    return vecZ;
  }

  /// @brief Method that calculates the jacobians of the measurement model.
  /// @param vecX State Space vector \vec{x}
  /// @return The jacobians of the measurement model.
  Matrix<DIM_Z_BEARING, DIM_X> getJacobianHk(Vector<DIM_X> const& vecX) const
  {
    Matrix<DIM_Z_BEARING, DIM_X> matH;
    matH.setZero();

    float32_t const px{vecX(IDX_X_PX) - m_sensPos2D(IDX_X_PX)};
    float32_t const py{vecX(IDX_X_PY) - m_sensPos2D(IDX_X_PY)};
    float32_t const denom{px * px + py * py};

    matH(IDX_Z_BEARING, IDX_X_PX) = -py / denom;
    matH(IDX_Z_BEARING, IDX_X_PY) = px / denom;
    return matH;
  }

  /// @brief Get the measurement noise covariance R
  /// @return The measurement noise covariance R
  Matrix<DIM_Z_BEARING, DIM_Z_BEARING> getMeasurementNoiseCov() const
  {
    float32_t const bearingNoiseSigma2{m_bearingNoiseSigma *
                                       m_bearingNoiseSigma};
    Matrix<DIM_Z_BEARING, DIM_Z_BEARING> matR;
    matR.setZero();
    matR(IDX_Z_BEARING, IDX_Z_BEARING) = bearingNoiseSigma2;
    return matR;
  }

  /// @brief Set bearing noise standard deviation
  /// @param sigma Bearing noise standard deviation
  void setBearingSigma(float32_t const sigma) { m_bearingNoiseSigma = sigma; }

 private:
  Vector<2> m_sensPos2D;          //< Sensor position in 2D
  float32_t m_bearingNoiseSigma;  //< Bearing measurement noise vector
};
}  // namespace measmodel

}  // namespace kf
#endif  // OPENKF_BEARING_MEAS_MODEL_H
